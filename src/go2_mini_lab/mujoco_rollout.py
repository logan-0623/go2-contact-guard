from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .controller import GaitConfig, LEGS, PDConfig, TrotGaitController, compute_pd_torque
from .trajectory import BaseState, make_trajectory


def run_mujoco_rollout(
    *,
    model_path: str | Path,
    duration: float = 6.0,
    export_dt: float = 1.0 / 60.0,
    warmup_duration: float = 0.0,
    gait_config: GaitConfig | None = None,
    pd_config: PDConfig | None = None,
    preset: str = "slow_trot",
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo dependencies are missing. Install with: pip install -e '.[sim]'"
        ) from exc

    model_path = Path(model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    controller = TrotGaitController(gait_config)
    pd = pd_config or PDConfig()

    joint_map = _find_joint_addresses(mujoco, model, controller.joint_names)
    actuator_map = _find_actuators_for_joints(mujoco, model, joint_map)
    _set_initial_pose(data, joint_map, controller.joint_targets(0.0))
    mujoco.mj_forward(model, data)

    frames: list[dict[str, Any]] = []
    warmup_duration = max(0.0, warmup_duration)
    duration = max(0.0, duration)
    next_export_t = 0.0
    record_start_t = warmup_duration
    sim_end_t = warmup_duration + duration

    while float(data.time) <= sim_end_t + 1e-12:
        t = float(data.time)
        targets = controller.joint_targets(t)

        for joint_name, actuator_id in actuator_map.items():
            qpos_adr, qvel_adr = joint_map[joint_name]
            data.ctrl[actuator_id] = compute_pd_torque(
                float(data.qpos[qpos_adr]),
                float(data.qvel[qvel_adr]),
                targets[joint_name],
                config=pd,
            )

        recorded_t = t - record_start_t
        if t + 1e-12 >= record_start_t and recorded_t + 1e-12 >= next_export_t:
            frames.append(
                _make_frame(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    joint_map=joint_map,
                    controller=controller,
                    frame_t=max(0.0, recorded_t),
                    controller_t=t,
                )
            )
            next_export_t += export_dt

        mujoco.mj_step(model, data)

    return make_trajectory(
        frames=frames,
        joint_order=list(joint_map.keys()),
        dt=export_dt,
        source=str(model_path),
        extra_metadata={
            "preset": preset,
            "mode": "mujoco-rollout",
            "warmup_duration": warmup_duration,
        },
        notes=(
            "MuJoCo rollout using an open-loop trot target generator and PD torque control. "
            "The included controller is for teaching, not robust locomotion."
        ),
    )


def _find_joint_addresses(mujoco: Any, model: Any, joint_names: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    joint_map: dict[str, tuple[int, int]] = {}
    missing: list[str] = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            missing.append(name)
            continue
        joint_map[name] = (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
        )

    if missing:
        raise ValueError(
            "Model is missing expected Go2 joint names: " + ", ".join(missing)
        )

    return joint_map


def _find_actuators_for_joints(mujoco: Any, model: Any, joint_map: dict[str, tuple[int, int]]) -> dict[str, int]:
    actuator_map: dict[str, int] = {}
    joint_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_map
    }

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        for joint_name, expected_joint_id in joint_ids.items():
            if joint_id == expected_joint_id:
                actuator_map[joint_name] = actuator_id

    missing = [name for name in joint_map if name not in actuator_map]
    if missing:
        raise ValueError(
            "Model is missing actuators attached to expected joints: "
            + ", ".join(missing)
        )

    return actuator_map


def _set_initial_pose(data: Any, joint_map: dict[str, tuple[int, int]], targets: dict[str, float]) -> None:
    if len(data.qpos) >= 7:
        data.qpos[2] = max(float(data.qpos[2]), 0.36)
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for joint_name, (qpos_adr, _) in joint_map.items():
        data.qpos[qpos_adr] = targets[joint_name]


def _make_frame(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    joint_map: dict[str, tuple[int, int]],
    controller: TrotGaitController,
    frame_t: float | None = None,
    controller_t: float | None = None,
) -> dict[str, Any]:
    qpos = [float(v) for v in data.qpos]
    qvel = [float(v) for v in data.qvel]
    ctrl = [float(v) for v in data.ctrl]

    base = BaseState(
        position=qpos[:3] if len(qpos) >= 3 else [0.0, 0.0, 0.0],
        quaternion=qpos[3:7] if len(qpos) >= 7 else [1.0, 0.0, 0.0, 0.0],
        linear_velocity=qvel[:3] if len(qvel) >= 3 else [0.0, 0.0, 0.0],
        angular_velocity=qvel[3:6] if len(qvel) >= 6 else [0.0, 0.0, 0.0],
    )

    joints = {
        joint_name: float(data.qpos[qpos_adr])
        for joint_name, (qpos_adr, _) in joint_map.items()
    }

    return {
        "t": round(float(data.time) if frame_t is None else frame_t, 6),
        "qpos": qpos,
        "qvel": qvel,
        "ctrl": ctrl,
        "base": asdict(base),
        "joints": joints,
        "contacts": _foot_contacts(mujoco, model, data),
        "gait_phase": {
            leg: round(phase, 6)
            for leg, phase in controller.gait_phase(
                float(data.time) if controller_t is None else controller_t
            ).items()
        },
    }


def _foot_contacts(mujoco: Any, model: Any, data: Any) -> dict[str, bool]:
    contacts = {leg: False for leg in LEGS}

    for i in range(data.ncon):
        contact = data.contact[i]
        geom_names = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "",
        )
        lower = " ".join(geom_names).lower()
        touches_ground = "floor" in lower or "ground" in lower
        if not touches_ground:
            continue

        for leg in LEGS:
            leg_token = leg.lower()
            if leg_token in {name.lower() for name in geom_names}:
                contacts[leg] = True
            elif leg_token in lower and "foot" in lower:
                contacts[leg] = True

    return contacts
