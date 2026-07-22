from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .backflip_reference import make_backflip_reference
from .controller import UNITREE_GO2_JOINTS, get_controller_preset
from .export import _strip_raw_state_arrays
from .mujoco_rollout import run_mujoco_rollout
from .rl_env import Go2StandBalanceEnv, StandBalanceEnvConfig
from .trajectory import BaseState, make_trajectory
from .trajectory_analysis import metrics_output_path, validate_trajectory


SKILL_STAGES = (
    "stand",
    "slow_walk",
    "push_recovery",
    "hop_reference",
    "backflip_reference",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a sanity-check trajectory for one stage of the Go2 skill curriculum."
    )
    parser.add_argument("--stage", choices=SKILL_STAGES, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
        help="Path to the Go2 MuJoCo MJCF scene for physics stages.",
    )
    parser.add_argument("--output", type=Path, help="Output trajectory JSON path.")
    parser.add_argument("--duration", type=float, help="Duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.05, help="Export timestep in seconds.")
    parser.add_argument("--compact-json", action="store_true", default=True)
    args = parser.parse_args(argv)

    output = args.output or Path(f"{args.stage}.json")
    duration = args.duration or _default_duration(args.stage)
    trajectory = _make_stage_trajectory(args.stage, args.model, duration, args.dt)
    _annotate_skill_check(trajectory, args.stage)
    if args.compact_json:
        _strip_raw_state_arrays(trajectory)

    result = validate_trajectory(trajectory)
    trajectory.setdefault("metadata", {})["metrics"] = result["metrics"]
    trajectory["metadata"]["validation"] = {
        "passed": result["passed"],
        "health_status": result["metrics"]["health_status"],
        "warnings": result["warnings"],
        "errors": result["errors"],
        "warning_events": result["metrics"]["warning_events"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

    metrics_path = metrics_output_path(output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")

    print(f"wrote {output} ({len(trajectory['frames'])} frames)")
    print(f"stage: {args.stage}")
    print(f"mode: {trajectory['metadata'].get('mode')}")
    print(f"realism: {trajectory['metadata'].get('realism_level')}")
    print(f"note: {trajectory['metadata'].get('realism_note')}")
    print(f"health: {result['metrics']['health_status']}")
    print(f"fall_detected: {str(result['metrics']['fall_detected']).lower()}")
    if result["warnings"]:
        print(f"validation warnings: {len(result['warnings'])}")
        for warning in result["warnings"][:5]:
            print(f"- {warning}")
    print(f"wrote {metrics_path}")
    return 0 if result["passed"] else 1


def _make_stage_trajectory(stage: str, model_path: Path, duration: float, dt: float) -> dict[str, Any]:
    if stage == "stand":
        return _stand_or_push_recovery(model_path=model_path, duration=duration, dt=dt, push=False)
    if stage == "push_recovery":
        return _stand_or_push_recovery(model_path=model_path, duration=duration, dt=dt, push=True)
    if stage == "slow_walk":
        preset = get_controller_preset("slow_trot")
        return run_mujoco_rollout(
            model_path=model_path,
            duration=duration,
            export_dt=dt,
            warmup_duration=1.0,
            gait_config=preset.gait,
            pd_config=preset.pd,
            preset="slow_trot",
        )
    if stage == "hop_reference":
        return make_hop_reference(duration=duration, dt=min(dt, 0.02))
    if stage == "backflip_reference":
        return make_backflip_reference(duration=duration, dt=min(dt, 0.02))
    raise ValueError(f"unsupported stage: {stage}")


def _annotate_skill_check(trajectory: dict[str, Any], stage: str) -> None:
    metadata = trajectory.setdefault("metadata", {})
    notes = {
        "stand": (
            "physics_sanity_check",
            "MuJoCo zero-action stand check; not a trained policy.",
        ),
        "slow_walk": (
            "open_loop_pd_check",
            "Open-loop PD rollout; not learned locomotion and may look unrealistic.",
        ),
        "push_recovery": (
            "gentle_push_sanity_check",
            "Zero-action stand check with a tiny push; not a trained recovery policy.",
        ),
        "hop_reference": (
            "kinematic_reference",
            "Kinematic reference motion; not MuJoCo physics and not a trained policy.",
        ),
        "backflip_reference": (
            "kinematic_reference",
            "Kinematic reference motion; not MuJoCo physics and not a trained policy.",
        ),
    }
    realism_level, realism_note = notes[stage]
    if stage == "slow_walk":
        metadata["mode"] = "mujoco-skill-check-slow_walk"
    metadata["skill_stage"] = stage
    metadata["training_status"] = "not_trained_policy"
    metadata["realism_level"] = realism_level
    metadata["realism_note"] = realism_note


def _stand_or_push_recovery(*, model_path: Path, duration: float, dt: float, push: bool) -> dict[str, Any]:
    env = Go2StandBalanceEnv(
        model_path=model_path,
        config=StandBalanceEnvConfig(
            control_dt=0.02,
            episode_length_s=duration,
        ),
    )
    env.reset(seed=1)
    frames = [env.make_frame(frame_t=0.0)]
    next_export_t = dt
    push_applied = False
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    while not terminated and not truncated:
        if push and not push_applied and env.elapsed_s >= 0.5:
            env.data.qvel[0] += 0.05
            env.data.qvel[1] += -0.02
            env.data.qvel[4] += 0.05
            push_applied = True
        _, reward, terminated, truncated, _ = env.step(env.zero_action())
        total_reward += reward
        steps += 1
        if env.elapsed_s + 1e-12 >= next_export_t:
            frames.append(env.make_frame(frame_t=next_export_t))
            next_export_t += dt

    stage = "push_recovery" if push else "stand"
    return make_trajectory(
        frames=frames,
        joint_order=list(env.joint_names),
        dt=dt,
        source=str(model_path),
        extra_metadata={
            "mode": f"mujoco-skill-check-{stage}",
            "preset": stage,
            "skill_stage": stage,
            "push_applied": push_applied,
            "terminated": terminated,
            "truncated": truncated,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(steps, 1),
        },
        notes=f"Physics sanity-check trajectory for curriculum stage: {stage}.",
    )


def make_hop_reference(*, duration: float = 1.4, dt: float = 0.02) -> dict[str, Any]:
    frames = []
    steps = max(1, int(duration / dt))
    previous = _hop_pose("stand")
    for index in range(steps + 1):
        t = round(index * dt, 6)
        progress = min(1.0, t / duration)
        base_z, contacts, pose_name = _hop_state(progress)
        joints = _hop_blend_pose(pose_name, progress)
        base = BaseState(
            position=[0.0, 0.0, base_z],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            linear_velocity=[0.0, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )
        frames.append(
            {
                "t": t,
                "qpos": [*base.position, *base.quaternion, *[joints[name] for name in UNITREE_GO2_JOINTS]],
                "qvel": [
                    *base.linear_velocity,
                    *base.angular_velocity,
                    *[
                        (joints[name] - previous[name]) / dt if index else 0.0
                        for name in UNITREE_GO2_JOINTS
                    ],
                ],
                "ctrl": [0.0 for _ in UNITREE_GO2_JOINTS],
                "base": asdict(base),
                "joints": {name: round(value, 6) for name, value in joints.items()},
                "contacts": contacts,
                "gait_phase": {foot: round(progress, 6) for foot in ("FR", "FL", "RR", "RL")},
            }
        )
        previous = joints

    return make_trajectory(
        frames=frames,
        joint_order=list(UNITREE_GO2_JOINTS),
        dt=dt,
        source="kinematic-hop-reference",
        extra_metadata={
            "mode": "kinematic-hop-reference",
            "preset": "hop_reference",
            "skill_stage": "hop_reference",
            "allow_flight_phase": True,
            "reference_motion": True,
        },
        notes="Kinematic hop reference for future jump and backflip curriculum stages.",
    )


def _hop_state(progress: float) -> tuple[float, dict[str, bool], str]:
    if progress < 0.25:
        amount = _smooth(progress / 0.25)
        return _lerp(0.36, 0.335, amount), _contacts(True), "crouch"
    if progress < 0.38:
        amount = _smooth((progress - 0.25) / 0.13)
        return _lerp(0.335, 0.52, amount), _contacts(True), "takeoff"
    if progress < 0.68:
        amount = (progress - 0.38) / 0.30
        return 0.52 + 0.08 * math.sin(math.pi * amount), _contacts(False), "tuck"
    if progress < 0.82:
        amount = _smooth((progress - 0.68) / 0.14)
        return _lerp(0.50, 0.36, amount), _contacts(True), "landing"
    return 0.36, _contacts(True), "stand"


def _hop_blend_pose(pose_name: str, progress: float) -> dict[str, float]:
    if progress < 0.25:
        return _interpolate_hop_pose("stand", "crouch", _smooth(progress / 0.25))
    if progress < 0.38:
        return _interpolate_hop_pose("crouch", "takeoff", _smooth((progress - 0.25) / 0.13))
    if progress < 0.68:
        return _hop_pose("tuck")
    if progress < 0.82:
        return _interpolate_hop_pose("tuck", "landing", _smooth((progress - 0.68) / 0.14))
    return _interpolate_hop_pose("landing", "stand", _smooth((progress - 0.82) / 0.18))


def _hop_pose(name: str) -> dict[str, float]:
    poses = {
        "stand": (0.0, 0.608813, -1.21763),
        "crouch": (0.0, 1.0, -1.95),
        "takeoff": (0.0, 0.25, -0.9),
        "tuck": (0.0, 0.86, -1.75),
        "landing": (0.0, 0.72, -1.45),
    }
    hip, thigh, calf = poses[name]
    return {
        joint_name: hip if "_hip_" in joint_name else thigh if "_thigh_" in joint_name else calf
        for joint_name in UNITREE_GO2_JOINTS
    }


def _interpolate_hop_pose(start: str, end: str, amount: float) -> dict[str, float]:
    a = _hop_pose(start)
    b = _hop_pose(end)
    return {name: _lerp(a[name], b[name], amount) for name in UNITREE_GO2_JOINTS}


def _contacts(value: bool) -> dict[str, bool]:
    return {foot: value for foot in ("FR", "FL", "RR", "RL")}


def _smooth(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _default_duration(stage: str) -> float:
    if stage == "hop_reference":
        return 1.4
    if stage == "backflip_reference":
        return 2.2
    return 3.0


if __name__ == "__main__":
    raise SystemExit(main())
