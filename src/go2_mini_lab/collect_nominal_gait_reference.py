from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .cli import (
    add_fullbody_reference_args,
    add_observation_mode_arg,
    add_policy_leg_order_arg,
    add_velocity_reward_args,
    parse_reward_overrides,
    resolve_policy_leg_order,
)
from .controller import PDConfig
from .gym_env import Go2StandBalanceGymEnv
from .reference.reference_audit import audit_nominal_gait_file, write_audit_report
from .rl_env import FEET, FULLBODY_TRACKING_KEYPOINTS, RL_TASKS, make_task_config
from .sb3_tools import load_vecnormalize_if_available, make_vec_env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect a phase-conditioned nominal Go2 gait reference from a PPO checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Stable-Baselines3 PPO checkpoint path.")
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
        help="Path to the Go2 MuJoCo MJCF scene.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference_datasets/go2_nominal_gait_040_reference.npz"),
        help="Output nominal gait reference .npz path.",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--target-forward-velocity", type=float, default=0.40)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument(
        "--fixed-command",
        action="store_true",
        help="Accepted for parity with train/evaluate; collection already uses fixed commands.",
    )
    parser.add_argument("--phase-bins", type=int, default=100)
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--action-smoothing", type=float)
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--reset-settle", type=float)
    parser.add_argument(
        "--velocity-pose-profile",
        choices=("mjlab", "official"),
        default="official",
    )
    add_policy_leg_order_arg(parser)
    add_observation_mode_arg(parser)
    add_fullbody_reference_args(parser)
    parser.set_defaults(fullbody_reference_mode="bounded_compliant")
    add_velocity_reward_args(parser)
    parser.add_argument("--pd-kp", type=float)
    parser.add_argument("--pd-kd", type=float)
    parser.add_argument("--torque-limit", type=float)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vecnormalize", type=Path)
    parser.add_argument("--no-vecnormalize", action="store_true")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/go2_nominal_reference_audit.json"),
        help="Optional audit JSON path written after collection.",
    )
    parser.add_argument(
        "--allow-audit-fail",
        action="store_true",
        help="Write the dataset even if the reference audit fails.",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        from stable_baselines3 import PPO
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3/NumPy are missing. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    velocity_reward_overrides = parse_reward_overrides(args.reward_term)
    env_config = make_task_config(
        task=args.task,
        control_dt=args.control_dt,
        episode_length_s=args.duration,
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        reward_scale=args.reward_scale,
        reset_settle_s=args.reset_settle,
        randomize_commands=False,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        observation_mode=args.observation_mode,
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
        fullbody_reference_mode=args.fullbody_reference_mode,
        nominal_reference_dataset=args.nominal_reference_dataset,
        gait_frequency_hz=args.gait_frequency,
        gait_step_length=args.gait_step_length,
        gait_swing_height=args.gait_swing_height,
        gait_joint_thigh_amplitude=args.gait_thigh_amplitude,
        gait_joint_calf_amplitude=args.gait_calf_amplitude,
        velocity_reward_overrides=velocity_reward_overrides,
    )
    if args.pd_kp is not None or args.pd_kd is not None or args.torque_limit is not None:
        env_config = replace(
            env_config,
            pd=PDConfig(
                kp=env_config.pd.kp if args.pd_kp is None else args.pd_kp,
                kd=env_config.pd.kd if args.pd_kd is None else args.pd_kd,
                torque_limit=env_config.pd.torque_limit if args.torque_limit is None else args.torque_limit,
            ),
        )

    env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
    vec_env = make_vec_env(model_path=args.model, config=env_config)
    normalizer, normalizer_path = load_vecnormalize_if_available(
        vec_env=vec_env,
        checkpoint_path=args.checkpoint,
        vecnormalize_path=args.vecnormalize,
        disabled=args.no_vecnormalize,
    )
    model = PPO.load(str(args.checkpoint), device="cpu")

    raw_samples: list[dict[str, Any]] = []
    for episode in range(max(1, int(args.episodes))):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            phase = (env.env.elapsed_s * float(env_config.gait_frequency_hz)) % 1.0
            policy_obs = normalizer.normalize_obs(obs) if normalizer_path else obs
            action, _ = model.predict(policy_obs, deterministic=args.deterministic)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            raw_samples.append(_collect_reference_sample(env.env, action=action, phase=phase, np_module=np))
            obs, _reward, terminated, truncated, _info = env.step(action)

    if not raw_samples:
        raise RuntimeError("nominal gait reference collection produced no samples")

    binned = _bin_phase_samples(
        raw_samples,
        phase_bins=max(2, int(args.phase_bins)),
        gait_frequency_hz=float(env_config.gait_frequency_hz),
        np_module=np,
    )
    metadata = {
        "source": "ppo_rollout",
        "source_checkpoint": str(args.checkpoint),
        "vecnormalize": str(normalizer_path) if normalizer_path else None,
        "model": str(args.model),
        "task": args.task,
        "episodes": max(1, int(args.episodes)),
        "duration": args.duration,
        "raw_samples": len(raw_samples),
        "phase_bins": max(2, int(args.phase_bins)),
        "control_dt": env_config.control_dt,
        "target_forward_velocity": env_config.target_forward_velocity,
        "target_lateral_velocity": env_config.target_lateral_velocity,
        "target_yaw_rate": env_config.target_yaw_rate,
        "gait_frequency_hz": env_config.gait_frequency_hz,
        "action_names": list(env.env.joint_names),
        "foot_names": list(FEET),
        "keypoint_names": list(FULLBODY_TRACKING_KEYPOINTS),
        "observation_mode": env_config.observation_mode,
        "fullbody_reference_mode": env_config.fullbody_reference_mode,
        "velocity_reward_profile": env_config.velocity_reward_profile,
        "velocity_reward_overrides": velocity_reward_overrides,
        "policy_leg_order": args.policy_leg_order,
        "pd_kp": env_config.pd.kp,
        "pd_kd": env_config.pd.kd,
        "torque_limit": env_config.pd.torque_limit,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **binned, metadata=json.dumps(metadata))
    report = audit_nominal_gait_file(args.output)
    write_audit_report(args.audit_output, report)

    print(f"wrote {args.output}")
    print(f"raw_samples: {len(raw_samples)}")
    print(f"phase_bins: {binned['phase_ref'].shape[0]}")
    print(f"mean_vx: {report.metrics['mean_vx']:.3f}")
    print(f"mean_abs_vy: {report.metrics['mean_abs_vy']:.3f}")
    print(f"mean_abs_yaw_rate: {report.metrics['mean_abs_yaw_rate']:.3f}")
    print(f"base_z_min: {report.metrics['base_z_min']:.3f}")
    print(f"audit: {'PASS' if report.passed else 'FAIL'}")
    print(f"wrote {args.audit_output}")
    if not report.passed:
        for name, reason in report.failures.items():
            print(f"FAIL {name}: {reason}")
    return 0 if report.passed or args.allow_audit_fail else 1


def _collect_reference_sample(env: Any, *, action: Any, phase: float, np_module: Any) -> dict[str, Any]:
    root_rotation = env._root_rotation_matrix()
    root_position = np_module.asarray(env.data.qpos[:3], dtype=np_module.float32)
    frame = env.make_frame()
    keypoint_positions = []
    keypoint_velocities = []
    for name in FULLBODY_TRACKING_KEYPOINTS:
        body_id = env.fullbody_tracking_body_ids.get(name, -1)
        if body_id >= 0:
            position_world = np_module.asarray(env.data.xpos[body_id], dtype=np_module.float32)
            velocity_world = np_module.asarray(env.data.cvel[body_id][3:6], dtype=np_module.float32)
            keypoint_positions.append(root_rotation.T @ (position_world - root_position))
            keypoint_velocities.append(root_rotation.T @ velocity_world)
        else:
            keypoint_positions.append(np_module.zeros(3, dtype=np_module.float32))
            keypoint_velocities.append(np_module.zeros(3, dtype=np_module.float32))

    foot_positions = []
    foot_velocities = []
    for foot in FEET:
        body_id = env.foot_body_ids[foot]
        if body_id >= 0:
            foot_positions.append(np_module.asarray(env.data.xpos[body_id], dtype=np_module.float32))
            foot_velocities.append(np_module.asarray(env.data.cvel[body_id][3:6], dtype=np_module.float32))
        else:
            foot_positions.append(np_module.zeros(3, dtype=np_module.float32))
            foot_velocities.append(np_module.zeros(3, dtype=np_module.float32))

    return {
        "phase_ref": float(phase),
        "q_ref": np_module.asarray(
            [float(env.data.qpos[env.joint_map[name][0]]) for name in env.joint_names],
            dtype=np_module.float32,
        ),
        "dq_ref": np_module.asarray(
            [float(env.data.qvel[env.joint_map[name][1]]) for name in env.joint_names],
            dtype=np_module.float32,
        ),
        "action_ref": np_module.asarray(action, dtype=np_module.float32),
        "joint_target_ref": np_module.asarray(
            [
                env.default_targets[name] + env.config.action_scale * float(action[index])
                for index, name in enumerate(env.joint_names)
            ],
            dtype=np_module.float32,
        ),
        "foot_pos_ref": np_module.stack(foot_positions).astype(np_module.float32),
        "foot_vel_ref": np_module.stack(foot_velocities).astype(np_module.float32),
        "contact_ref": np_module.asarray(
            [1.0 if frame.get("contacts", {}).get(foot, False) else 0.0 for foot in FEET],
            dtype=np_module.float32,
        ),
        "base_pos_ref": np_module.asarray(env.data.qpos[:3], dtype=np_module.float32).copy(),
        "base_quat_ref": np_module.asarray(env.data.qpos[3:7], dtype=np_module.float32).copy(),
        "base_rpy_ref": np_module.asarray(_quat_wxyz_to_rpy(env.data.qpos[3:7]), dtype=np_module.float32),
        "base_height_ref": float(env.data.qpos[2]),
        "base_vel_ref": np_module.asarray(env.data.qvel[:3], dtype=np_module.float32).copy(),
        "base_angvel_ref": np_module.asarray(env.data.qvel[3:6], dtype=np_module.float32).copy(),
        "keypoint_pos_ref": np_module.stack(keypoint_positions).astype(np_module.float32),
        "keypoint_vel_ref": np_module.stack(keypoint_velocities).astype(np_module.float32),
    }


def _bin_phase_samples(
    samples: list[dict[str, Any]],
    *,
    phase_bins: int,
    gait_frequency_hz: float,
    np_module: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {"phase_ref": np_module.asarray(
        [(index + 0.5) / phase_bins for index in range(phase_bins)],
        dtype=np_module.float32,
    )}
    keys = [key for key in samples[0] if key != "phase_ref"]
    phases = np_module.asarray([sample["phase_ref"] for sample in samples], dtype=np_module.float32)
    indices_by_bin = [[] for _ in range(phase_bins)]
    for index, phase in enumerate(phases):
        bin_index = min(phase_bins - 1, int(float(phase % 1.0) * phase_bins))
        indices_by_bin[bin_index].append(index)

    for key in keys:
        values = np_module.stack([sample[key] for sample in samples]).astype(np_module.float32)
        binned_values = []
        for bin_index in range(phase_bins):
            sample_indices = indices_by_bin[bin_index]
            if sample_indices:
                binned = np_module.mean(values[sample_indices], axis=0)
            else:
                center = (bin_index + 0.5) / phase_bins
                nearest = int(np_module.argmin(np_module.minimum(abs(phases - center), 1.0 - abs(phases - center))))
                binned = values[nearest]
            if key == "base_pos_ref":
                binned = np_module.asarray([0.0, 0.0, float(binned[2])], dtype=np_module.float32)
            if key == "base_quat_ref":
                binned = _normalize_quat(binned, np_module=np_module)
            binned_values.append(binned)
        output[key] = np_module.stack(binned_values).astype(np_module.float32)
    output["time_ref"] = output["phase_ref"] / max(1e-6, float(gait_frequency_hz))
    return output


def _quat_wxyz_to_rpy(quat: Any) -> tuple[float, float, float]:
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _normalize_quat(quat: Any, *, np_module: Any) -> Any:
    value = np_module.asarray(quat, dtype=np_module.float32)
    norm = float(np_module.linalg.norm(value))
    if norm <= 1e-8:
        return np_module.asarray([1.0, 0.0, 0.0, 0.0], dtype=np_module.float32)
    return (value / norm).astype(np_module.float32)


if __name__ == "__main__":
    raise SystemExit(main())
