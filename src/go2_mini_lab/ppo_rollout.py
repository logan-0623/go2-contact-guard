from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .controller import PDConfig
from .cli import (
    add_external_force_args,
    add_fullbody_reference_args,
    add_observation_mode_arg,
    add_policy_leg_order_arg,
    add_velocity_reward_args,
    parse_reward_overrides,
    resolve_external_force_body_names,
    resolve_policy_leg_order,
)
from .export import _strip_raw_state_arrays
from .gym_env import Go2StandBalanceGymEnv
from .rl_env import RL_TASKS, make_task_config
from .sb3_tools import load_vecnormalize_if_available, make_vec_env
from .trajectory import make_trajectory
from .trajectory_analysis import metrics_output_path, validate_trajectory


DEFAULT_EXPORT_DT = 0.02


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a MuJoCo trajectory rollout from a trained PPO checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Stable-Baselines3 PPO checkpoint path.")
    parser.add_argument(
        "--task",
        choices=RL_TASKS,
        default="stand",
        help="Task used when the checkpoint was trained.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
        help="Path to the Go2 MuJoCo MJCF scene.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ppo_rollout.json"),
        help="Output trajectory JSON path.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Rollout duration in seconds.",
    )
    parser.add_argument(
        "--export-dt",
        type=float,
        default=DEFAULT_EXPORT_DT,
        help="Trajectory export timestep in seconds. Defaults to the 50 Hz control rate.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.02,
        help="Environment control timestep in seconds.",
    )
    parser.add_argument(
        "--target-forward-velocity",
        type=float,
        help="Target x velocity in m/s for fixed-command velocity tasks. Default is task-specific.",
    )
    parser.add_argument(
        "--target-lateral-velocity",
        type=float,
        default=0.0,
        help="Target y velocity in m/s.",
    )
    parser.add_argument(
        "--target-yaw-rate",
        type=float,
        default=0.0,
        help="Target yaw rate in rad/s.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        help="Max joint target offset in rad when action is +/-1. Default is task-specific.",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        help="EMA smoothing for policy actions. Use the same value as training.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        help="Multiplier applied to environment rewards. Use the same value as training.",
    )
    parser.add_argument(
        "--reset-settle",
        type=float,
        help="Seconds to settle with zero action after reset. Use the same value as training.",
    )
    parser.add_argument(
        "--velocity-pose-profile",
        choices=("mjlab", "official"),
        default="official",
        help="Default joint pose profile for velocity tasks.",
    )
    add_policy_leg_order_arg(parser)
    add_observation_mode_arg(parser)
    add_fullbody_reference_args(parser)
    parser.add_argument("--pd-kp", type=float, help="Joint PD stiffness. Use the same value as training.")
    parser.add_argument("--pd-kd", type=float, help="Joint PD damping. Use the same value as training.")
    parser.add_argument("--torque-limit", type=float, help="Joint torque limit. Use the same value as training.")
    add_velocity_reward_args(parser)
    parser.add_argument(
        "--randomize-commands",
        action="store_true",
        help="Sample velocity commands at reset. Enabled by default for velocity_flat.",
    )
    parser.add_argument(
        "--fixed-command",
        action="store_true",
        help="Disable command randomization even for velocity_flat.",
    )
    parser.add_argument(
        "--push-time",
        type=float,
        help="Push time in seconds for push_recovery. Default is task-specific.",
    )
    parser.add_argument("--push-vx", type=float, help="Root velocity impulse x component.")
    parser.add_argument("--push-vy", type=float, help="Root velocity impulse y component.")
    parser.add_argument("--push-vz", type=float, help="Root velocity impulse z component.")
    parser.add_argument("--push-wx", type=float, help="Root angular velocity impulse x component.")
    parser.add_argument("--push-wy", type=float, help="Root angular velocity impulse y component.")
    parser.add_argument("--push-wz", type=float, help="Root angular velocity impulse z component.")
    add_external_force_args(parser)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions.",
    )
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        help="VecNormalize stats path. Defaults to <checkpoint>_vecnormalize.pkl when present.",
    )
    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Do not load VecNormalize stats even if the default stats file exists.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Omit raw qpos/qvel/ctrl arrays from the exported replay JSON.",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is missing. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    randomize_commands = None
    if args.randomize_commands:
        randomize_commands = True
    if args.fixed_command:
        randomize_commands = False

    push_linear_velocity = _optional_vec(args.push_vx, args.push_vy, args.push_vz)
    push_angular_velocity = _optional_vec(args.push_wx, args.push_wy, args.push_wz)
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
        randomize_commands=randomize_commands,
        push_time_s=args.push_time,
        push_linear_velocity=push_linear_velocity,
        push_angular_velocity=push_angular_velocity,
        external_force_mode=args.external_force_mode,
        external_force_probability=args.external_force_probability,
        external_force_body_names=resolve_external_force_body_names(
            args.external_force_body,
            args.external_force_body_profile,
        ),
        external_force_active_body_count=args.external_force_active_body_count,
        external_force_event_count_range=(args.external_force_events_min, args.external_force_events_max),
        external_force_rest_s_range=(args.external_force_rest_min, args.external_force_rest_max),
        external_force_start_s_range=(args.external_force_start_min, args.external_force_start_max),
        external_force_duration_s_range=(args.external_force_duration_min, args.external_force_duration_max),
        external_force_min_n=args.external_force_min,
        external_force_max_n=args.external_force_max,
        external_force_z_fraction=args.external_force_z_fraction,
        external_force_direction_angle_rad=args.external_force_direction_angle,
        external_force_torque_max_nm=args.external_force_torque_max,
        external_force_spring_stiffness_range=(
            args.external_force_spring_stiffness_min,
            args.external_force_spring_stiffness_max,
        ),
        external_force_spring_damping=args.external_force_spring_damping,
        external_force_guiding_probability=args.external_force_guiding_probability,
        external_force_transition_s=args.external_force_transition,
        external_force_net_force_limit_n=args.external_force_net_force_limit,
        external_force_net_torque_limit_nm=args.external_force_net_torque_limit,
        external_force_reference_mass=args.external_force_reference_mass,
        external_force_reference_damping=args.external_force_reference_damping,
        external_force_reference_velocity_clip=args.external_force_reference_velocity_clip,
        external_force_reference_acceleration_clip=args.external_force_reference_acceleration_clip,
        include_external_force_observation=args.external_force_observation,
        observation_mode=args.observation_mode,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
        velocity_reward_overrides=velocity_reward_overrides,
        fullbody_reference_mode=args.fullbody_reference_mode,
        nominal_reference_dataset=args.nominal_reference_dataset,
        gait_frequency_hz=args.gait_frequency,
        gait_step_length=args.gait_step_length,
        gait_swing_height=args.gait_swing_height,
        gait_joint_thigh_amplitude=args.gait_thigh_amplitude,
        gait_joint_calf_amplitude=args.gait_calf_amplitude,
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
    env = Go2StandBalanceGymEnv(
        model_path=args.model,
        config=env_config,
    )
    vec_env = make_vec_env(model_path=args.model, config=env_config)
    normalizer, normalizer_path = load_vecnormalize_if_available(
        vec_env=vec_env,
        checkpoint_path=args.checkpoint,
        vecnormalize_path=args.vecnormalize,
        disabled=args.no_vecnormalize,
    )
    model = PPO.load(str(args.checkpoint), device="cpu")
    obs, _ = env.reset()
    frames = [env.env.make_frame(frame_t=0.0)]
    next_export_t = args.export_dt
    total_reward = 0.0
    steps = 0
    external_force_steps = 0
    external_force_max_magnitude = 0.0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        policy_obs = normalizer.normalize_obs(obs) if normalizer_path else obs
        action, _ = model.predict(policy_obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if bool(info.get("external_force_active", False)):
            external_force_steps += 1
            external_force_max_magnitude = max(
                external_force_max_magnitude,
                float(info.get("external_force_magnitude", 0.0)),
            )
        if env.env.elapsed_s + 1e-12 >= next_export_t:
            frames.append(env.env.make_frame(frame_t=next_export_t))
            next_export_t += args.export_dt

    trajectory = make_trajectory(
        frames=frames,
        joint_order=list(env.env.joint_names),
        dt=args.export_dt,
        source=str(args.checkpoint),
        extra_metadata={
            "mode": "mujoco-ppo-rollout",
            "preset": f"{args.task}_ppo",
            "task": args.task,
            "policy": "ppo",
            "control_dt": args.control_dt,
            "observation_size": len(obs),
            "observation_names": list(env.env.observation_names),
            "action_size": env.env.action_size,
            "action_names": list(env.env.joint_names),
            "target_forward_velocity": env_config.target_forward_velocity,
            "target_lateral_velocity": env_config.target_lateral_velocity,
            "target_yaw_rate": env_config.target_yaw_rate,
            "command": [float(v) for v in env.env._command()],
            "action_scale": env_config.action_scale,
            "action_smoothing": env_config.action_smoothing,
            "reward_scale": env_config.reward_scale,
            "reset_settle_s": env_config.reset_settle_s,
            "velocity_pose_profile": env_config.velocity_pose_profile,
            "policy_leg_order": args.policy_leg_order,
            "velocity_reward_profile": env_config.velocity_reward_profile,
            "velocity_command_frame": env_config.velocity_command_frame,
            "velocity_reward_overrides": velocity_reward_overrides,
            "fullbody_reference_mode": env_config.fullbody_reference_mode,
            "nominal_reference_dataset": env_config.nominal_reference_dataset,
            "gait_frequency_hz": env_config.gait_frequency_hz,
            "gait_step_length": env_config.gait_step_length,
            "gait_swing_height": env_config.gait_swing_height,
            "pd_kp": env_config.pd.kp,
            "pd_kd": env_config.pd.kd,
            "torque_limit": env_config.pd.torque_limit,
            "randomize_commands": env_config.randomize_commands,
            "vecnormalize": str(normalizer_path) if normalizer_path else None,
            "push_time_s": env_config.push_time_s,
            "push_linear_velocity": env_config.push_linear_velocity,
            "push_angular_velocity": env_config.push_angular_velocity,
            "push_applied": env.env._push_applied,
            "external_force_mode": env_config.external_force_mode,
            "external_force_probability": env_config.external_force_probability,
            "external_force_body_names": env_config.external_force_body_names,
            "external_force_active_body_count": env_config.external_force_active_body_count,
            "external_force_event_count_range": env_config.external_force_event_count_range,
            "external_force_rest_s_range": env_config.external_force_rest_s_range,
            "external_force_start_s_range": env_config.external_force_start_s_range,
            "external_force_duration_s_range": env_config.external_force_duration_s_range,
            "external_force_min_n": env_config.external_force_min_n,
            "external_force_max_n": env_config.external_force_max_n,
            "external_force_z_fraction": env_config.external_force_z_fraction,
            "external_force_direction_angle_rad": env_config.external_force_direction_angle_rad,
            "external_force_torque_max_nm": env_config.external_force_torque_max_nm,
            "external_force_spring_stiffness_range": env_config.external_force_spring_stiffness_range,
            "external_force_spring_damping": env_config.external_force_spring_damping,
            "external_force_guiding_probability": env_config.external_force_guiding_probability,
            "external_force_transition_s": env_config.external_force_transition_s,
            "external_force_net_force_limit_n": env_config.external_force_net_force_limit_n,
            "external_force_net_torque_limit_nm": env_config.external_force_net_torque_limit_nm,
            "external_force_reference_mass": env_config.external_force_reference_mass,
            "external_force_reference_damping": env_config.external_force_reference_damping,
            "external_force_reference_velocity_clip": env_config.external_force_reference_velocity_clip,
            "external_force_reference_acceleration_clip": env_config.external_force_reference_acceleration_clip,
            "include_external_force_observation": env_config.include_external_force_observation,
            "external_force_steps": external_force_steps,
            "external_force_max_magnitude": external_force_max_magnitude,
            "deterministic": args.deterministic,
            "terminated": terminated,
            "truncated": truncated,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(steps, 1),
        },
        notes="Rollout from a Stable-Baselines3 PPO checkpoint in the MuJoCo RL environment.",
    )
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(frames)} frames)")
    print(f"task: {args.task}")
    print(f"steps: {steps}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"command: {[round(float(v), 4) for v in env.env._command()]}")
    print(f"action_scale: {env_config.action_scale}")
    print(f"action_smoothing: {env_config.action_smoothing}")
    print(f"reward_scale: {env_config.reward_scale}")
    print(f"reset_settle_s: {env_config.reset_settle_s}")
    print(f"velocity_pose_profile: {env_config.velocity_pose_profile}")
    print(f"velocity_command_frame: {env_config.velocity_command_frame}")
    print(f"policy_leg_order: {args.policy_leg_order}")
    print(f"pd_kp: {env_config.pd.kp}")
    print(f"pd_kd: {env_config.pd.kd}")
    print(f"torque_limit: {env_config.pd.torque_limit}")
    print(f"randomize_commands: {env_config.randomize_commands}")
    print(f"vecnormalize: {normalizer_path if normalizer_path else 'none'}")
    if env_config.push_time_s is not None:
        print(f"push_applied: {str(env.env._push_applied).lower()}")
    if env_config.external_force_probability > 0.0 and env_config.external_force_max_n > 0.0:
        print(f"external_force_steps: {external_force_steps}")
        print(f"external_force_max_magnitude: {external_force_max_magnitude:.2f} N")
    print(f"total_reward: {total_reward:.3f}")
    print(f"mean_reward: {total_reward / max(steps, 1):.3f}")
    print(f"terminated: {str(terminated).lower()}")
    print(f"truncated: {str(truncated).lower()}")
    print(f"health: {result['metrics']['health_status']}")

    metrics_path = metrics_output_path(args.output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    print(f"wrote {metrics_path}")
    return 0 if result["passed"] else 1


def _optional_vec(
    x: float | None,
    y: float | None,
    z: float | None,
) -> tuple[float, float, float] | None:
    if x is None and y is None and z is None:
        return None
    return (float(x or 0.0), float(y or 0.0), float(z or 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
