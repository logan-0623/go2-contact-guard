from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

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
from .controller import PDConfig
from .gym_env import Go2StandBalanceGymEnv
from .rl_env import RL_TASKS, make_task_config
from .sb3_tools import load_vecnormalize_if_available, make_vec_env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Roll out a privileged PPO teacher and save deployable student "
            "obs/action pairs for behavior cloning distillation."
        )
    )
    parser.add_argument("teacher_checkpoint", type=Path)
    parser.add_argument(
        "--teacher-vecnormalize",
        type=Path,
        help="VecNormalize stats for the teacher. Defaults to <teacher>_vecnormalize.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference_datasets/go2_teacher_student_distill.npz"),
    )
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--target-forward-velocity", type=float, default=0.40)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--action-smoothing", type=float, default=0.15)
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--reset-settle", type=float, default=0.0)
    parser.add_argument("--velocity-pose-profile", choices=("mjlab", "official"), default="official")
    add_policy_leg_order_arg(parser)
    add_observation_mode_arg(parser)
    parser.set_defaults(observation_mode="privileged")
    parser.add_argument(
        "--student-observation-mode",
        choices=("policy", "privileged"),
        default="policy",
        help="Observation vocabulary saved for the student dataset.",
    )
    parser.add_argument(
        "--student-external-force-observation",
        action="store_true",
        help="Append force observations to saved student observations. Leave off for sim-to-real.",
    )
    add_fullbody_reference_args(parser)
    add_velocity_reward_args(parser)
    add_external_force_args(parser)
    parser.add_argument("--pd-kp", type=float)
    parser.add_argument("--pd-kd", type=float)
    parser.add_argument("--torque-limit", type=float)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-teacher-observations", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        import numpy as np
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing. Install with: pip install -e '.[train]'") from exc

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
        fullbody_reference_mode=args.fullbody_reference_mode,
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
        checkpoint_path=args.teacher_checkpoint,
        vecnormalize_path=args.teacher_vecnormalize,
        disabled=False,
    )
    model = PPO.load(str(args.teacher_checkpoint), device="cpu")

    student_observations: list[Any] = []
    teacher_observations: list[Any] = []
    actions: list[Any] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    commands: list[Any] = []
    external_force_active: list[bool] = []
    episode_lengths: list[int] = []

    for episode in range(max(1, args.episodes)):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        steps = 0
        while not terminated and not truncated:
            student_obs = env.env.observation_for_mode(
                args.student_observation_mode,
                include_external_force_observation=args.student_external_force_observation,
            )
            policy_obs = normalizer.normalize_obs(obs) if normalizer_path else obs
            action, _ = model.predict(policy_obs, deterministic=args.deterministic)
            action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)

            student_observations.append(np.asarray(student_obs, dtype=np.float32))
            if args.save_teacher_observations:
                teacher_observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            commands.append(np.asarray(env.env._command(), dtype=np.float32))

            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            terminals.append(bool(terminated))
            external_force_active.append(bool(info.get("external_force_active", False)))
            steps += 1
        episode_lengths.append(steps)

    if not student_observations:
        raise RuntimeError("teacher distillation rollout produced no samples")

    student_obs_array = np.stack(student_observations).astype(np.float32)
    actions_array = np.stack(actions).astype(np.float32)
    metadata = {
        "source_policy": str(args.teacher_checkpoint),
        "source_policy_format": "sb3_ppo_teacher",
        "teacher_vecnormalize": str(normalizer_path) if normalizer_path else None,
        "model": str(args.model),
        "task": args.task,
        "episodes": max(1, args.episodes),
        "duration": args.duration,
        "control_dt": args.control_dt,
        "samples": int(student_obs_array.shape[0]),
        "episode_lengths": episode_lengths,
        "observation_size": int(student_obs_array.shape[1]),
        "observation_mode": args.student_observation_mode,
        "include_external_force_observation": args.student_external_force_observation,
        "teacher_observation_mode": args.observation_mode,
        "teacher_observation_size": int(env.observation_space.shape[0]),
        "observation_names": list(
            env.env._velocity_policy_observation_names(
                include_external_force_observation=args.student_external_force_observation
            )
            if args.student_observation_mode == "policy"
            else env.env._velocity_privileged_observation_names()
        ),
        "action_size": int(actions_array.shape[1]),
        "action_names": list(env.env.joint_names),
        "target_forward_velocity": env_config.target_forward_velocity,
        "target_lateral_velocity": env_config.target_lateral_velocity,
        "target_yaw_rate": env_config.target_yaw_rate,
        "action_scale": env_config.action_scale,
        "action_smoothing": env_config.action_smoothing,
        "reward_scale": env_config.reward_scale,
        "reset_settle_s": env_config.reset_settle_s,
        "velocity_pose_profile": env_config.velocity_pose_profile,
        "policy_leg_order": args.policy_leg_order,
        "velocity_reward_profile": env_config.velocity_reward_profile,
        "velocity_command_frame": env_config.velocity_command_frame,
        "fullbody_reference_mode": env_config.fullbody_reference_mode,
        "gait_frequency_hz": env_config.gait_frequency_hz,
        "gait_step_length": env_config.gait_step_length,
        "gait_swing_height": env_config.gait_swing_height,
        "gait_joint_thigh_amplitude": env_config.gait_joint_thigh_amplitude,
        "gait_joint_calf_amplitude": env_config.gait_joint_calf_amplitude,
        "pd_kp": env_config.pd.kp,
        "pd_kd": env_config.pd.kd,
        "torque_limit": env_config.pd.torque_limit,
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
        "external_force_torque_max_nm": env_config.external_force_torque_max_nm,
        "external_force_net_force_limit_n": env_config.external_force_net_force_limit_n,
        "external_force_net_torque_limit_nm": env_config.external_force_net_torque_limit_nm,
    }

    arrays: dict[str, Any] = {
        "observations": student_obs_array,
        "actions": actions_array,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.bool_),
        "commands": np.stack(commands).astype(np.float32),
        "external_force_active": np.asarray(external_force_active, dtype=np.bool_),
        "metadata": json.dumps(metadata),
    }
    if args.save_teacher_observations:
        arrays["teacher_observations"] = np.stack(teacher_observations).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    print(f"teacher_checkpoint: {args.teacher_checkpoint}")
    print(f"teacher_observation_mode: {args.observation_mode}")
    print(f"student_observation_mode: {args.student_observation_mode}")
    print(f"samples: {student_obs_array.shape[0]}")
    print(f"student_observation_size: {student_obs_array.shape[1]}")
    print(f"action_size: {actions_array.shape[1]}")
    print(f"external_force_active_fraction: {float(np.mean(arrays['external_force_active'])):.3f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
