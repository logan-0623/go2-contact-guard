from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .export import _strip_raw_state_arrays
from .rl_env import Go2StandBalanceEnv, RL_TASKS, make_task_config
from .trajectory import make_trajectory
from .trajectory_analysis import metrics_output_path, validate_trajectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a short Go2 RL environment sanity check."
    )
    parser.add_argument(
        "--task",
        choices=RL_TASKS,
        default="stand",
        help="Environment task.",
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
        default=Path("rl_stand_check.json"),
        help="Output trajectory JSON path.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Check rollout duration in seconds.",
    )
    parser.add_argument(
        "--export-dt",
        type=float,
        default=0.05,
        help="Trajectory export timestep in seconds.",
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
        "--action-scale",
        type=float,
        help="Max joint target offset in rad when action is +/-1. Default is task-specific.",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        help="EMA smoothing for actions. Default is task-specific.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        help="Multiplier applied to environment rewards. Default is task-specific.",
    )
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
        "--policy",
        choices=("zero", "random"),
        default="zero",
        help="Action source for the sanity check.",
    )
    parser.add_argument(
        "--random-scale",
        type=float,
        default=0.25,
        help="Random action scale in [-scale, scale] when --policy random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Omit raw qpos/qvel/ctrl arrays from the exported replay JSON.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not write rollout metrics JSON.",
    )
    args = parser.parse_args(argv)

    randomize_commands = None
    if args.randomize_commands:
        randomize_commands = True
    if args.fixed_command:
        randomize_commands = False

    env_config = make_task_config(
        task=args.task,
        control_dt=args.control_dt,
        episode_length_s=args.duration,
        target_forward_velocity=args.target_forward_velocity,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        reward_scale=args.reward_scale,
        randomize_commands=randomize_commands,
    )
    env = Go2StandBalanceEnv(
        model_path=args.model,
        config=env_config,
    )
    obs, info = env.reset(seed=args.seed)
    frames = [env.make_frame(frame_t=0.0)]
    next_export_t = args.export_dt
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.zero_action() if args.policy == "zero" else env.sample_action(args.random_scale)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if env.elapsed_s + 1e-12 >= next_export_t:
            frames.append(env.make_frame(frame_t=next_export_t))
            next_export_t += args.export_dt

    trajectory = make_trajectory(
        frames=frames,
        joint_order=list(env.joint_names),
        dt=args.export_dt,
        source=str(args.model),
        extra_metadata={
            "mode": "mujoco-rl-env-check",
            "preset": f"{args.task}_{args.policy}",
            "task": args.task,
            "policy": args.policy,
            "control_dt": args.control_dt,
            "target_forward_velocity": env_config.target_forward_velocity,
            "target_lateral_velocity": env_config.target_lateral_velocity,
            "target_yaw_rate": env_config.target_yaw_rate,
            "command": [float(v) for v in env._command()],
            "action_scale": env_config.action_scale,
            "action_smoothing": env_config.action_smoothing,
            "reward_scale": env_config.reward_scale,
            "velocity_pose_profile": env_config.velocity_pose_profile,
            "velocity_reward_profile": env_config.velocity_reward_profile,
            "randomize_commands": env_config.randomize_commands,
            "push_time_s": env_config.push_time_s,
            "push_linear_velocity": env_config.push_linear_velocity,
            "push_angular_velocity": env_config.push_angular_velocity,
            "push_applied": env._push_applied,
            "observation_size": len(obs),
            "observation_names": list(env.observation_names),
            "action_size": env.action_size,
            "action_names": list(env.joint_names),
            "terminated": terminated,
            "truncated": truncated,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(steps, 1),
        },
        notes=(
            "Short environment sanity check. This is not PPO training; it verifies "
            "observation, action, reward, termination, and trajectory export plumbing."
        ),
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
    print(f"observation_size: {len(obs)}")
    print(f"action_size: {env.action_size}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"command: {[round(float(v), 4) for v in env._command()]}")
    print(f"action_scale: {env_config.action_scale}")
    print(f"action_smoothing: {env_config.action_smoothing}")
    print(f"reward_scale: {env_config.reward_scale}")
    print(f"randomize_commands: {env_config.randomize_commands}")
    if env_config.push_time_s is not None:
        print(f"push_applied: {str(env._push_applied).lower()}")
    print(f"total_reward: {total_reward:.3f}")
    print(f"mean_reward: {total_reward / max(steps, 1):.3f}")
    print(f"terminated: {str(terminated).lower()}")
    print(f"truncated: {str(truncated).lower()}")
    print(f"health: {result['metrics']['health_status']}")
    if result["warnings"]:
        print(f"validation warnings: {len(result['warnings'])}")
        for warning in result["warnings"][:5]:
            print(f"- {warning}")
    if result["errors"]:
        print(f"validation errors: {len(result['errors'])}")
        for error in result["errors"][:5]:
            print(f"- {error}")

    if not args.no_metrics:
        metrics_path = metrics_output_path(args.output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        print(f"wrote {metrics_path}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
