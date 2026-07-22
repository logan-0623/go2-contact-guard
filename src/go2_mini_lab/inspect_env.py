from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .cli import (
    add_external_force_args,
    add_velocity_reward_args,
    parse_reward_overrides,
    resolve_external_force_body_names,
)
from .rl_env import RL_TASKS, VELOCITY_TASKS, Go2StandBalanceEnv, make_task_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Go2 RL environment observation and action layout."
    )
    parser.add_argument(
        "--task",
        choices=RL_TASKS,
        default="velocity_flat",
        help="Environment task.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
        help="Path to the Go2 MuJoCo MJCF scene.",
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
        help="EMA smoothing for policy actions. Default is task-specific.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        help="Multiplier applied to environment rewards. Default is task-specific.",
    )
    add_velocity_reward_args(parser)
    add_external_force_args(parser)
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    randomize_commands = None
    if args.randomize_commands:
        randomize_commands = True
    if args.fixed_command:
        randomize_commands = False

    velocity_reward_overrides = parse_reward_overrides(args.reward_term)
    config = make_task_config(
        task=args.task,
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        reward_scale=args.reward_scale,
        randomize_commands=randomize_commands,
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
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
        velocity_reward_overrides=velocity_reward_overrides,
    )
    env = Go2StandBalanceEnv(model_path=args.model, config=config)
    report = _make_report(env)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    _print_report(report)
    return 0


def _make_report(env: Go2StandBalanceEnv) -> dict[str, Any]:
    observation_sections = _observation_sections(env)
    action_names = list(env.joint_names)
    return {
        "task": env.config.task,
        "observation_size": len(env.observation_names),
        "observation_sections": observation_sections,
        "observation_names": list(env.observation_names),
        "action_size": env.action_size,
        "action_sections": [
            {
                "start": 0,
                "end": env.action_size,
                "name": "joint_target_offset",
                "scale": env.config.action_scale,
                "items": action_names,
            }
        ],
        "action_names": action_names,
        "config": {
            "control_dt": env.config.control_dt,
            "episode_length_s": env.config.episode_length_s,
            "target_forward_velocity": env.config.target_forward_velocity,
            "target_lateral_velocity": env.config.target_lateral_velocity,
            "target_yaw_rate": env.config.target_yaw_rate,
            "randomize_commands": env.config.randomize_commands,
            "action_scale": env.config.action_scale,
            "action_smoothing": env.config.action_smoothing,
            "reward_scale": env.config.reward_scale,
            "velocity_pose_profile": env.config.velocity_pose_profile,
            "velocity_reward_profile": env.config.velocity_reward_profile,
            "velocity_command_frame": env.config.velocity_command_frame,
            "velocity_reward": asdict(env.config.velocity_reward),
            "push_time_s": env.config.push_time_s,
            "push_linear_velocity": env.config.push_linear_velocity,
            "push_angular_velocity": env.config.push_angular_velocity,
            "external_force_mode": env.config.external_force_mode,
            "external_force_probability": env.config.external_force_probability,
            "external_force_body_names": env.config.external_force_body_names,
            "external_force_active_body_count": env.config.external_force_active_body_count,
            "external_force_event_count_range": env.config.external_force_event_count_range,
            "external_force_rest_s_range": env.config.external_force_rest_s_range,
            "external_force_start_s_range": env.config.external_force_start_s_range,
            "external_force_duration_s_range": env.config.external_force_duration_s_range,
            "external_force_min_n": env.config.external_force_min_n,
            "external_force_max_n": env.config.external_force_max_n,
            "external_force_z_fraction": env.config.external_force_z_fraction,
            "external_force_direction_angle_rad": env.config.external_force_direction_angle_rad,
            "external_force_torque_max_nm": env.config.external_force_torque_max_nm,
            "external_force_spring_stiffness_range": env.config.external_force_spring_stiffness_range,
            "external_force_spring_damping": env.config.external_force_spring_damping,
            "external_force_guiding_probability": env.config.external_force_guiding_probability,
            "external_force_transition_s": env.config.external_force_transition_s,
            "external_force_net_force_limit_n": env.config.external_force_net_force_limit_n,
            "external_force_net_torque_limit_nm": env.config.external_force_net_torque_limit_nm,
        },
    }


def _observation_sections(env: Go2StandBalanceEnv) -> list[dict[str, Any]]:
    if env.config.task in VELOCITY_TASKS:
        return _sections(
            env,
            [
                ("base_ang_vel", 3),
                ("projected_gravity", 3),
                ("command", 3),
                ("joint_pos_error", 12),
                ("joint_vel", 12),
                ("last_action", 12),
            ],
        )

    parts = [
        ("projected_gravity", 3),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
    ]
    if env.config.include_command:
        parts.append(("command", 3))
    parts.extend(
        [
            ("joint_pos_error", 12),
            ("joint_vel", 12),
            ("last_action", 12),
            ("contacts", 4),
        ]
    )
    return _sections(env, parts)


def _sections(env: Go2StandBalanceEnv, parts: list[tuple[str, int]]) -> list[dict[str, Any]]:
    sections = []
    cursor = 0
    names = list(env.observation_names)
    for name, width in parts:
        end = cursor + width
        sections.append(
            {
                "start": cursor,
                "end": end,
                "name": name,
                "items": names[cursor:end],
            }
        )
        cursor = end
    return sections


def _print_report(report: dict[str, Any]) -> None:
    print(f"Task: {report['task']}")
    print(f"Observation size: {report['observation_size']}")
    print("Observation:")
    for section in report["observation_sections"]:
        print(f"  [{section['start']}:{section['end']}] {section['name']}")
    print("")
    print(f"Action size: {report['action_size']}")
    action = report["action_sections"][0]
    print(f"Action:")
    print(f"  [{action['start']}:{action['end']}] {action['name']} scale={action['scale']} rad")
    print("")
    print("Config:")
    for key, value in report["config"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
