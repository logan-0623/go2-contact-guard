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
            "Roll out a Go2 env and audit raw/normalized observation statistics. "
            "Use this before teacher distillation when privileged observations change."
        )
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Optional PPO checkpoint. If omitted, --action-source auto uses zero actions.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
    )
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument("--episodes", type=int, default=4)
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
    add_fullbody_reference_args(parser)
    parser.set_defaults(fullbody_reference_mode="phase_trot")
    add_velocity_reward_args(parser)
    parser.set_defaults(velocity_reward_profile="gentle_fullbody_teacher_impedance_tracking")
    add_external_force_args(parser)
    parser.add_argument("--pd-kp", type=float, default=20.0)
    parser.add_argument("--pd-kd", type=float, default=1.0)
    parser.add_argument("--torque-limit", type=float, default=23.5)
    parser.add_argument("--randomize-commands", action="store_true")
    parser.add_argument("--fixed-command", action="store_true")
    parser.add_argument(
        "--action-source",
        choices=("auto", "policy", "zero", "random"),
        default="auto",
        help="Action source used during audit rollout.",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vecnormalize", type=Path)
    parser.add_argument("--no-vecnormalize", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top", type=int, default=12, help="Number of worst dimensions printed per section.")
    parser.add_argument(
        "--constant-std-threshold",
        type=float,
        default=1e-6,
        help="Stddev threshold for flagging constant observation dimensions.",
    )
    parser.add_argument(
        "--raw-abs-threshold",
        type=float,
        default=50.0,
        help="Warn when any raw observation dimension exceeds this absolute value.",
    )
    parser.add_argument(
        "--normalized-abs-threshold",
        type=float,
        default=8.0,
        help="Warn when any normalized observation dimension approaches VecNormalize clipping.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit nonzero when audit checks emit warnings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/obs_audit.json"),
        help="Output JSON audit report.",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    randomize_commands = None
    if args.randomize_commands:
        randomize_commands = True
    if args.fixed_command:
        randomize_commands = False

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
    env_config = replace(
        env_config,
        pd=PDConfig(kp=args.pd_kp, kd=args.pd_kd, torque_limit=args.torque_limit),
    )

    env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
    names = list(env.env.observation_names)
    model = None
    if args.checkpoint is not None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "Stable-Baselines3 is missing. Install with: pip install -e '.[train]'"
            ) from exc
        model = PPO.load(str(args.checkpoint), device="cpu")

    normalizer = None
    normalizer_path = None
    if not args.no_vecnormalize and (args.checkpoint is not None or args.vecnormalize is not None):
        vec_env = make_vec_env(model_path=args.model, config=env_config)
        normalizer, normalizer_path = load_vecnormalize_if_available(
            vec_env=vec_env,
            checkpoint_path=args.checkpoint or args.vecnormalize,
            vecnormalize_path=args.vecnormalize,
            disabled=False,
        )
        if normalizer_path is None:
            normalizer = None

    action_source = args.action_source
    if action_source == "auto":
        action_source = "policy" if model is not None else "zero"
    if action_source == "policy" and model is None:
        raise ValueError("--action-source policy requires a checkpoint")

    raw_observations = []
    normalized_observations = []
    external_force_active_steps = 0
    total_steps = 0

    for episode in range(max(1, args.episodes)):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            raw_observations.append(obs.copy())
            policy_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            if normalizer is not None:
                normalized_observations.append(policy_obs.copy())

            if action_source == "policy":
                action, _ = model.predict(policy_obs, deterministic=args.deterministic)
            elif action_source == "random":
                action = env.action_space.sample()
            else:
                action = env.env.zero_action()

            obs, _, terminated, truncated, info = env.step(action)
            external_force_active_steps += int(bool(info.get("external_force_active", False)))
            total_steps += 1

    if not raw_observations:
        raise RuntimeError("observation audit produced no samples")

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for observation audit") from exc

    raw = np.stack(raw_observations).astype(np.float64)
    normalized = (
        np.stack(normalized_observations).astype(np.float64)
        if normalized_observations
        else None
    )
    report = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "vecnormalize": str(normalizer_path) if normalizer_path else None,
        "action_source": action_source,
        "samples": int(raw.shape[0]),
        "observation_size": int(raw.shape[1]),
        "observation_mode": env_config.observation_mode,
        "velocity_reward_profile": env_config.velocity_reward_profile,
        "velocity_reward_overrides": velocity_reward_overrides,
        "external_force_active_fraction": external_force_active_steps / max(total_steps, 1),
        "raw": _stats_report(raw, names),
        "raw_groups": _group_stats(raw, names),
        "normalized": _stats_report(normalized, names) if normalized is not None else None,
        "normalized_groups": _group_stats(normalized, names) if normalized is not None else None,
    }
    report["checks"] = _audit_checks(
        report,
        constant_std_threshold=args.constant_std_threshold,
        raw_abs_threshold=args.raw_abs_threshold,
        normalized_abs_threshold=args.normalized_abs_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report, top=max(1, args.top), constant_std_threshold=args.constant_std_threshold)
    print(f"wrote {args.output}")
    if args.fail_on_warning and any(check["level"] == "warning" for check in report["checks"]):
        return 1
    return 0


def _stats_report(values: Any, names: Sequence[str]) -> list[dict[str, Any]]:
    import numpy as np

    stats = []
    for index, name in enumerate(names):
        column = values[:, index]
        finite = np.isfinite(column)
        finite_values = column[finite]
        if finite_values.size:
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values))
            minimum = float(np.min(finite_values))
            maximum = float(np.max(finite_values))
            abs_max = float(np.max(np.abs(finite_values)))
            zero_fraction = float(np.mean(np.abs(finite_values) <= 1e-8))
        else:
            mean = std = minimum = maximum = abs_max = zero_fraction = None
        stats.append(
            {
                "index": index,
                "name": name,
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
                "abs_max": abs_max,
                "zero_fraction": zero_fraction,
                "nonfinite_count": int(column.size - finite_values.size),
            }
        )
    return stats


def _group_stats(values: Any, names: Sequence[str]) -> dict[str, dict[str, float | int]]:
    groups = {
        "external_force": [i for i, name in enumerate(names) if "external_force" in name],
        "target_root": [i for i, name in enumerate(names) if "target_root" in name],
        "keypoint_target_diff": [i for i, name in enumerate(names) if "target_diff" in name],
        "compliant_offset": [i for i, name in enumerate(names) if "compliant_offset" in name],
        "compliant_velocity": [i for i, name in enumerate(names) if "compliant_vel" in name],
        "reference_velocity": [i for i, name in enumerate(names) if "reference_vel" in name],
        "reference_driving_force": [i for i, name in enumerate(names) if "reference_driving_force" in name],
        "reference_interaction_force": [
            i for i, name in enumerate(names) if "reference_interaction_force" in name
        ],
        "actual_interaction_force": [i for i, name in enumerate(names) if "actual_interaction_force" in name],
        "interaction_force_error": [i for i, name in enumerate(names) if "interaction_force_error" in name],
        "contact_force": [i for i, name in enumerate(names) if "contact_force" in name],
        "ctrl": [i for i, name in enumerate(names) if name.endswith("_ctrl")],
    }
    return {
        name: _one_group_stats(values[:, indices]) if indices else _empty_group()
        for name, indices in groups.items()
    }


def _one_group_stats(values: Any) -> dict[str, float | int]:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "dims": int(values.shape[1]),
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "mean_std": 0.0,
            "zero_fraction": 1.0,
            "nonfinite_count": int(values.size),
        }
    return {
        "dims": int(values.shape[1]),
        "mean_abs": float(np.mean(np.abs(finite))),
        "max_abs": float(np.max(np.abs(finite))),
        "mean_std": float(np.mean(np.std(values, axis=0))),
        "zero_fraction": float(np.mean(np.abs(finite) <= 1e-8)),
        "nonfinite_count": int(values.size - finite.size),
    }


def _empty_group() -> dict[str, float | int]:
    return {
        "dims": 0,
        "mean_abs": 0.0,
        "max_abs": 0.0,
        "mean_std": 0.0,
        "zero_fraction": 1.0,
        "nonfinite_count": 0,
    }


def _audit_checks(
    report: dict[str, Any],
    *,
    constant_std_threshold: float,
    raw_abs_threshold: float,
    normalized_abs_threshold: float,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    raw_rows = report["raw"]
    raw_nonfinite = sum(int(row.get("nonfinite_count") or 0) for row in raw_rows)
    if raw_nonfinite > 0:
        checks.append({"level": "warning", "message": f"raw observations contain {raw_nonfinite} nonfinite values"})

    raw_max = max((float(row["abs_max"]) for row in raw_rows if row.get("abs_max") is not None), default=0.0)
    if raw_max > raw_abs_threshold:
        checks.append({
            "level": "warning",
            "message": f"raw observation abs_max {raw_max:.4g} exceeds threshold {raw_abs_threshold:.4g}",
        })

    normalized_rows = report.get("normalized")
    if normalized_rows is not None:
        normalized_nonfinite = sum(int(row.get("nonfinite_count") or 0) for row in normalized_rows)
        if normalized_nonfinite > 0:
            checks.append({
                "level": "warning",
                "message": f"normalized observations contain {normalized_nonfinite} nonfinite values",
            })
        normalized_max = max(
            (float(row["abs_max"]) for row in normalized_rows if row.get("abs_max") is not None),
            default=0.0,
        )
        if normalized_max > normalized_abs_threshold:
            checks.append({
                "level": "warning",
                "message": (
                    f"normalized observation abs_max {normalized_max:.4g} exceeds threshold "
                    f"{normalized_abs_threshold:.4g}"
                ),
            })

    groups = report["raw_groups"]
    if report["observation_mode"] == "privileged":
        required_groups = (
            "target_root",
            "keypoint_target_diff",
            "compliant_offset",
            "compliant_velocity",
            "reference_velocity",
            "reference_driving_force",
            "reference_interaction_force",
            "actual_interaction_force",
        )
        for group_name in required_groups:
            stats = groups.get(group_name, {})
            if int(stats.get("dims") or 0) <= 0:
                checks.append({"level": "warning", "message": f"missing privileged obs group {group_name}"})
        if float(report.get("external_force_active_fraction") or 0.0) > 0.0:
            for group_name in ("reference_interaction_force", "actual_interaction_force"):
                stats = groups.get(group_name, {})
                if int(stats.get("dims") or 0) > 0 and float(stats.get("zero_fraction") or 0.0) >= 0.999:
                    checks.append({
                        "level": "warning",
                        "message": f"{group_name} stayed zero despite active external force",
                    })

    constant_count = sum(
        1
        for row in raw_rows
        if row.get("std") is not None and float(row["std"]) <= constant_std_threshold
    )
    if constant_count:
        checks.append({"level": "info", "message": f"{constant_count} raw observation dimensions are constant"})

    if not checks:
        checks.append({"level": "ok", "message": "observation audit checks passed"})
    return checks


def _print_summary(report: dict[str, Any], *, top: int, constant_std_threshold: float) -> None:
    print(f"samples: {report['samples']}")
    print(f"observation_size: {report['observation_size']}")
    print(f"action_source: {report['action_source']}")
    print(f"external_force_active_fraction: {report['external_force_active_fraction']:.3f}")
    _print_group_summary("raw_groups", report["raw_groups"])
    if report["normalized_groups"] is not None:
        _print_group_summary("normalized_groups", report["normalized_groups"])
    _print_worst("raw abs_max", report["raw"], key="abs_max", top=top)
    _print_constant("raw constant", report["raw"], threshold=constant_std_threshold, top=top)
    _print_nonfinite("raw nonfinite", report["raw"], top=top)
    if report["normalized"] is not None:
        _print_worst("normalized abs_max", report["normalized"], key="abs_max", top=top)
        _print_constant("normalized constant", report["normalized"], threshold=constant_std_threshold, top=top)
        _print_nonfinite("normalized nonfinite", report["normalized"], top=top)
    print("checks:")
    for check in report.get("checks", []):
        print(f"  {check['level']}: {check['message']}")


def _print_group_summary(label: str, groups: dict[str, dict[str, float | int]]) -> None:
    print(label + ":")
    for name, stats in groups.items():
        print(
            f"  {name}: dims={stats['dims']} "
            f"mean_abs={float(stats['mean_abs']):.4g} "
            f"max_abs={float(stats['max_abs']):.4g} "
            f"zero_fraction={float(stats['zero_fraction']):.3f} "
            f"nonfinite={stats['nonfinite_count']}"
        )


def _print_worst(label: str, rows: list[dict[str, Any]], *, key: str, top: int) -> None:
    ranked = sorted(
        (row for row in rows if row.get(key) is not None),
        key=lambda row: float(row[key]),
        reverse=True,
    )[:top]
    print(label + ":")
    for row in ranked:
        print(f"  {row['index']:03d} {row['name']}: {key}={float(row[key]):.4g}")


def _print_constant(label: str, rows: list[dict[str, Any]], *, threshold: float, top: int) -> None:
    selected = [
        row
        for row in rows
        if row.get("std") is not None and float(row["std"]) <= threshold
    ][:top]
    print(label + ":")
    if not selected:
        print("  none")
    for row in selected:
        print(
            f"  {row['index']:03d} {row['name']}: "
            f"std={float(row['std']):.4g} zero_fraction={float(row['zero_fraction']):.3f}"
        )


def _print_nonfinite(label: str, rows: list[dict[str, Any]], *, top: int) -> None:
    selected = [
        row
        for row in rows
        if int(row.get("nonfinite_count") or 0) > 0
    ][:top]
    print(label + ":")
    if not selected:
        print("  none")
    for row in selected:
        print(f"  {row['index']:03d} {row['name']}: nonfinite={row['nonfinite_count']}")


if __name__ == "__main__":
    raise SystemExit(main())
