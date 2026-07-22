from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cli import (
    add_fullbody_reference_args,
    add_observation_mode_arg,
    add_policy_leg_order_arg,
    add_velocity_reward_args,
    parse_reward_overrides,
    resolve_policy_leg_order,
)
from .controller import PDConfig
from .export import _strip_raw_state_arrays
from .gym_env import Go2StandBalanceGymEnv
from .onnx_rollout import DEFAULT_REFERENCE_POLICY, _load_actor_obs_normalizer
from .rl_env import RL_TASKS, make_task_config
from .sb3_tools import load_vecnormalize_if_available, make_vec_env
from .trajectory import make_trajectory
from .trajectory_analysis import metrics_output_path, validate_trajectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a MuJoCo Go2 rollout with a scheduled joint torque perturbation."
    )
    policy_group = parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument(
        "--onnx-policy",
        type=Path,
        nargs="?",
        const=DEFAULT_REFERENCE_POLICY,
        help="Run an ONNX teacher policy.",
    )
    policy_group.add_argument(
        "--ppo-checkpoint",
        type=Path,
        help="Run a Stable-Baselines3 PPO checkpoint.",
    )
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument("--model", type=Path, default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"))
    parser.add_argument("--output", type=Path, default=Path("perturb_rollout.json"))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--export-dt", type=float, default=0.02)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--target-forward-velocity", type=float)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--action-smoothing", type=float)
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--reset-settle", type=float)
    parser.add_argument("--velocity-pose-profile", choices=("mjlab", "official"), default="official")
    add_policy_leg_order_arg(parser)
    add_observation_mode_arg(parser)
    add_fullbody_reference_args(parser)
    parser.add_argument("--pd-kp", type=float)
    parser.add_argument("--pd-kd", type=float)
    parser.add_argument("--torque-limit", type=float)
    add_velocity_reward_args(parser)
    parser.add_argument("--fixed-command", action="store_true")
    parser.add_argument("--randomize-commands", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--vecnormalize", type=Path)
    parser.add_argument("--no-vecnormalize", action="store_true")
    parser.add_argument("--normalizer-checkpoint", type=Path)
    parser.add_argument("--no-normalizer", action="store_true")
    parser.add_argument(
        "--perturb-joint",
        default="FR_thigh_joint",
        help="Joint receiving the extra torque, for example FR_thigh_joint.",
    )
    parser.add_argument("--perturb-time", type=float, default=2.0)
    parser.add_argument("--perturb-duration", type=float, default=0.25)
    parser.add_argument("--perturb-torque", type=float, default=6.0, help="Extra joint torque in Nm.")
    parser.add_argument("--recovery-window", type=float, default=0.5)
    parser.add_argument("--recovery-velocity-error", type=float, default=0.10)
    parser.add_argument("--recovery-base-z-min", type=float, default=0.30)
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

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
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        reward_scale=args.reward_scale,
        reset_settle_s=args.reset_settle,
        randomize_commands=randomize_commands,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
        velocity_reward_overrides=parse_reward_overrides(args.reward_term),
        observation_mode=args.observation_mode,
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

    env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
    if args.perturb_joint not in env.env.actuator_map:
        valid = ", ".join(env.env.joint_names)
        raise ValueError(f"unknown perturb joint {args.perturb_joint!r}; choose one of: {valid}")
    policy = _load_policy(args, env_config=env_config)

    obs, _ = env.reset()
    frames = [env.env.make_frame(frame_t=0.0)]
    next_export_t = args.export_dt
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    perturb_applied_steps = 0
    perturb_torque_sum = 0.0
    perturb_start = float(args.perturb_time)
    perturb_end = perturb_start + max(0.0, float(args.perturb_duration))

    while not terminated and not truncated:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = _step_with_joint_torque(
            env=env.env,
            action=action,
            joint_name=args.perturb_joint,
            perturb_start=perturb_start,
            perturb_end=perturb_end,
            perturb_torque=float(args.perturb_torque),
        )
        if info["perturb_active"]:
            perturb_applied_steps += 1
            perturb_torque_sum += abs(float(args.perturb_torque))
        total_reward += float(reward)
        steps += 1
        while env.env.elapsed_s + 1e-12 >= next_export_t:
            frame = env.env.make_frame(frame_t=next_export_t)
            if perturb_start <= next_export_t < perturb_end:
                frame["perturbation"] = {
                    "type": "joint_torque",
                    "joint": args.perturb_joint,
                    "torque": float(args.perturb_torque),
                }
            frames.append(frame)
            next_export_t += args.export_dt

    perturbation = {
        "type": "joint_torque",
        "joint": args.perturb_joint,
        "torque": float(args.perturb_torque),
        "start_t": perturb_start,
        "duration": max(0.0, float(args.perturb_duration)),
        "end_t": perturb_end,
        "applied_steps": perturb_applied_steps,
        "mean_abs_torque": perturb_torque_sum / max(perturb_applied_steps, 1),
    }
    recovery = _compute_recovery_metrics(
        frames,
        perturb_end=perturb_end,
        target_forward_velocity=float(env_config.target_forward_velocity),
        target_lateral_velocity=float(env_config.target_lateral_velocity),
        velocity_error_threshold=float(args.recovery_velocity_error),
        base_z_min_threshold=float(args.recovery_base_z_min),
        window_s=max(0.0, float(args.recovery_window)),
    )

    trajectory = make_trajectory(
        frames=frames,
        joint_order=list(env.env.joint_names),
        dt=args.export_dt,
        source=str(args.ppo_checkpoint or args.onnx_policy),
        extra_metadata={
            "mode": "mujoco-perturb-rollout",
            "preset": f"{args.task}_joint_torque_perturbation",
            "task": args.task,
            "policy": policy.label,
            "model": str(args.model),
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
            "observation_mode": env_config.observation_mode,
            "fullbody_reference_mode": env_config.fullbody_reference_mode,
            "nominal_reference_dataset": env_config.nominal_reference_dataset,
            "velocity_reward_profile": env_config.velocity_reward_profile,
            "velocity_command_frame": env_config.velocity_command_frame,
            "pd_kp": env_config.pd.kp,
            "pd_kd": env_config.pd.kd,
            "torque_limit": env_config.pd.torque_limit,
            "randomize_commands": env_config.randomize_commands,
            "deterministic": args.deterministic,
            "perturbation": perturbation,
            "recovery": recovery,
            "terminated": terminated,
            "truncated": truncated,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(steps, 1),
        },
        notes="MuJoCo policy rollout with scheduled joint torque perturbation for recovery analysis.",
    )
    if args.compact_json:
        _strip_raw_state_arrays(trajectory)

    result = validate_trajectory(trajectory)
    warning_events = list(result["metrics"].get("warning_events", []))
    warning_events.extend(_perturbation_events(perturbation, recovery, result["metrics"].get("fall_detected")))
    metrics = dict(result["metrics"])
    metrics["perturbation"] = perturbation
    metrics["recovery"] = recovery
    trajectory.setdefault("metadata", {})["metrics"] = metrics
    trajectory["metadata"]["validation"] = {
        "passed": result["passed"],
        "health_status": result["metrics"]["health_status"],
        "warnings": result["warnings"],
        "errors": result["errors"],
        "warning_events": warning_events,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    metrics_path = metrics_output_path(args.output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"wrote {args.output} ({len(frames)} frames)")
    print(f"policy: {policy.label}")
    print(f"task: {args.task}")
    print(f"steps: {steps}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"perturb_joint: {args.perturb_joint}")
    print(f"perturb_torque: {float(args.perturb_torque):.3f} Nm")
    print(f"perturb_time: {perturb_start:.3f}s")
    print(f"perturb_duration: {perturbation['duration']:.3f}s")
    print(f"recovered: {str(recovery['recovered']).lower()}")
    print(f"recovery_time: {_fmt(recovery['recovery_time_s'])}s")
    print(f"post_perturb_base_z_min: {_fmt(recovery['post_perturb_base_z_min'])} m")
    print(f"post_perturb_max_velocity_error_xy: {_fmt(recovery['post_perturb_max_velocity_error_xy'])} m/s")
    print(f"terminated: {str(terminated).lower()}")
    print(f"truncated: {str(truncated).lower()}")
    print(f"health: {result['metrics']['health_status']}")
    print(f"wrote {metrics_path}")
    return 0 if result["passed"] else 1


class _Policy:
    def __init__(self, *, label: str, actuator: Any) -> None:
        self.label = label
        self._actuator = actuator

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._actuator(obs)


def _load_policy(args: argparse.Namespace, *, env_config: Any) -> _Policy:
    if args.onnx_policy is not None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is missing. Install with: pip install -e '.[reference]'") from exc
        session = ort.InferenceSession(str(args.onnx_policy), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        normalizer = None
        if args.normalizer_checkpoint is not None and not args.no_normalizer:
            normalizer = _load_actor_obs_normalizer(args.normalizer_checkpoint)

        def _act(obs: np.ndarray) -> np.ndarray:
            policy_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            if normalizer is not None:
                policy_obs = normalizer(policy_obs)
            action = session.run([output_name], {input_name: policy_obs})[0]
            return np.asarray(action, dtype=np.float32).reshape(-1)

        return _Policy(label=f"onnx:{args.onnx_policy}", actuator=_act)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Stable-Baselines3 is missing. Install with: pip install -e '.[train]'") from exc

    vec_env = make_vec_env(model_path=args.model, config=env_config)
    normalizer, normalizer_path = load_vecnormalize_if_available(
        vec_env=vec_env,
        checkpoint_path=args.ppo_checkpoint,
        vecnormalize_path=args.vecnormalize,
        disabled=args.no_vecnormalize,
    )
    model = PPO.load(str(args.ppo_checkpoint), device="cpu")

    def _act(obs: np.ndarray) -> np.ndarray:
        policy_obs = normalizer.normalize_obs(obs) if normalizer_path else obs
        action, _ = model.predict(policy_obs, deterministic=args.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    label = f"ppo:{args.ppo_checkpoint}"
    if normalizer_path:
        label = f"{label} vecnormalize:{normalizer_path}"
    return _Policy(label=label, actuator=_act)


def _step_with_joint_torque(
    *,
    env: Any,
    action: np.ndarray,
    joint_name: str,
    perturb_start: float,
    perturb_end: float,
    perturb_torque: float,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    action_array = env.np.asarray(action, dtype=env.np.float64)
    if action_array.shape != (env.action_size,):
        raise ValueError(f"action must have shape ({env.action_size},), got {action_array.shape}")
    action_array = env.np.clip(action_array, -1.0, 1.0)
    previous_action = env._last_action.copy()
    previous_joint_vel = env._joint_vel_array()
    smoothing = max(0.0, min(0.98, float(env.config.action_smoothing)))
    filtered_action = smoothing * previous_action + (1.0 - smoothing) * action_array
    env._apply_action(filtered_action)
    env._maybe_apply_push()
    perturb_active = perturb_start <= env.elapsed_s < perturb_end
    if perturb_active:
        actuator_id = env.actuator_map[joint_name]
        env.data.ctrl[actuator_id] += float(perturb_torque)
    env._last_action = filtered_action

    for _ in range(env._control_steps):
        env.mujoco.mj_step(env.model, env.data)
    env._elapsed_s += env._control_steps * env.model.opt.timestep

    reward, raw_reward_terms, reward_terms = env._reward(previous_action, previous_joint_vel)
    env._previous_joint_vel = env._joint_vel_array()
    reward *= env.config.reward_scale
    raw_reward_terms = {
        name: value * env.config.reward_scale
        for name, value in raw_reward_terms.items()
    }
    reward_terms = {
        name: value * env.config.reward_scale
        for name, value in reward_terms.items()
    }
    terminated = env._terminated()
    truncated = env.elapsed_s >= env.config.episode_length_s
    info = env._info(reward_terms=reward_terms, reward_raw_terms=raw_reward_terms)
    info["perturb_active"] = bool(perturb_active)
    info["perturb_joint"] = joint_name if perturb_active else None
    info["perturb_torque"] = float(perturb_torque) if perturb_active else 0.0
    return env.observation(), float(reward), bool(terminated), bool(truncated), info


def _compute_recovery_metrics(
    frames: list[dict[str, Any]],
    *,
    perturb_end: float,
    target_forward_velocity: float,
    target_lateral_velocity: float,
    velocity_error_threshold: float,
    base_z_min_threshold: float,
    window_s: float,
) -> dict[str, Any]:
    post = [frame for frame in frames if float(frame.get("t", 0.0)) >= perturb_end]
    if not post:
        return {
            "recovered": False,
            "recovery_time_s": None,
            "post_perturb_base_z_min": None,
            "post_perturb_max_velocity_error_xy": None,
            "velocity_error_threshold": velocity_error_threshold,
            "base_z_min_threshold": base_z_min_threshold,
            "window_s": window_s,
        }

    velocity_errors = [
        _velocity_error_xy(frame, target_forward_velocity, target_lateral_velocity)
        for frame in post
    ]
    base_z_values = [float((frame.get("base") or {}).get("position", [0.0, 0.0, 0.0])[2]) for frame in post]
    window_count = max(1, round(window_s / max(_frame_dt(frames), 1e-6)))
    recovered_t: float | None = None
    for index in range(0, max(1, len(post) - window_count + 1)):
        window_errors = velocity_errors[index:index + window_count]
        window_base_z = base_z_values[index:index + window_count]
        if not window_errors or not window_base_z:
            continue
        if max(window_errors) <= velocity_error_threshold and min(window_base_z) >= base_z_min_threshold:
            recovered_t = float(post[index].get("t", 0.0))
            break

    return {
        "recovered": recovered_t is not None,
        "recovery_time_s": None if recovered_t is None else max(0.0, recovered_t - perturb_end),
        "post_perturb_base_z_min": min(base_z_values),
        "post_perturb_max_velocity_error_xy": max(velocity_errors),
        "velocity_error_threshold": velocity_error_threshold,
        "base_z_min_threshold": base_z_min_threshold,
        "window_s": window_s,
    }


def _perturbation_events(
    perturbation: dict[str, Any],
    recovery: dict[str, Any],
    fall_detected: Any,
) -> list[dict[str, Any]]:
    events = [
        {
            "type": "joint_torque_perturbation",
            "severity": "warning",
            "t": perturbation["start_t"],
            "message": (
                f"{perturbation['joint']} torque perturbation "
                f"{perturbation['torque']:.1f}Nm started"
            ),
        },
        {
            "type": "joint_torque_perturbation_end",
            "severity": "warning",
            "t": perturbation["end_t"],
            "message": f"{perturbation['joint']} perturbation ended",
        },
    ]
    if recovery.get("recovered"):
        events.append({
            "type": "recovered",
            "severity": "warning",
            "t": perturbation["end_t"] + float(recovery.get("recovery_time_s") or 0.0),
            "message": f"recovered after {_fmt(recovery.get('recovery_time_s'))}s",
        })
    else:
        events.append({
            "type": "recovery_failed",
            "severity": "error" if fall_detected else "warning",
            "t": perturbation["end_t"],
            "message": "recovery threshold was not reached",
        })
    return events


def _velocity_error_xy(frame: dict[str, Any], target_forward_velocity: float, target_lateral_velocity: float) -> float:
    velocity = (frame.get("base") or {}).get("linear_velocity") or [0.0, 0.0, 0.0]
    return float(np.hypot(float(velocity[0]) - target_forward_velocity, float(velocity[1]) - target_lateral_velocity))


def _frame_dt(frames: list[dict[str, Any]]) -> float:
    if len(frames) < 2:
        return 0.02
    return float(frames[1].get("t", 0.02)) - float(frames[0].get("t", 0.0))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
