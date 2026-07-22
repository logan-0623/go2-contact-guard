from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .cli import add_policy_leg_order_arg, resolve_policy_leg_order
from .controller import PDConfig
from .export import _strip_raw_state_arrays
from .gym_env import Go2StandBalanceGymEnv
from .rl_env import DEFAULT_SLOW_WALK_VELOCITY, RL_TASKS, make_task_config
from .trajectory import make_trajectory
from .trajectory_analysis import metrics_output_path, validate_trajectory


DEFAULT_REFERENCE_POLICY = Path(
    "external/reference_policies/unitree_go2_velocity_flat/policy.onnx"
)
DEFAULT_EXPORT_DT = 0.02


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a MuJoCo trajectory rollout from an ONNX Go2 reference policy."
    )
    parser.add_argument(
        "policy",
        type=Path,
        nargs="?",
        default=DEFAULT_REFERENCE_POLICY,
        help="ONNX actor policy path.",
    )
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
        default=Path("onnx_rollout.json"),
        help="Output trajectory JSON path.",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--export-dt",
        type=float,
        default=DEFAULT_EXPORT_DT,
        help="Trajectory export timestep in seconds. Defaults to the 50 Hz control rate.",
    )
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument(
        "--target-forward-velocity",
        type=float,
        default=DEFAULT_SLOW_WALK_VELOCITY,
        help="Fixed x velocity command in m/s.",
    )
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument(
        "--velocity-pose-profile",
        choices=("mjlab", "official"),
        default="mjlab",
        help="Default joint pose profile used by the rollout environment.",
    )
    add_policy_leg_order_arg(parser)
    parser.add_argument(
        "--action-scale",
        type=float,
        help="Joint target action scale. Defaults to 0.5 for mjlab, otherwise task default.",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        help="EMA action smoothing. Defaults to 0 for ONNX reference rollout.",
    )
    parser.add_argument(
        "--reset-settle",
        type=float,
        default=0.0,
        help="Seconds to settle with zero action after reset. Defaults to 0 for ONNX reference rollout.",
    )
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--pd-kp", type=float, default=20.0)
    parser.add_argument("--pd-kd", type=float, default=1.0)
    parser.add_argument("--torque-limit", type=float, default=23.5)
    parser.add_argument(
        "--normalizer-checkpoint",
        type=Path,
        help=(
            "RSL-RL checkpoint containing actor_obs_normalizer stats. "
            "Only use this for ONNX files that do not already include a normalizer."
        ),
    )
    parser.add_argument(
        "--no-normalizer",
        action="store_true",
        help="Feed raw observations to the ONNX policy.",
    )
    parser.add_argument("--compact-json", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    if not args.policy.exists():
        raise FileNotFoundError(
            f"ONNX policy not found: {args.policy}\n"
            "Download the reference policy with:\n"
            "  mkdir -p external/reference_policies/unitree_go2_velocity_flat\n"
            "  curl -L https://huggingface.co/diasAiMaster/unitree-go2-velocity-flat/resolve/main/policy.onnx "
            "-o external/reference_policies/unitree_go2_velocity_flat/policy.onnx\n"
            "  curl -L https://huggingface.co/diasAiMaster/unitree-go2-velocity-flat/resolve/main/policy.onnx.data "
            "-o external/reference_policies/unitree_go2_velocity_flat/policy.onnx.data"
        )

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is missing. Install reference dependencies with: "
            "pip install -e '.[reference]'"
        ) from exc

    action_scale = args.action_scale
    if action_scale is None and args.velocity_pose_profile == "mjlab":
        action_scale = 0.5
    action_smoothing = 0.0 if args.action_smoothing is None else args.action_smoothing

    env_config = make_task_config(
        task=args.task,
        control_dt=args.control_dt,
        episode_length_s=args.duration,
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        action_scale=action_scale,
        action_smoothing=action_smoothing,
        reward_scale=args.reward_scale,
        reset_settle_s=args.reset_settle,
        randomize_commands=False,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
    )
    env_config = replace(
        env_config,
        pd=PDConfig(kp=args.pd_kp, kd=args.pd_kd, torque_limit=args.torque_limit),
    )
    env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    normalizer = None
    normalizer_path = None
    if args.normalizer_checkpoint is not None and not args.no_normalizer:
        normalizer_path = args.normalizer_checkpoint
        normalizer = _load_actor_obs_normalizer(normalizer_path)

    obs, _ = env.reset()
    frames = [env.env.make_frame(frame_t=0.0)]
    next_export_t = args.export_dt
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        policy_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if normalizer is not None:
            policy_obs = normalizer(policy_obs)
        action = session.run([output_name], {input_name: policy_obs})[0]
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if env.env.elapsed_s + 1e-12 >= next_export_t:
            frames.append(env.env.make_frame(frame_t=next_export_t))
            next_export_t += args.export_dt

    trajectory = make_trajectory(
        frames=frames,
        joint_order=list(env.env.joint_names),
        dt=args.export_dt,
        source=str(args.policy),
        extra_metadata={
            "mode": "mujoco-onnx-reference-rollout",
            "preset": "unitree_go2_velocity_flat_reference",
            "task": args.task,
            "policy": "onnx",
            "reference_policy": "diasAiMaster/unitree-go2-velocity-flat",
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
            "reset_settle_s": env_config.reset_settle_s,
            "reward_scale": env_config.reward_scale,
            "velocity_pose_profile": env_config.velocity_pose_profile,
            "policy_leg_order": args.policy_leg_order,
            "velocity_reward_profile": env_config.velocity_reward_profile,
            "pd_kp": env_config.pd.kp,
            "pd_kd": env_config.pd.kd,
            "torque_limit": env_config.pd.torque_limit,
            "onnx_input": input_name,
            "onnx_output": output_name,
            "normalizer_checkpoint": str(normalizer_path) if normalizer_path else None,
            "terminated": terminated,
            "truncated": truncated,
            "total_reward": total_reward,
            "mean_reward": total_reward / max(steps, 1),
        },
        notes="Rollout from a pretrained ONNX reference policy in the MuJoCo RL environment.",
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
    metrics_path = metrics_output_path(args.output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")

    print(f"wrote {args.output} ({len(frames)} frames)")
    print(f"policy: {args.policy}")
    print(f"task: {args.task}")
    print(f"steps: {steps}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"command: {[round(float(v), 4) for v in env.env._command()]}")
    print(f"velocity_pose_profile: {env_config.velocity_pose_profile}")
    print(f"policy_leg_order: {args.policy_leg_order}")
    print(f"action_scale: {env_config.action_scale}")
    print(f"action_smoothing: {env_config.action_smoothing}")
    print(f"reset_settle_s: {env_config.reset_settle_s}")
    print(f"pd_kp: {env_config.pd.kp}")
    print(f"pd_kd: {env_config.pd.kd}")
    print(f"torque_limit: {env_config.pd.torque_limit}")
    print(f"normalizer: {normalizer_path if normalizer_path else 'none'}")
    print(f"total_reward: {total_reward:.3f}")
    print(f"mean_reward: {total_reward / max(steps, 1):.3f}")
    print(f"terminated: {str(terminated).lower()}")
    print(f"truncated: {str(truncated).lower()}")
    print(f"health: {result['metrics']['health_status']}")
    print(f"wrote {metrics_path}")
    return 0 if result["passed"] else 1


def _load_actor_obs_normalizer(path: Path):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Loading RSL-RL normalizer stats requires torch. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mean = state_dict.get("actor_obs_normalizer._mean")
    std = state_dict.get("actor_obs_normalizer._std")
    if mean is None or std is None:
        raise ValueError(f"checkpoint does not contain actor_obs_normalizer stats: {path}")
    mean_array = mean.detach().cpu().numpy().astype(np.float32)
    std_array = std.detach().cpu().numpy().astype(np.float32)

    def _normalize(obs):
        return ((obs - mean_array) / np.maximum(std_array, 1e-6)).astype(np.float32)

    return _normalize


if __name__ == "__main__":
    raise SystemExit(main())
