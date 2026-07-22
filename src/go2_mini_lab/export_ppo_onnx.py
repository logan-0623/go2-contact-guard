from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .cli import add_fullbody_reference_args, add_observation_mode_arg, add_policy_leg_order_arg
from .rl_env import RL_TASKS, make_task_config
from .sb3_tools import default_vecnormalize_path, load_vecnormalize_if_available, make_vec_env
from .cli import resolve_policy_leg_order


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a deterministic SB3 PPO actor to ONNX for deployment."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        help="VecNormalize stats path. Defaults to <checkpoint>_vecnormalize.pkl when present.",
    )
    parser.add_argument("--output", type=Path, default=Path("exported/go2_student_policy.onnx"))
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
    )
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--target-forward-velocity", type=float, default=0.40)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--action-smoothing", type=float, default=0.15)
    parser.add_argument("--reset-settle", type=float, default=0.0)
    parser.add_argument("--velocity-pose-profile", choices=("mjlab", "official"), default="official")
    add_policy_leg_order_arg(parser)
    add_observation_mode_arg(parser)
    add_fullbody_reference_args(parser)
    parser.add_argument("--no-vecnormalize", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        import numpy as np
        import torch as th
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing. Install with: pip install -e '.[train]'") from exc

    env_config = make_task_config(
        task=args.task,
        control_dt=args.control_dt,
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        reset_settle_s=args.reset_settle,
        randomize_commands=False,
        observation_mode=args.observation_mode,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        fullbody_reference_mode=args.fullbody_reference_mode,
        gait_frequency_hz=args.gait_frequency,
        gait_step_length=args.gait_step_length,
        gait_swing_height=args.gait_swing_height,
        gait_joint_thigh_amplitude=args.gait_thigh_amplitude,
        gait_joint_calf_amplitude=args.gait_calf_amplitude,
    )
    vec_env = make_vec_env(model_path=args.model, config=env_config)
    normalizer, normalizer_path = load_vecnormalize_if_available(
        vec_env=vec_env,
        checkpoint_path=args.checkpoint,
        vecnormalize_path=args.vecnormalize,
        disabled=args.no_vecnormalize,
    )
    model = PPO.load(str(args.checkpoint), device=args.device)
    model.policy.set_training_mode(False)

    observation_size = int(vec_env.observation_space.shape[0])
    if normalizer_path:
        mean = normalizer.obs_rms.mean.astype(np.float32)
        var = normalizer.obs_rms.var.astype(np.float32)
        clip_obs = float(normalizer.clip_obs)
        epsilon = float(normalizer.epsilon)
    else:
        mean = np.zeros(observation_size, dtype=np.float32)
        var = np.ones(observation_size, dtype=np.float32)
        clip_obs = 1.0e9
        epsilon = 1.0e-8

    actor = _make_normalized_deterministic_actor(
        model.policy,
        mean=th.as_tensor(mean, dtype=th.float32, device=model.policy.device),
        var=th.as_tensor(var, dtype=th.float32, device=model.policy.device),
        clip_obs=clip_obs,
        epsilon=epsilon,
    )
    actor.eval()

    dummy_obs = th.zeros(1, observation_size, dtype=th.float32, device=model.policy.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        th.onnx.export(
            actor,
            dummy_obs,
            str(args.output),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=args.opset,
            dynamo=False,
        )
    except Exception as exc:
        if "onnx" in str(exc).lower() or "onnxscript" in str(exc).lower():
            raise RuntimeError(
                "ONNX export dependencies are missing. Install the updated training extras with "
                "pip install -e '.[train]', or install onnx and onnxscript in this venv."
            ) from exc
        raise

    print(f"checkpoint: {args.checkpoint}")
    print(f"vecnormalize: {normalizer_path if normalizer_path else 'none'}")
    print(f"observation_size: {observation_size}")
    print(f"action_size: {int(vec_env.action_space.shape[0])}")
    print(f"observation_mode: {env_config.observation_mode}")
    print(f"fullbody_reference_mode: {env_config.fullbody_reference_mode}")
    print(f"wrote {args.output}")
    return 0


def _make_normalized_deterministic_actor(
    policy: object,
    *,
    mean: object,
    var: object,
    clip_obs: float,
    epsilon: float,
) -> object:
    import torch as th

    class _NormalizedDeterministicActor(th.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = policy
            self.register_buffer("mean", mean)
            self.register_buffer("var", var)
            self.clip_obs = float(clip_obs)
            self.epsilon = float(epsilon)

        def forward(self, observation: object) -> object:
            obs = (observation - self.mean) / th.sqrt(self.var + self.epsilon)
            obs = th.clamp(obs, -self.clip_obs, self.clip_obs)
            distribution = self.policy.get_distribution(obs)
            torch_distribution = getattr(distribution, "distribution", None)
            if torch_distribution is not None and hasattr(torch_distribution, "mean"):
                return torch_distribution.mean
            return distribution.mode()

    return _NormalizedDeterministicActor()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
