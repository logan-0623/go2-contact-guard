from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .controller import PDConfig
from .cli import (
    POLICY_LEG_ORDER_PROFILES,
    add_fullbody_reference_args,
    add_observation_mode_arg,
    resolve_policy_leg_order,
)
from .rl_env import RL_TASKS, make_task_config
from .sb3_tools import (
    default_vecnormalize_path,
    make_vec_env,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Behavior-clone a PPO actor from reference obs/action pairs, then save "
            "a checkpoint that can initialize PPO fine-tuning."
        )
    )
    parser.add_argument("dataset", type=Path, help="Reference dataset .npz.")
    parser.add_argument(
        "--task",
        choices=RL_TASKS,
        help="Training task. Defaults to the dataset metadata task.",
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
        default=Path("checkpoints/go2_velocity_flat_bc_ppo.zip"),
        help="Output PPO checkpoint initialized by BC.",
    )
    parser.add_argument(
        "--vecnormalize-output",
        type=Path,
        help="Output VecNormalize stats path. Defaults to <checkpoint>_vecnormalize.pkl.",
    )
    parser.add_argument("--episode-length", type=float, default=20.0)
    parser.add_argument("--target-forward-velocity", type=float)
    parser.add_argument("--target-lateral-velocity", type=float)
    parser.add_argument("--target-yaw-rate", type=float)
    parser.add_argument(
        "--velocity-pose-profile",
        choices=("mjlab", "official"),
        help="Default joint pose profile. Defaults to dataset metadata when present.",
    )
    parser.add_argument(
        "--policy-leg-order",
        choices=sorted(POLICY_LEG_ORDER_PROFILES),
        help="Policy joint block leg order. Defaults to dataset metadata when present.",
    )
    add_observation_mode_arg(parser)
    add_fullbody_reference_args(parser)
    parser.add_argument(
        "--reset-settle",
        type=float,
        help="Seconds to settle with zero action after reset. Defaults to dataset metadata when present.",
    )
    parser.add_argument("--pd-kp", type=float, help="Joint PD stiffness. Defaults to dataset metadata when present.")
    parser.add_argument("--pd-kd", type=float, help="Joint PD damping. Defaults to dataset metadata when present.")
    parser.add_argument("--torque-limit", type=float, help="Joint torque limit. Defaults to dataset metadata when present.")
    parser.add_argument(
        "--normalize-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit VecNormalize observation stats from the BC dataset.",
    )
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save VecNormalize with reward normalization enabled for PPO fine-tuning.",
    )
    parser.add_argument("--clip-observation", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--report",
        type=Path,
        help="Output BC report JSON path. Defaults to <checkpoint>.bc.json.",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        import numpy as np
        import torch as th
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    data = np.load(args.dataset, allow_pickle=False)
    observations = np.asarray(data["observations"], dtype=np.float32)
    actions = np.clip(np.asarray(data["actions"], dtype=np.float32), -1.0, 1.0)
    metadata = _load_metadata(data)

    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("dataset observations and actions must be 2D arrays")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            "dataset sample count mismatch: "
            f"{observations.shape[0]} observations vs {actions.shape[0]} actions"
        )

    task = args.task or str(metadata.get("task") or "velocity_flat")
    control_dt = _metadata_float(None, metadata, "control_dt", 0.02)
    target_forward_velocity = _metadata_float(
        args.target_forward_velocity,
        metadata,
        "target_forward_velocity",
        None,
    )
    target_lateral_velocity = _metadata_float(
        args.target_lateral_velocity,
        metadata,
        "target_lateral_velocity",
        0.0,
    )
    target_yaw_rate = _metadata_float(
        args.target_yaw_rate,
        metadata,
        "target_yaw_rate",
        0.0,
    )
    action_scale = _metadata_float(None, metadata, "action_scale", None)
    action_smoothing = _metadata_float(None, metadata, "action_smoothing", None)
    reward_scale = _metadata_float(None, metadata, "reward_scale", None)
    reset_settle_s = _metadata_float(args.reset_settle, metadata, "reset_settle_s", None)
    velocity_pose_profile = args.velocity_pose_profile or str(
        metadata.get("velocity_pose_profile") or "official"
    )
    policy_leg_order_profile = args.policy_leg_order or str(
        metadata.get("policy_leg_order") or "actuator"
    )
    observation_mode = (
        args.observation_mode
        if args.observation_mode != "policy" or "observation_mode" not in metadata
        else str(metadata.get("observation_mode") or "policy")
    )
    fullbody_reference_mode = (
        args.fullbody_reference_mode
        if args.fullbody_reference_mode != "static" or "fullbody_reference_mode" not in metadata
        else str(metadata.get("fullbody_reference_mode") or "static")
    )

    env_config = make_task_config(
        task=task,
        control_dt=control_dt,
        episode_length_s=args.episode_length,
        target_forward_velocity=target_forward_velocity,
        target_lateral_velocity=target_lateral_velocity,
        target_yaw_rate=target_yaw_rate,
        action_scale=action_scale,
        action_smoothing=action_smoothing,
        reward_scale=reward_scale,
        reset_settle_s=reset_settle_s,
        randomize_commands=False,
        observation_mode=observation_mode,
        velocity_pose_profile=velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(policy_leg_order_profile),
        fullbody_reference_mode=fullbody_reference_mode,
        gait_frequency_hz=_metadata_float(args.gait_frequency, metadata, "gait_frequency_hz", None),
        gait_step_length=_metadata_float(args.gait_step_length, metadata, "gait_step_length", None),
        gait_swing_height=_metadata_float(args.gait_swing_height, metadata, "gait_swing_height", None),
        gait_joint_thigh_amplitude=_metadata_float(
            args.gait_thigh_amplitude,
            metadata,
            "gait_joint_thigh_amplitude",
            None,
        ),
        gait_joint_calf_amplitude=_metadata_float(
            args.gait_calf_amplitude,
            metadata,
            "gait_joint_calf_amplitude",
            None,
        ),
    )
    pd_kp = _metadata_float(args.pd_kp, metadata, "pd_kp", None)
    pd_kd = _metadata_float(args.pd_kd, metadata, "pd_kd", None)
    torque_limit = _metadata_float(args.torque_limit, metadata, "torque_limit", None)
    if pd_kp is not None or pd_kd is not None or torque_limit is not None:
        env_config = replace(
            env_config,
            pd=PDConfig(
                kp=env_config.pd.kp if pd_kp is None else pd_kp,
                kd=env_config.pd.kd if pd_kd is None else pd_kd,
                torque_limit=env_config.pd.torque_limit if torque_limit is None else torque_limit,
            ),
        )

    env = make_vec_env(model_path=args.model, config=env_config, num_envs=1)
    if args.normalize_observation or args.normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=args.normalize_observation,
            norm_reward=args.normalize_reward,
            clip_obs=args.clip_observation,
        )
        if args.normalize_observation:
            env.obs_rms.update(observations)
        env.training = True

    if tuple(env.observation_space.shape or ()) != (observations.shape[1],):
        raise ValueError(
            "dataset observation shape does not match env: "
            f"dataset {(observations.shape[1],)} vs env {env.observation_space.shape}"
        )
    if tuple(env.action_space.shape or ()) != (actions.shape[1],):
        raise ValueError(
            "dataset action shape does not match env: "
            f"dataset {(actions.shape[1],)} vs env {env.action_space.shape}"
        )

    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy_observations = (
        env.normalize_obs(observations.copy())
        if args.normalize_observation and hasattr(env, "normalize_obs")
        else observations
    )
    train_obs, train_actions, val_obs, val_actions = _split_dataset(
        policy_observations,
        actions,
        validation_split=args.validation_split,
        rng=rng,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        learning_rate=5e-5,
        clip_range=0.08,
        target_kl=0.015,
        policy_kwargs={
            "log_std_init": -2.3,
            "net_arch": {"pi": [512, 256, 128], "vf": [512, 256, 128]},
            "activation_fn": th.nn.ELU,
        },
        device=args.device,
        seed=args.seed,
    )

    device = model.policy.device
    train_dataset = TensorDataset(
        th.as_tensor(train_obs, dtype=th.float32),
        th.as_tensor(train_actions, dtype=th.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
    )
    optimizer = th.optim.AdamW(
        model.policy.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    model.policy.set_training_mode(True)

    history: list[dict[str, float | int]] = []
    for epoch in range(max(1, args.epochs)):
        total_loss = 0.0
        total_samples = 0
        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            predicted = _policy_mean_action(model.policy, obs_batch)
            loss = th.nn.functional.mse_loss(predicted, action_batch)
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            batch_size = int(obs_batch.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size

        train_mse = total_loss / max(total_samples, 1)
        val_mse = _mse_on_arrays(model.policy, val_obs, val_actions, device=device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_action_mse": train_mse,
                "validation_action_mse": val_mse,
            }
        )
        print(
            f"epoch {epoch + 1}/{max(1, args.epochs)} "
            f"train_action_mse={train_mse:.6f} "
            f"validation_action_mse={val_mse:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    vecnormalize_output = args.vecnormalize_output or default_vecnormalize_path(args.output)
    if isinstance(env, VecNormalize):
        vecnormalize_output.parent.mkdir(parents=True, exist_ok=True)
        env.save(str(vecnormalize_output))

    report_path = args.report or args.output.with_suffix(".bc.json")
    report = {
        "dataset": str(args.dataset),
        "output": str(args.output),
        "vecnormalize": str(vecnormalize_output) if isinstance(env, VecNormalize) else None,
        "task": task,
        "samples": int(observations.shape[0]),
        "train_samples": int(train_obs.shape[0]),
        "validation_samples": int(val_obs.shape[0]),
        "observation_size": int(observations.shape[1]),
        "action_size": int(actions.shape[1]),
        "target_forward_velocity": env_config.target_forward_velocity,
        "action_scale": env_config.action_scale,
        "action_smoothing": env_config.action_smoothing,
        "reset_settle_s": env_config.reset_settle_s,
        "velocity_pose_profile": env_config.velocity_pose_profile,
        "policy_leg_order": policy_leg_order_profile,
        "observation_mode": env_config.observation_mode,
        "fullbody_reference_mode": env_config.fullbody_reference_mode,
        "gait_frequency_hz": env_config.gait_frequency_hz,
        "gait_step_length": env_config.gait_step_length,
        "gait_swing_height": env_config.gait_swing_height,
        "gait_joint_thigh_amplitude": env_config.gait_joint_thigh_amplitude,
        "gait_joint_calf_amplitude": env_config.gait_joint_calf_amplitude,
        "pd_kp": env_config.pd.kp,
        "pd_kd": env_config.pd.kd,
        "torque_limit": env_config.pd.torque_limit,
        "normalize_observation": args.normalize_observation,
        "normalize_reward": args.normalize_reward,
        "epochs": max(1, args.epochs),
        "learning_rate": args.learning_rate,
        "final_train_action_mse": history[-1]["train_action_mse"],
        "final_validation_action_mse": history[-1]["validation_action_mse"],
        "history": history,
        "reference_metadata": metadata,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"task: {task}")
    print(f"samples: {observations.shape[0]}")
    print(f"observation_size: {observations.shape[1]}")
    print(f"action_size: {actions.shape[1]}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"velocity_pose_profile: {env_config.velocity_pose_profile}")
    print(f"policy_leg_order: {policy_leg_order_profile}")
    print(f"observation_mode: {env_config.observation_mode}")
    print(f"fullbody_reference_mode: {env_config.fullbody_reference_mode}")
    print(f"action_scale: {env_config.action_scale}")
    print(f"action_smoothing: {env_config.action_smoothing}")
    print(f"reset_settle_s: {env_config.reset_settle_s}")
    print(f"pd_kp: {env_config.pd.kp}")
    print(f"pd_kd: {env_config.pd.kd}")
    print(f"torque_limit: {env_config.pd.torque_limit}")
    print(f"final_train_action_mse: {history[-1]['train_action_mse']:.6f}")
    print(f"final_validation_action_mse: {history[-1]['validation_action_mse']:.6f}")
    print(f"wrote {args.output}")
    if isinstance(env, VecNormalize):
        print(f"wrote {vecnormalize_output}")
    print(f"wrote {report_path}")
    return 0


def _load_metadata(data: Any) -> dict[str, Any]:
    if "metadata" not in data.files:
        return {}
    raw = data["metadata"]
    value = raw.item() if getattr(raw, "shape", ()) == () else raw.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _metadata_float(
    explicit: float | None,
    metadata: dict[str, Any],
    key: str,
    default: float | None,
) -> float | None:
    if explicit is not None:
        return float(explicit)
    value = metadata.get(key, default)
    return None if value is None else float(value)


def _split_dataset(
    observations: Any,
    actions: Any,
    *,
    validation_split: float,
    rng: Any,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    sample_count = int(observations.shape[0])
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    validation_count = int(round(sample_count * max(0.0, min(0.5, validation_split))))
    validation_count = min(max(validation_count, 1 if sample_count > 1 else 0), sample_count - 1)
    val_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return (
        observations[train_indices],
        actions[train_indices],
        observations[val_indices],
        actions[val_indices],
    )


def _policy_mean_action(policy: Any, obs_batch: Any) -> Any:
    distribution = policy.get_distribution(obs_batch)
    torch_distribution = getattr(distribution, "distribution", None)
    if torch_distribution is not None and hasattr(torch_distribution, "mean"):
        return torch_distribution.mean
    if hasattr(distribution, "mode"):
        return distribution.mode()
    raise RuntimeError("policy distribution does not expose mean or mode")


def _mse_on_arrays(policy: Any, observations: Any, actions: Any, *, device: Any) -> float:
    import torch as th

    if int(observations.shape[0]) == 0:
        return 0.0
    policy.set_training_mode(False)
    with th.no_grad():
        obs_tensor = th.as_tensor(observations, dtype=th.float32, device=device)
        action_tensor = th.as_tensor(actions, dtype=th.float32, device=device)
        predicted = _policy_mean_action(policy, obs_tensor)
        loss = th.nn.functional.mse_loss(predicted, action_tensor)
    policy.set_training_mode(True)
    return float(loss.detach().cpu())


if __name__ == "__main__":
    raise SystemExit(main())
