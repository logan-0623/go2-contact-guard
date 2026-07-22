from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .cli import add_policy_leg_order_arg, resolve_policy_leg_order
from .controller import PDConfig
from .gym_env import Go2StandBalanceGymEnv
from .onnx_rollout import DEFAULT_REFERENCE_POLICY, _load_actor_obs_normalizer
from .rl_env import RL_TASKS, make_task_config


DEFAULT_REFERENCE_GUIDED_VELOCITY = 0.08


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect obs/action pairs from a pretrained ONNX Go2 reference policy "
            "for lightweight behavior cloning."
        )
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
        default=Path("reference_datasets/go2_velocity_flat_onnx_reference.npz"),
        help="Output dataset .npz path.",
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument(
        "--target-forward-velocity",
        type=float,
        default=DEFAULT_REFERENCE_GUIDED_VELOCITY,
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
        default=0.0,
        help="EMA action smoothing used while collecting reference actions.",
    )
    parser.add_argument("--reset-settle", type=float, default=0.0)
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
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    if not args.policy.exists():
        raise FileNotFoundError(f"ONNX policy not found: {args.policy}")

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Reference dependencies are missing. Install with: pip install -e '.[reference]'"
        ) from exc

    action_scale = args.action_scale
    if action_scale is None and args.velocity_pose_profile == "mjlab":
        action_scale = 0.5

    env_config = make_task_config(
        task=args.task,
        control_dt=args.control_dt,
        episode_length_s=args.duration,
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        action_scale=action_scale,
        action_smoothing=args.action_smoothing,
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

    observations: list[object] = []
    actions: list[object] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    commands: list[object] = []
    amp_observations: list[object] = []
    qpos_values: list[object] = []
    qvel_values: list[object] = []
    ctrl_values: list[object] = []
    base_positions: list[object] = []
    base_quaternions: list[object] = []
    base_linear_velocities: list[object] = []
    base_angular_velocities: list[object] = []
    base_yaws: list[float] = []
    tracking_linear_velocities: list[object] = []
    joint_positions: list[object] = []
    joint_velocities: list[object] = []
    joint_targets: list[object] = []
    foot_positions: list[object] = []
    foot_velocities: list[object] = []
    contact_states: list[object] = []
    episode_lengths: list[int] = []

    for episode in range(max(1, args.episodes)):
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        episode_steps = 0

        while not terminated and not truncated:
            policy_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            if normalizer is not None:
                policy_obs = normalizer(policy_obs)
            action = session.run([output_name], {input_name: policy_obs})[0]
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            action = np.clip(action, -1.0, 1.0)
            rich_state = _collect_rich_state(env.env, np_module=np)

            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            commands.append(np.asarray(env.env._command(), dtype=np.float32))
            amp_observations.append(np.asarray(env.env.amp_observation(), dtype=np.float32))
            qpos_values.append(rich_state["qpos"])
            qvel_values.append(rich_state["qvel"])
            base_positions.append(rich_state["base_position"])
            base_quaternions.append(rich_state["base_quaternion"])
            base_linear_velocities.append(rich_state["base_linear_velocity"])
            base_angular_velocities.append(rich_state["base_angular_velocity"])
            base_yaws.append(float(rich_state["base_yaw"]))
            tracking_linear_velocities.append(rich_state["tracking_linear_velocity"])
            joint_positions.append(rich_state["joint_positions"])
            joint_velocities.append(rich_state["joint_velocities"])
            joint_targets.append(
                np.asarray(
                    [
                        env.env.default_targets[name] + env_config.action_scale * float(action[index])
                        for index, name in enumerate(env.env.joint_names)
                    ],
                    dtype=np.float32,
                )
            )
            foot_positions.append(rich_state["foot_positions"])
            foot_velocities.append(rich_state["foot_velocities"])
            contact_states.append(rich_state["contacts"])
            obs, reward, terminated, truncated, _info = env.step(action)
            ctrl_values.append(np.asarray(env.env.data.ctrl, dtype=np.float32).copy())
            rewards.append(float(reward))
            terminals.append(bool(terminated))
            episode_steps += 1

        episode_lengths.append(episode_steps)

    if not observations:
        raise RuntimeError("reference collection produced no samples")

    observations_array = np.stack(observations).astype(np.float32)
    actions_array = np.stack(actions).astype(np.float32)
    commands_array = np.stack(commands).astype(np.float32)
    amp_observations_array = np.stack(amp_observations).astype(np.float32)
    rewards_array = np.asarray(rewards, dtype=np.float32)
    terminals_array = np.asarray(terminals, dtype=np.bool_)
    qpos_array = np.stack(qpos_values).astype(np.float32)
    qvel_array = np.stack(qvel_values).astype(np.float32)
    ctrl_array = np.stack(ctrl_values).astype(np.float32)
    base_positions_array = np.stack(base_positions).astype(np.float32)
    base_quaternions_array = np.stack(base_quaternions).astype(np.float32)
    base_linear_velocities_array = np.stack(base_linear_velocities).astype(np.float32)
    base_angular_velocities_array = np.stack(base_angular_velocities).astype(np.float32)
    base_yaws_array = np.asarray(base_yaws, dtype=np.float32)
    tracking_linear_velocities_array = np.stack(tracking_linear_velocities).astype(np.float32)
    joint_positions_array = np.stack(joint_positions).astype(np.float32)
    joint_velocities_array = np.stack(joint_velocities).astype(np.float32)
    joint_targets_array = np.stack(joint_targets).astype(np.float32)
    foot_positions_array = np.stack(foot_positions).astype(np.float32)
    foot_velocities_array = np.stack(foot_velocities).astype(np.float32)
    contacts_array = np.stack(contact_states).astype(np.float32)

    metadata = {
        "source_policy": str(args.policy),
        "policy_format": "onnx",
        "model": str(args.model),
        "task": args.task,
        "episodes": max(1, args.episodes),
        "duration": args.duration,
        "control_dt": args.control_dt,
        "observation_size": int(observations_array.shape[1]),
        "observation_names": list(env.env.observation_names),
        "action_size": int(actions_array.shape[1]),
        "action_names": list(env.env.joint_names),
        "foot_names": ["FR", "FL", "RR", "RL"],
        "samples": int(observations_array.shape[0]),
        "episode_lengths": episode_lengths,
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
        "pd_kp": env_config.pd.kp,
        "pd_kd": env_config.pd.kd,
        "torque_limit": env_config.pd.torque_limit,
        "onnx_input": input_name,
        "onnx_output": output_name,
        "normalizer_checkpoint": str(normalizer_path) if normalizer_path else None,
        "rich_state_keys": [
            "amp_observations",
            "qpos",
            "qvel",
            "ctrl",
            "base_positions",
            "base_quaternions",
            "base_linear_velocities",
            "base_angular_velocities",
            "base_yaws",
            "tracking_linear_velocities",
            "joint_positions",
            "joint_velocities",
            "joint_targets",
            "foot_positions",
            "foot_velocities",
            "contacts",
        ],
        "seed": args.seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=observations_array,
        actions=actions_array,
        commands=commands_array,
        amp_observations=amp_observations_array,
        rewards=rewards_array,
        terminals=terminals_array,
        qpos=qpos_array,
        qvel=qvel_array,
        ctrl=ctrl_array,
        base_positions=base_positions_array,
        base_quaternions=base_quaternions_array,
        base_linear_velocities=base_linear_velocities_array,
        base_angular_velocities=base_angular_velocities_array,
        base_yaws=base_yaws_array,
        tracking_linear_velocities=tracking_linear_velocities_array,
        joint_positions=joint_positions_array,
        joint_velocities=joint_velocities_array,
        joint_targets=joint_targets_array,
        foot_positions=foot_positions_array,
        foot_velocities=foot_velocities_array,
        contacts=contacts_array,
        metadata=json.dumps(metadata),
    )

    print(f"wrote {args.output}")
    print(f"samples: {observations_array.shape[0]}")
    print(f"episodes: {len(episode_lengths)}")
    print(f"observation_size: {observations_array.shape[1]}")
    print(f"action_size: {actions_array.shape[1]}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"policy_leg_order: {args.policy_leg_order}")
    print(f"mean_episode_steps: {sum(episode_lengths) / max(len(episode_lengths), 1):.1f}")
    print(f"normalizer: {normalizer_path if normalizer_path else 'none'}")
    return 0


def _collect_rich_state(env: Any, *, np_module: Any) -> dict[str, Any]:
    frame = env.make_frame()
    qpos = np_module.asarray(env.data.qpos, dtype=np_module.float32).copy()
    qvel = np_module.asarray(env.data.qvel, dtype=np_module.float32).copy()
    base_linear_velocity = np_module.asarray(env.data.qvel[:3], dtype=np_module.float32).copy()
    return {
        "qpos": qpos,
        "qvel": qvel,
        "base_position": np_module.asarray(env.data.qpos[:3], dtype=np_module.float32).copy(),
        "base_quaternion": np_module.asarray(env.data.qpos[3:7], dtype=np_module.float32).copy(),
        "base_linear_velocity": base_linear_velocity,
        "base_angular_velocity": np_module.asarray(env.data.qvel[3:6], dtype=np_module.float32).copy(),
        "base_yaw": float(env._base_yaw()),
        "tracking_linear_velocity": np_module.asarray(
            env._velocity_tracking_linear_velocity(base_linear_velocity),
            dtype=np_module.float32,
        ),
        "joint_positions": np_module.asarray(
            [float(env.data.qpos[env.joint_map[name][0]]) for name in env.joint_names],
            dtype=np_module.float32,
        ),
        "joint_velocities": np_module.asarray(
            [float(env.data.qvel[env.joint_map[name][1]]) for name in env.joint_names],
            dtype=np_module.float32,
        ),
        "foot_positions": np_module.asarray(
            [
                env.data.xpos[env.foot_body_ids[foot]] if env.foot_body_ids[foot] >= 0 else [0.0, 0.0, 0.0]
                for foot in ("FR", "FL", "RR", "RL")
            ],
            dtype=np_module.float32,
        ),
        "foot_velocities": np_module.asarray(
            [
                env.data.cvel[env.foot_body_ids[foot]][3:6] if env.foot_body_ids[foot] >= 0 else [0.0, 0.0, 0.0]
                for foot in ("FR", "FL", "RR", "RL")
            ],
            dtype=np_module.float32,
        ),
        "contacts": np_module.asarray(
            [1.0 if (frame.get("contacts") or {}).get(foot, False) else 0.0 for foot in ("FR", "FL", "RR", "RL")],
            dtype=np_module.float32,
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
