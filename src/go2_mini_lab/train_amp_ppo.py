from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .amp import (
    AmpDiscriminator,
    AmpLoggingCallback,
    AmpRewardWrapper,
    load_amp_reference_transitions,
)
from .cli import (
    add_external_force_args,
    add_policy_leg_order_arg,
    add_velocity_reward_args,
    parse_reward_overrides,
    resolve_external_force_body_names,
    resolve_policy_leg_order,
)
from .controller import PDConfig
from .gym_env import Go2StandBalanceGymEnv
from .rl_env import RL_TASKS, make_task_config
from .sb3_tools import (
    ExternalForceCurriculumCallback,
    LocomotionEvalCallback,
    RewardTermLoggingCallback,
    default_vecnormalize_path,
    initialize_policy_from_checkpoint,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a Go2 locomotion policy with PPO plus an AMP style reward."
    )
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--episode-length", type=float, default=20.0)
    parser.add_argument("--target-forward-velocity", type=float)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--action-smoothing", type=float)
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--reset-settle", type=float)
    parser.add_argument(
        "--velocity-pose-profile",
        choices=("mjlab", "official"),
        default="official",
        help="Default joint pose profile for velocity tasks.",
    )
    add_policy_leg_order_arg(parser)
    parser.add_argument("--pd-kp", type=float)
    parser.add_argument("--pd-kd", type=float)
    parser.add_argument("--torque-limit", type=float)
    add_velocity_reward_args(parser)
    parser.add_argument(
        "--amp-reference",
        type=Path,
        action="append",
        required=True,
        help=(
            "Reference motion used by AMP. Supports Go2 Mini Lab trajectory JSON "
            "and motion_imitation data/motions/*.txt. May be passed multiple times."
        ),
    )
    parser.add_argument("--amp-style-weight", type=float, default=0.02)
    parser.add_argument("--amp-task-weight", type=float, default=1.0)
    parser.add_argument("--amp-learning-rate", type=float, default=1e-4)
    parser.add_argument("--amp-batch-size", type=int, default=256)
    parser.add_argument("--amp-train-freq", type=int, default=512)
    parser.add_argument("--amp-updates", type=int, default=2)
    parser.add_argument("--amp-gradient-penalty", type=float, default=0.0)
    parser.add_argument("--amp-buffer-size", type=int, default=200_000)
    parser.add_argument("--amp-device", default="cpu")
    parser.add_argument("--normalize-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-observation", type=float, default=10.0)
    parser.add_argument("--vecnormalize-output", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--init-vecnormalize", type=Path)
    parser.add_argument("--no-init-vecnormalize", action="store_true")
    parser.add_argument(
        "--freeze-vecnormalize",
        action="store_true",
        help="Keep loaded VecNormalize statistics fixed during AMP fine-tuning.",
    )
    parser.add_argument("--randomize-commands", action="store_true")
    parser.add_argument("--fixed-command", action="store_true")
    add_external_force_args(parser, include_curriculum=True)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--clip-range", type=float, default=0.08)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--log-std-init", type=float, default=-2.3)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-duration", type=float, default=8.0)
    parser.add_argument("--best-output", type=Path)
    parser.add_argument("--eval-report", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import torch as th
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is missing. Install training dependencies with: pip install -e '.[train]'"
        ) from exc

    randomize_commands = None
    if args.randomize_commands:
        randomize_commands = True
    if args.fixed_command:
        randomize_commands = False

    external_force_curriculum_start = args.external_force_curriculum_start
    if args.external_force_curriculum_steps > 0 and external_force_curriculum_start is None:
        external_force_curriculum_start = 0.0
    velocity_reward_overrides = parse_reward_overrides(args.reward_term)
    env_config = make_task_config(
        task=args.task,
        control_dt=args.control_dt,
        episode_length_s=args.episode_length,
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
        external_force_curriculum_start_n=external_force_curriculum_start,
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
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
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

    probe_env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
    reference_transitions = load_amp_reference_transitions(args.amp_reference, env=probe_env.env)
    discriminator = AmpDiscriminator(
        reference_transitions=reference_transitions,
        input_dim=int(reference_transitions.shape[1]),
        device=args.amp_device,
        learning_rate=args.amp_learning_rate,
        buffer_size=args.amp_buffer_size,
    )
    print(f"amp_reference_transitions: {len(reference_transitions)}")
    print(f"amp_transition_dim: {reference_transitions.shape[1]}")

    def _factory():
        env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
        env = AmpRewardWrapper(
            env,
            discriminator=discriminator,
            style_weight=args.amp_style_weight,
            task_weight=args.amp_task_weight,
        )
        return Monitor(env)

    output = args.output or Path(f"checkpoints/go2_{args.task}_amp_ppo.zip")
    vecnormalize_output = args.vecnormalize_output or default_vecnormalize_path(output)
    env = DummyVecEnv([_factory for _ in range(max(1, int(args.num_envs)))])
    loaded_init_vecnormalize_path: Path | None = None
    if args.normalize_observation or args.normalize_reward:
        init_vecnormalize_path = args.init_vecnormalize
        auto_init_vecnormalize = False
        if init_vecnormalize_path is None and args.init_checkpoint and not args.no_init_vecnormalize:
            candidate = default_vecnormalize_path(args.init_checkpoint)
            if candidate.exists():
                init_vecnormalize_path = candidate
                auto_init_vecnormalize = True

        if init_vecnormalize_path is not None:
            try:
                env = VecNormalize.load(str(init_vecnormalize_path), env)
                env.training = not args.freeze_vecnormalize
                env.norm_obs = args.normalize_observation
                env.norm_reward = args.normalize_reward
                env.clip_obs = args.clip_observation
                loaded_init_vecnormalize_path = init_vecnormalize_path
                print(f"initialized VecNormalize from {init_vecnormalize_path}")
            except Exception as exc:
                if not auto_init_vecnormalize:
                    raise
                print(f"skipped init VecNormalize {init_vecnormalize_path}: {exc}")
                env = VecNormalize(
                    env,
                    norm_obs=args.normalize_observation,
                    norm_reward=args.normalize_reward,
                    clip_obs=args.clip_observation,
                )
        else:
            if args.freeze_vecnormalize:
                parser.error("--freeze-vecnormalize requires loaded VecNormalize stats")
            env = VecNormalize(
                env,
                norm_obs=args.normalize_observation,
                norm_reward=args.normalize_reward,
                clip_obs=args.clip_observation,
            )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs={
            "log_std_init": args.log_std_init,
            "net_arch": {"pi": [512, 256, 128], "vf": [512, 256, 128]},
            "activation_fn": th.nn.ELU,
        },
        device=args.device,
        seed=args.seed,
    )
    if args.init_checkpoint:
        init_report = initialize_policy_from_checkpoint(
            model=model,
            checkpoint_path=args.init_checkpoint,
            device=args.device,
        )
        print(f"initialized policy from {init_report['checkpoint']}")
        print(f"copied tensors: {len(init_report['copied'])}")
        print(f"partially copied tensors: {len(init_report['partially_copied'])}")
        print(f"skipped tensors: {len(init_report['skipped'])}")

    callbacks = [
        RewardTermLoggingCallback(log_interval=args.n_steps).callback,
        AmpLoggingCallback(
            discriminator=discriminator,
            train_freq=args.amp_train_freq,
            batch_size=args.amp_batch_size,
            updates=args.amp_updates,
            gradient_penalty=args.amp_gradient_penalty,
        ).callback,
    ]
    if args.external_force_curriculum_steps > 0:
        callbacks.append(
            ExternalForceCurriculumCallback(total_steps=args.external_force_curriculum_steps).callback
        )
    if args.eval_freq > 0:
        best_output = args.best_output or output.with_name(f"{output.stem}_best.zip")
        callbacks.append(
            LocomotionEvalCallback(
                model_path=args.model,
                config=env_config,
                eval_freq=args.eval_freq,
                best_model_path=best_output,
                report_path=args.eval_report,
                episodes=args.eval_episodes,
                duration=args.eval_duration,
                deterministic=True,
                seed=args.seed + 10_000,
            ).callback
        )

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    if isinstance(env, VecNormalize):
        vecnormalize_output.parent.mkdir(parents=True, exist_ok=True)
        env.save(str(vecnormalize_output))

    print(f"task: {args.task}")
    print(f"observation_size: {env.observation_space.shape[0]}")
    print(f"action_size: {env.action_space.shape[0]}")
    print(f"action_scale: {env_config.action_scale}")
    print(f"action_smoothing: {env_config.action_smoothing}")
    print(f"reward_scale: {env_config.reward_scale}")
    print(f"reset_settle_s: {env_config.reset_settle_s}")
    print(f"velocity_pose_profile: {env_config.velocity_pose_profile}")
    print(f"policy_leg_order: {args.policy_leg_order}")
    print(f"pd_kp: {env_config.pd.kp}")
    print(f"pd_kd: {env_config.pd.kd}")
    print(f"torque_limit: {env_config.pd.torque_limit}")
    print(f"velocity_reward_profile: {env_config.velocity_reward_profile}")
    print(f"velocity_command_frame: {env_config.velocity_command_frame}")
    if velocity_reward_overrides:
        print(f"velocity_reward_overrides: {velocity_reward_overrides}")
    print(f"amp_references: {[str(path) for path in args.amp_reference]}")
    print(f"amp_style_weight: {args.amp_style_weight}")
    print(f"amp_task_weight: {args.amp_task_weight}")
    print(f"amp_train_freq: {args.amp_train_freq}")
    print(f"amp_updates: {args.amp_updates}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"num_envs: {args.num_envs}")
    print(f"n_steps: {args.n_steps}")
    print(f"batch_size: {args.batch_size}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"randomize_commands: {env_config.randomize_commands}")
    print(f"normalize_observation: {args.normalize_observation}")
    print(f"normalize_reward: {args.normalize_reward}")
    if loaded_init_vecnormalize_path is not None:
        print(f"init_vecnormalize: {loaded_init_vecnormalize_path}")
        print(f"freeze_vecnormalize: {args.freeze_vecnormalize}")
    if env_config.external_force_probability > 0.0 and env_config.external_force_max_n > 0.0:
        print(f"external_force_mode: {env_config.external_force_mode}")
        print(f"external_force_probability: {env_config.external_force_probability}")
        print(f"external_force_body_names: {env_config.external_force_body_names}")
        print(f"external_force_active_body_count: {env_config.external_force_active_body_count}")
        print(f"external_force_event_count_range: {env_config.external_force_event_count_range}")
        print(f"external_force_rest_s_range: {env_config.external_force_rest_s_range}")
        print(f"external_force_min_n: {env_config.external_force_min_n}")
        print(f"external_force_max_n: {env_config.external_force_max_n}")
        print(f"external_force_curriculum_start_n: {env_config.external_force_curriculum_start_n}")
        print(f"external_force_curriculum_steps: {args.external_force_curriculum_steps}")
        print(f"external_force_duration_s_range: {env_config.external_force_duration_s_range}")
        print(f"external_force_z_fraction: {env_config.external_force_z_fraction}")
        print(f"external_force_torque_max_nm: {env_config.external_force_torque_max_nm}")
    if args.eval_freq > 0:
        print(f"eval_freq: {args.eval_freq}")
        print(f"best_output: {args.best_output or output.with_name(f'{output.stem}_best.zip')}")
    print(f"wrote {output}")
    if isinstance(env, VecNormalize):
        print(f"wrote {vecnormalize_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
