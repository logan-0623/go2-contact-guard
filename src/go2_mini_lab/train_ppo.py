from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .controller import PDConfig
from .cli import (
    add_external_force_args,
    add_force_response_router_args,
    add_force_safety_trigger_args,
    add_fullbody_reference_args,
    add_observation_mode_arg,
    add_policy_action_mode_args,
    add_policy_leg_order_arg,
    add_velocity_reward_args,
    parse_reward_overrides,
    resolve_external_force_body_names,
    resolve_policy_leg_order,
)
from .rl_env import (
    FORCE_REFERENCE_GOVERNOR_MODES,
    FORCE_YIELDING_COMMAND_MODES,
    RL_TASKS,
    make_task_config,
)
from .sb3_tools import (
    ExternalForceCurriculumCallback,
    LocomotionEvalCallback,
    RewardTermLoggingCallback,
    default_vecnormalize_path,
    initialize_training_vecnormalize,
    initialize_policy_from_checkpoint,
    make_vec_env,
    reset_vecnormalize_observation_stats,
    unwrap_go2_env,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a PPO Go2 policy in MuJoCo with Stable-Baselines3."
    )
    parser.add_argument(
        "--task",
        choices=RL_TASKS,
        default="stand",
        help="Training task.",
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
        help="Output PPO checkpoint path. Defaults to checkpoints/go2_<task>_ppo.zip.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Training timesteps. Use a much larger value for serious training.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.02,
        help="Environment control timestep in seconds.",
    )
    parser.add_argument(
        "--episode-length",
        type=float,
        default=20.0,
        help="Episode length in seconds.",
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
        "--force-yielding-command-mode",
        choices=FORCE_YIELDING_COMMAND_MODES,
        default="scaled",
        help=(
            "Command-yielding mode for external-force windows. "
            "scaled uses gain times force; unit uses the clipped unit force direction; "
            "unit_pulse uses a delayed clipped unit-direction pulse."
        ),
    )
    parser.add_argument(
        "--force-yielding-command-velocity-per-n",
        type=float,
        default=0.0,
        help="Signed command velocity offset gain in (m/s)/N along the external-force direction.",
    )
    parser.add_argument(
        "--force-yielding-command-velocity-clip",
        type=float,
        default=0.0,
        help="Clip for the horizontal force-yielding command offset in m/s. Zero disables it.",
    )
    parser.add_argument(
        "--force-yielding-command-pulse-start-s",
        type=float,
        default=0.10,
        help="Delay after force onset before unit_pulse command yielding starts.",
    )
    parser.add_argument(
        "--force-yielding-command-pulse-duration-s",
        type=float,
        default=0.20,
        help="Full-amplitude unit_pulse command yielding duration in seconds.",
    )
    parser.add_argument(
        "--force-yielding-command-pulse-recovery-s",
        type=float,
        default=0.10,
        help="Linear recovery duration after the full-amplitude unit_pulse window.",
    )
    parser.add_argument(
        "--force-yielding-command-pulse-post-clip",
        type=float,
        default=0.0,
        help="Residual command-yielding clip after unit_pulse recovery in m/s.",
    )
    parser.add_argument(
        "--force-reference-governor-mode",
        choices=FORCE_REFERENCE_GOVERNOR_MODES,
        default="off",
        help="External-force reference governor mode. Use onset or active for v13b-style probes.",
    )
    parser.add_argument(
        "--force-reference-governor-admittance",
        type=float,
        default=0.0,
        help="Reference-governor velocity gain in (m/s)/N along the external-force direction.",
    )
    parser.add_argument(
        "--force-reference-governor-damping",
        type=float,
        default=5.0,
        help="Reference-governor first-order damping.",
    )
    parser.add_argument(
        "--force-reference-governor-offset-clip",
        type=float,
        default=0.0,
        help="Reference-governor horizontal offset clip in meters. Zero disables offset motion.",
    )
    parser.add_argument(
        "--force-reference-governor-velocity-clip",
        type=float,
        default=0.0,
        help="Reference-governor velocity clip in m/s. Zero disables reference velocity motion.",
    )
    parser.add_argument(
        "--force-reference-governor-delay-s",
        type=float,
        default=0.10,
        help="Delay after force onset before the onset reference governor activates.",
    )
    parser.add_argument(
        "--force-reference-governor-hold-s",
        type=float,
        default=0.20,
        help="Hold duration for the onset reference governor.",
    )
    parser.add_argument(
        "--force-reference-governor-recovery-s",
        type=float,
        default=0.10,
        help="Recovery duration for reference-governor offset decay.",
    )
    parser.add_argument("--force-reference-governor-tail-admittance-scale", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-tail-offset-clip-scale", type=float, default=1.0)
    parser.add_argument("--force-reference-governor-tail-velocity-clip-scale", type=float, default=1.0)
    parser.add_argument(
        "--force-impedance-mode",
        choices=("off", "onset", "active", "two_phase"),
        default="off",
        help="Event-triggered PD impedance modulation mode.",
    )
    parser.add_argument(
        "--force-impedance-joint-scope",
        choices=("all", "hip", "thigh", "calf", "hip_calf", "stance_hip_calf"),
        default="all",
        help="Joint family affected by force-triggered impedance modulation.",
    )
    parser.add_argument(
        "--force-impedance-kp-scale",
        type=float,
        default=1.0,
        help="Kp multiplier during the force impedance window.",
    )
    parser.add_argument(
        "--force-impedance-kd-scale",
        type=float,
        default=1.0,
        help="Kd multiplier during the force impedance window.",
    )
    parser.add_argument(
        "--force-impedance-delay-s",
        type=float,
        default=0.0,
        help="Delay after safety onset before impedance modulation starts.",
    )
    parser.add_argument(
        "--force-impedance-hold-s",
        type=float,
        default=0.15,
        help="Onset-mode impedance modulation hold duration.",
    )
    parser.add_argument(
        "--force-impedance-recovery-s",
        type=float,
        default=0.10,
        help="Ramp duration back to nominal Kp/Kd.",
    )
    parser.add_argument("--force-impedance-tail-kp-scale", type=float, default=1.0)
    parser.add_argument("--force-impedance-tail-kd-scale", type=float, default=1.0)
    add_force_response_router_args(parser)
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
        help="Multiplier applied to environment rewards before PPO sees them. Default is task-specific.",
    )
    parser.add_argument(
        "--reset-settle",
        type=float,
        help="Seconds to settle with zero action after reset. Default is task-specific.",
    )
    parser.add_argument(
        "--velocity-pose-profile",
        choices=("mjlab", "official"),
        default="official",
        help="Default joint pose profile for velocity tasks.",
    )
    add_policy_leg_order_arg(parser)
    add_observation_mode_arg(parser)
    add_policy_action_mode_args(parser)
    add_fullbody_reference_args(parser)
    add_force_safety_trigger_args(parser)
    parser.add_argument("--pd-kp", type=float, help="Joint PD stiffness. Default is task-specific.")
    parser.add_argument("--pd-kd", type=float, help="Joint PD damping. Default is task-specific.")
    parser.add_argument("--torque-limit", type=float, help="Joint torque limit. Default is task-specific.")
    add_velocity_reward_args(parser)
    parser.add_argument(
        "--normalize-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize observations with VecNormalize.",
    )
    parser.add_argument(
        "--normalize-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize rewards with VecNormalize.",
    )
    parser.add_argument(
        "--clip-observation",
        type=float,
        default=10.0,
        help="VecNormalize observation clipping value.",
    )
    parser.add_argument(
        "--vecnormalize-output",
        type=Path,
        help="Output VecNormalize stats path. Defaults to <checkpoint>_vecnormalize.pkl.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Initialize compatible policy weights from an existing PPO checkpoint. "
            "Optimizer state is not copied."
        ),
    )
    parser.add_argument(
        "--init-bc",
        type=Path,
        help=(
            "Initialize compatible policy weights from a behavior-cloned PPO checkpoint. "
            "This is an alias for --init-checkpoint with clearer intent."
        ),
    )
    parser.add_argument(
        "--init-vecnormalize",
        type=Path,
        help=(
            "Initialize VecNormalize stats from an existing stats file. "
            "Defaults to <init-checkpoint>_vecnormalize.pkl or "
            "<init-bc>_vecnormalize.pkl when compatible."
        ),
    )
    parser.add_argument(
        "--no-init-vecnormalize",
        action="store_true",
        help="Do not auto-load VecNormalize stats from --init-checkpoint or --init-bc.",
    )
    parser.add_argument(
        "--init-vecnormalize-prefix-count",
        type=float,
        default=1024.0,
        help=(
            "Bootstrap sample count used when only a prefix of VecNormalize obs stats can be copied. "
            "Keep this modest so newly added privileged obs dimensions can adapt."
        ),
    )
    parser.add_argument(
        "--init-vecnormalize-reset-obs-pattern",
        action="append",
        default=[],
        help=(
            "Substring pattern for observation dimensions whose loaded VecNormalize "
            "mean/variance should be reset. Repeat for multiple patterns."
        ),
    )
    parser.add_argument(
        "--init-vecnormalize-reset-obs-std",
        type=float,
        default=1.0,
        help="Standard deviation used for reset VecNormalize observation dimensions.",
    )
    parser.add_argument(
        "--freeze-vecnormalize",
        action="store_true",
        help=(
            "Keep loaded VecNormalize statistics fixed during fine-tuning. "
            "Use this when preserving a pretrained policy's observation scale."
        ),
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
    parser.add_argument("--standing-command-prob", type=float)
    parser.add_argument("--command-vx-min", type=float)
    parser.add_argument("--command-vx-max", type=float)
    parser.add_argument("--command-vy-min", type=float)
    parser.add_argument("--command-vy-max", type=float)
    parser.add_argument("--command-yaw-rate-min", type=float)
    parser.add_argument("--command-yaw-rate-max", type=float)
    parser.add_argument("--randomize-gait-parameters", action="store_true")
    parser.add_argument("--gait-frequency-min", type=float)
    parser.add_argument("--gait-frequency-max", type=float)
    parser.add_argument("--gait-step-length-min", type=float)
    parser.add_argument("--gait-step-length-max", type=float)
    parser.add_argument("--gait-swing-height-min", type=float)
    parser.add_argument("--gait-swing-height-max", type=float)
    parser.add_argument(
        "--push-time",
        type=float,
        help="Push time in seconds for push_recovery. Default is task-specific.",
    )
    parser.add_argument("--push-vx", type=float, help="Root velocity impulse x component.")
    parser.add_argument("--push-vy", type=float, help="Root velocity impulse y component.")
    parser.add_argument("--push-vz", type=float, help="Root velocity impulse z component.")
    parser.add_argument("--push-wx", type=float, help="Root angular velocity impulse x component.")
    parser.add_argument("--push-wy", type=float, help="Root angular velocity impulse y component.")
    parser.add_argument("--push-wz", type=float, help="Root angular velocity impulse z component.")
    add_external_force_args(parser, include_curriculum=True)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="PPO rollout steps per update.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel MuJoCo environments for PPO collection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="PPO optimization epochs per rollout batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.08,
        help="PPO clipping range.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.015,
        help="Early-stop PPO epochs when KL exceeds this value.",
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=-2.3,
        help="Initial log standard deviation for Gaussian actions.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value-function loss coefficient.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Run deterministic metric evaluation every N timesteps. 0 disables training-time eval.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Episodes per training-time evaluation.",
    )
    parser.add_argument(
        "--eval-duration",
        type=float,
        default=8.0,
        help="Duration in seconds per training-time evaluation episode.",
    )
    parser.add_argument(
        "--eval-external-force-probability",
        type=float,
        help=(
            "Override external force probability for training-time evaluation. "
            "Use 1.0 for disturbance-stage best checkpoint selection."
        ),
    )
    parser.add_argument(
        "--eval-suite",
        choices=("single", "gentle_30n"),
        default="single",
        help="Training-time best-checkpoint evaluation suite.",
    )
    parser.add_argument(
        "--best-output",
        type=Path,
        help="Best checkpoint path used by --eval-freq. Defaults to <output>_best.zip.",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        help="Best evaluation JSON report path used by --eval-freq.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to Stable-Baselines3.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Training seed.")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")
    if args.init_checkpoint is not None and args.init_bc is not None:
        parser.error("choose only one of --init-checkpoint or --init-bc")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.vec_env import VecNormalize
        import torch as th
    except ImportError as exc:
        raise RuntimeError(
            "Stable-Baselines3 is missing. Install training dependencies with: "
            "pip install -e '.[train]'"
        ) from exc

    randomize_commands = None
    if args.randomize_commands:
        randomize_commands = True
    if args.fixed_command:
        randomize_commands = False

    push_linear_velocity = _optional_vec(args.push_vx, args.push_vy, args.push_vz)
    push_angular_velocity = _optional_vec(args.push_wx, args.push_wy, args.push_wz)
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
        force_yielding_command_mode=args.force_yielding_command_mode,
        force_yielding_command_velocity_per_n=args.force_yielding_command_velocity_per_n,
        force_yielding_command_velocity_clip=args.force_yielding_command_velocity_clip,
        force_yielding_command_pulse_start_s=args.force_yielding_command_pulse_start_s,
        force_yielding_command_pulse_duration_s=args.force_yielding_command_pulse_duration_s,
        force_yielding_command_pulse_recovery_s=args.force_yielding_command_pulse_recovery_s,
        force_yielding_command_pulse_post_clip=args.force_yielding_command_pulse_post_clip,
        force_impedance_mode=args.force_impedance_mode,
        force_impedance_joint_scope=args.force_impedance_joint_scope,
        force_impedance_kp_scale=args.force_impedance_kp_scale,
        force_impedance_kd_scale=args.force_impedance_kd_scale,
        force_impedance_delay_s=args.force_impedance_delay_s,
        force_impedance_hold_s=args.force_impedance_hold_s,
        force_impedance_recovery_s=args.force_impedance_recovery_s,
        force_impedance_tail_kp_scale=args.force_impedance_tail_kp_scale,
        force_impedance_tail_kd_scale=args.force_impedance_tail_kd_scale,
        force_reference_governor_mode=args.force_reference_governor_mode,
        force_reference_governor_admittance_mps_per_n=args.force_reference_governor_admittance,
        force_reference_governor_damping=args.force_reference_governor_damping,
        force_reference_governor_offset_clip_m=args.force_reference_governor_offset_clip,
        force_reference_governor_velocity_clip_mps=args.force_reference_governor_velocity_clip,
        force_reference_governor_delay_s=args.force_reference_governor_delay_s,
        force_reference_governor_hold_s=args.force_reference_governor_hold_s,
        force_reference_governor_recovery_s=args.force_reference_governor_recovery_s,
        force_reference_governor_tail_admittance_scale=args.force_reference_governor_tail_admittance_scale,
        force_reference_governor_tail_offset_clip_scale=args.force_reference_governor_tail_offset_clip_scale,
        force_reference_governor_tail_velocity_clip_scale=args.force_reference_governor_tail_velocity_clip_scale,
        force_response_router_mode=args.force_response_router_mode,
        force_response_profile=args.force_response_profile,
        force_response_foot_kp_scale=args.force_response_foot_kp_scale,
        force_response_foot_kd_scale=args.force_response_foot_kd_scale,
        force_safety_trigger_source=args.force_safety_trigger_source,
        force_safety_history_estimator_path=args.force_safety_history_estimator,
        force_safety_detector_linear_acceleration_threshold=(
            args.force_safety_detector_linear_acceleration_threshold
        ),
        force_safety_detector_angular_acceleration_threshold=(
            args.force_safety_detector_angular_acceleration_threshold
        ),
        force_safety_detector_joint_error_threshold=args.force_safety_detector_joint_error_threshold,
        force_safety_detector_joint_velocity_threshold=args.force_safety_detector_joint_velocity_threshold,
        force_safety_detector_contact_loss=args.force_safety_detector_contact_loss,
        force_safety_detector_enable_after_s=args.force_safety_detector_enable_after_s,
        force_safety_detector_hold_s=args.force_safety_detector_hold_s,
        force_safety_detector_recovery_s=args.force_safety_detector_recovery_s,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        reward_scale=args.reward_scale,
        reset_settle_s=args.reset_settle,
        randomize_commands=randomize_commands,
        standing_command_prob=args.standing_command_prob,
        command_lin_vel_x_range=(
            (args.command_vx_min, args.command_vx_max)
            if args.command_vx_min is not None and args.command_vx_max is not None
            else None
        ),
        command_lin_vel_y_range=(
            (args.command_vy_min, args.command_vy_max)
            if args.command_vy_min is not None and args.command_vy_max is not None
            else None
        ),
        command_yaw_rate_range=(
            (args.command_yaw_rate_min, args.command_yaw_rate_max)
            if args.command_yaw_rate_min is not None and args.command_yaw_rate_max is not None
            else None
        ),
        push_time_s=args.push_time,
        push_linear_velocity=push_linear_velocity,
        push_angular_velocity=push_angular_velocity,
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
        external_force_direction_mode=args.external_force_direction_mode,
        external_force_lateral_probability=args.external_force_lateral_probability,
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
        external_force_safe_limit_min_n=args.external_force_safe_limit_min,
        external_force_safe_limit_max_n=args.external_force_safe_limit_max,
        external_force_safe_margin_n=args.external_force_safe_margin,
        include_external_force_observation=args.external_force_observation,
        observation_mode=args.observation_mode,
        policy_action_mode=args.policy_action_mode,
        onnx_policy_path=args.onnx_policy,
        onnx_normalizer_checkpoint=args.onnx_normalizer_checkpoint,
        residual_action_scale=args.residual_action_scale,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
        fullbody_reference_mode=args.fullbody_reference_mode,
        nominal_reference_dataset=args.nominal_reference_dataset,
        gait_frequency_hz=args.gait_frequency,
        gait_step_length=args.gait_step_length,
        gait_swing_height=args.gait_swing_height,
        randomize_gait_parameters=args.randomize_gait_parameters,
        gait_frequency_range=(
            (args.gait_frequency_min, args.gait_frequency_max)
            if args.gait_frequency_min is not None and args.gait_frequency_max is not None
            else None
        ),
        gait_step_length_range=(
            (args.gait_step_length_min, args.gait_step_length_max)
            if args.gait_step_length_min is not None and args.gait_step_length_max is not None
            else None
        ),
        gait_swing_height_range=(
            (args.gait_swing_height_min, args.gait_swing_height_max)
            if args.gait_swing_height_min is not None and args.gait_swing_height_max is not None
            else None
        ),
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
    init_policy_checkpoint = args.init_checkpoint or args.init_bc
    output = args.output or Path(f"checkpoints/go2_{args.task}_ppo.zip")
    vecnormalize_output = args.vecnormalize_output or default_vecnormalize_path(output)
    env = make_vec_env(model_path=args.model, config=env_config, num_envs=args.num_envs)
    observation_names = unwrap_go2_env(env).observation_names
    loaded_init_vecnormalize_path: Path | None = None
    if args.normalize_observation or args.normalize_reward:
        init_vecnormalize_path = args.init_vecnormalize
        auto_init_vecnormalize = False
        if (
            init_vecnormalize_path is None
            and init_policy_checkpoint
            and not args.no_init_vecnormalize
        ):
            candidate = default_vecnormalize_path(init_policy_checkpoint)
            if candidate.exists():
                init_vecnormalize_path = candidate
                auto_init_vecnormalize = True

        if init_vecnormalize_path is not None:
            try:
                env, loaded_init_vecnormalize_path, init_vecnormalize_mode = initialize_training_vecnormalize(
                    vec_env=env,
                    normalize_observation=args.normalize_observation,
                    normalize_reward=args.normalize_reward,
                    clip_observation=args.clip_observation,
                    init_path=init_vecnormalize_path,
                    freeze=args.freeze_vecnormalize,
                    prefix_count=args.init_vecnormalize_prefix_count,
                    observation_names=observation_names,
                    reset_obs_patterns=args.init_vecnormalize_reset_obs_pattern,
                    reset_obs_std=args.init_vecnormalize_reset_obs_std,
                )
                loaded_init_vecnormalize_path = init_vecnormalize_path
                print(f"initialized VecNormalize from {init_vecnormalize_path} ({init_vecnormalize_mode})")
            except Exception as exc:
                if not auto_init_vecnormalize:
                    raise
                print(
                    "skipped init VecNormalize "
                    f"{init_vecnormalize_path}: {exc}"
                )
                env = VecNormalize(
                    env,
                    norm_obs=args.normalize_observation,
                    norm_reward=args.normalize_reward,
                    clip_obs=args.clip_observation,
                )
        else:
            try:
                env, _, _ = initialize_training_vecnormalize(
                    vec_env=env,
                    normalize_observation=args.normalize_observation,
                    normalize_reward=args.normalize_reward,
                    clip_observation=args.clip_observation,
                    init_path=None,
                    freeze=args.freeze_vecnormalize,
                    prefix_count=args.init_vecnormalize_prefix_count,
                )
            except ValueError as exc:
                parser.error(str(exc))

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
    if init_policy_checkpoint:
        init_report = initialize_policy_from_checkpoint(
            model=model,
            checkpoint_path=init_policy_checkpoint,
            device=args.device,
        )
        label = "BC policy" if args.init_bc is not None else "policy"
        print(f"initialized {label} from {init_report['checkpoint']}")
        print(f"copied tensors: {len(init_report['copied'])}")
        print(f"partially copied tensors: {len(init_report['partially_copied'])}")
        print(f"skipped tensors: {len(init_report['skipped'])}")

    callbacks = [RewardTermLoggingCallback(log_interval=args.n_steps).callback]
    if args.external_force_curriculum_steps > 0:
        callbacks.append(
            ExternalForceCurriculumCallback(
                total_steps=args.external_force_curriculum_steps,
            ).callback
        )
    if args.eval_freq > 0:
        best_output = args.best_output or output.with_name(f"{output.stem}_best.zip")
        eval_config = env_config
        if args.eval_external_force_probability is not None:
            eval_config = replace(
                env_config,
                external_force_probability=args.eval_external_force_probability,
            )
        callbacks.append(
            LocomotionEvalCallback(
                model_path=args.model,
                config=eval_config,
                eval_freq=args.eval_freq,
                best_model_path=best_output,
                report_path=args.eval_report,
                episodes=args.eval_episodes,
                duration=args.eval_duration,
                deterministic=True,
                seed=args.seed + 10_000,
                eval_suite=args.eval_suite,
            ).callback
        )
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]
    model.learn(total_timesteps=args.timesteps, callback=callback)
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
    print(f"learning_rate: {args.learning_rate}")
    print(f"num_envs: {args.num_envs}")
    print(f"n_steps: {args.n_steps}")
    print(f"batch_size: {args.batch_size}")
    print(f"n_epochs: {args.n_epochs}")
    print(f"clip_range: {args.clip_range}")
    print(f"target_kl: {args.target_kl}")
    print(f"log_std_init: {args.log_std_init}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"force_yielding_command_mode: {env_config.force_yielding_command_mode}")
    print(f"force_yielding_command_velocity_per_n: {env_config.force_yielding_command_velocity_per_n}")
    print(f"force_yielding_command_velocity_clip: {env_config.force_yielding_command_velocity_clip}")
    print(f"force_impedance_mode: {env_config.force_impedance_mode}")
    print(f"force_impedance_joint_scope: {env_config.force_impedance_joint_scope}")
    print(f"force_impedance_kp_scale: {env_config.force_impedance_kp_scale}")
    print(f"force_impedance_kd_scale: {env_config.force_impedance_kd_scale}")
    print(f"force_impedance_delay_s: {env_config.force_impedance_delay_s}")
    print(f"force_impedance_hold_s: {env_config.force_impedance_hold_s}")
    print(f"force_impedance_recovery_s: {env_config.force_impedance_recovery_s}")
    print(f"force_impedance_tail_kp_scale: {env_config.force_impedance_tail_kp_scale}")
    print(f"force_impedance_tail_kd_scale: {env_config.force_impedance_tail_kd_scale}")
    print(f"force_reference_governor_mode: {env_config.force_reference_governor_mode}")
    print(f"force_reference_governor_admittance_mps_per_n: {env_config.force_reference_governor_admittance_mps_per_n}")
    print(f"force_reference_governor_damping: {env_config.force_reference_governor_damping}")
    print(f"force_reference_governor_offset_clip_m: {env_config.force_reference_governor_offset_clip_m}")
    print(f"force_reference_governor_velocity_clip_mps: {env_config.force_reference_governor_velocity_clip_mps}")
    print(f"force_reference_governor_delay_s: {env_config.force_reference_governor_delay_s}")
    print(f"force_reference_governor_hold_s: {env_config.force_reference_governor_hold_s}")
    print(f"force_reference_governor_recovery_s: {env_config.force_reference_governor_recovery_s}")
    print(
        "force_reference_governor_tail_admittance_scale: "
        f"{env_config.force_reference_governor_tail_admittance_scale}"
    )
    print(
        "force_reference_governor_tail_offset_clip_scale: "
        f"{env_config.force_reference_governor_tail_offset_clip_scale}"
    )
    print(
        "force_reference_governor_tail_velocity_clip_scale: "
        f"{env_config.force_reference_governor_tail_velocity_clip_scale}"
    )
    print(f"force_response_router_mode: {env_config.force_response_router_mode}")
    print(f"force_response_profile: {env_config.force_response_profile}")
    print(f"force_response_foot_kp_scale: {env_config.force_response_foot_kp_scale}")
    print(f"force_response_foot_kd_scale: {env_config.force_response_foot_kd_scale}")
    print(f"force_safety_trigger_source: {env_config.force_safety_trigger_source}")
    print(f"force_safety_history_estimator_path: {env_config.force_safety_history_estimator_path}")
    print(
        "force_safety_detector_linear_acceleration_threshold: "
        f"{env_config.force_safety_detector_linear_acceleration_threshold}"
    )
    print(
        "force_safety_detector_angular_acceleration_threshold: "
        f"{env_config.force_safety_detector_angular_acceleration_threshold}"
    )
    print(f"force_safety_detector_joint_error_threshold: {env_config.force_safety_detector_joint_error_threshold}")
    print(
        "force_safety_detector_joint_velocity_threshold: "
        f"{env_config.force_safety_detector_joint_velocity_threshold}"
    )
    print(f"force_safety_detector_contact_loss: {env_config.force_safety_detector_contact_loss}")
    print(f"force_safety_detector_enable_after_s: {env_config.force_safety_detector_enable_after_s}")
    print(f"force_safety_detector_hold_s: {env_config.force_safety_detector_hold_s}")
    print(f"force_safety_detector_recovery_s: {env_config.force_safety_detector_recovery_s}")
    print(f"randomize_commands: {env_config.randomize_commands}")
    print(f"reset_settle_s: {env_config.reset_settle_s}")
    print(f"velocity_pose_profile: {env_config.velocity_pose_profile}")
    print(f"policy_leg_order: {args.policy_leg_order}")
    print(f"observation_mode: {env_config.observation_mode}")
    print(f"policy_action_mode: {env_config.policy_action_mode}")
    if env_config.onnx_policy_path:
        print(f"onnx_policy_path: {env_config.onnx_policy_path}")
    if env_config.onnx_normalizer_checkpoint:
        print(f"onnx_normalizer_checkpoint: {env_config.onnx_normalizer_checkpoint}")
    print(f"residual_action_scale: {env_config.residual_action_scale}")
    print(f"velocity_reward_profile: {env_config.velocity_reward_profile}")
    print(f"velocity_command_frame: {env_config.velocity_command_frame}")
    print(f"fullbody_reference_mode: {env_config.fullbody_reference_mode}")
    if env_config.nominal_reference_dataset:
        print(f"nominal_reference_dataset: {env_config.nominal_reference_dataset}")
    print(f"pd_kp: {env_config.pd.kp}")
    print(f"pd_kd: {env_config.pd.kd}")
    print(f"torque_limit: {env_config.pd.torque_limit}")
    if velocity_reward_overrides:
        print(f"velocity_reward_overrides: {velocity_reward_overrides}")
    print(f"normalize_observation: {args.normalize_observation}")
    print(f"normalize_reward: {args.normalize_reward}")
    if args.eval_freq > 0:
        print(f"eval_freq: {args.eval_freq}")
        print(f"eval_episodes: {args.eval_episodes}")
        print(f"eval_duration: {args.eval_duration}")
        print(f"eval_suite: {args.eval_suite}")
        if args.eval_external_force_probability is not None:
            print(f"eval_external_force_probability: {args.eval_external_force_probability}")
        print(f"best_output: {args.best_output or output.with_name(f'{output.stem}_best.zip')}")
    if loaded_init_vecnormalize_path is not None:
        print(f"init_vecnormalize: {loaded_init_vecnormalize_path}")
        print(f"init_vecnormalize_mode: {init_vecnormalize_mode}")
        print(f"init_vecnormalize_prefix_count: {args.init_vecnormalize_prefix_count}")
        if args.init_vecnormalize_reset_obs_pattern:
            print(f"init_vecnormalize_reset_obs_patterns: {args.init_vecnormalize_reset_obs_pattern}")
            print(f"init_vecnormalize_reset_obs_std: {args.init_vecnormalize_reset_obs_std}")
        print(f"freeze_vecnormalize: {args.freeze_vecnormalize}")
    if env_config.push_time_s is not None:
        print(f"push_time_s: {env_config.push_time_s}")
        print(f"push_linear_velocity: {env_config.push_linear_velocity}")
        print(f"push_angular_velocity: {env_config.push_angular_velocity}")
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
        print(f"external_force_start_s_range: {env_config.external_force_start_s_range}")
        print(f"external_force_duration_s_range: {env_config.external_force_duration_s_range}")
        print(f"external_force_z_fraction: {env_config.external_force_z_fraction}")
        print(f"external_force_direction_angle_rad: {env_config.external_force_direction_angle_rad}")
        print(f"external_force_direction_mode: {env_config.external_force_direction_mode}")
        print(f"external_force_lateral_probability: {env_config.external_force_lateral_probability}")
        print(f"external_force_torque_max_nm: {env_config.external_force_torque_max_nm}")
        if env_config.external_force_mode == "spring":
            print(f"external_force_spring_stiffness_range: {env_config.external_force_spring_stiffness_range}")
            print(f"external_force_spring_damping: {env_config.external_force_spring_damping}")
            print(f"external_force_guiding_probability: {env_config.external_force_guiding_probability}")
            print(f"external_force_transition_s: {env_config.external_force_transition_s}")
            print(f"external_force_net_force_limit_n: {env_config.external_force_net_force_limit_n}")
            print(f"external_force_net_torque_limit_nm: {env_config.external_force_net_torque_limit_nm}")
            print(f"external_force_reference_mass: {env_config.external_force_reference_mass}")
            print(f"external_force_reference_damping: {env_config.external_force_reference_damping}")
            print(f"external_force_reference_velocity_clip: {env_config.external_force_reference_velocity_clip}")
            print(
                "external_force_reference_acceleration_clip: "
                f"{env_config.external_force_reference_acceleration_clip}"
            )
            print(f"external_force_safe_limit_min_n: {env_config.external_force_safe_limit_min_n}")
            print(f"external_force_safe_limit_max_n: {env_config.external_force_safe_limit_max_n}")
            print(f"external_force_safe_margin_n: {env_config.external_force_safe_margin_n}")
            print(f"include_external_force_observation: {env_config.include_external_force_observation}")
    print(f"wrote {output}")
    if isinstance(env, VecNormalize):
        print(f"wrote {vecnormalize_output}")
    return 0


def _optional_vec(
    x: float | None,
    y: float | None,
    z: float | None,
) -> tuple[float, float, float] | None:
    if x is None and y is None and z is None:
        return None
    return (float(x or 0.0), float(y or 0.0), float(z or 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
