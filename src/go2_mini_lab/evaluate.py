from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

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
from .diagnostics import ACTION_BALANCE_KEYS, CONTACT_BALANCE_KEYS, action_balance_metrics, contact_balance_metrics
from .force_metrics import ForceComplianceTracker, aggregate_force_compliance_v2
from .gym_env import Go2StandBalanceGymEnv
from .rl_env import RL_TASKS, make_task_config
from .sb3_tools import load_vecnormalize_if_available, make_vec_env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Go2 PPO locomotion policy with task-level metrics."
    )
    parser.add_argument("checkpoint", type=Path, help="Stable-Baselines3 PPO checkpoint path.")
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
        help="Path to the Go2 MuJoCo MJCF scene.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes.")
    parser.add_argument("--duration", type=float, default=8.0, help="Episode duration in seconds.")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Environment control timestep.")
    parser.add_argument(
        "--target-forward-velocity",
        type=float,
        help="Target x velocity in m/s for fixed-command velocity tasks. Default is task-specific.",
    )
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument(
        "--force-yielding-command-mode",
        choices=("scaled", "unit", "unit_pulse"),
        default="scaled",
        help=(
            "Command-yielding mode: scaled uses gain times force; unit uses the clipped unit force "
            "direction; unit_pulse uses a delayed clipped unit-direction pulse."
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
    add_observation_mode_arg(parser)
    add_policy_action_mode_args(parser)
    add_fullbody_reference_args(parser)
    add_force_safety_trigger_args(parser)
    add_force_response_router_args(parser)
    parser.add_argument("--pd-kp", type=float)
    parser.add_argument("--pd-kd", type=float)
    parser.add_argument("--torque-limit", type=float)
    parser.add_argument(
        "--force-impedance-mode",
        choices=("off", "onset", "active", "two_phase"),
        default="off",
        help="Eval-only PD impedance probe mode for external-force windows.",
    )
    parser.add_argument(
        "--force-impedance-joint-scope",
        choices=("all", "hip", "thigh", "calf", "hip_calf", "stance_hip_calf"),
        default="all",
        help="Joint family affected by the force impedance probe.",
    )
    parser.add_argument(
        "--force-impedance-kp-scale",
        type=float,
        default=1.0,
        help="Kp multiplier during the force impedance yield window.",
    )
    parser.add_argument(
        "--force-impedance-kd-scale",
        type=float,
        default=1.0,
        help="Kd multiplier during the force impedance yield window.",
    )
    parser.add_argument(
        "--force-impedance-delay-s",
        type=float,
        default=0.0,
        help="Delay after force onset before applying impedance modulation.",
    )
    parser.add_argument(
        "--force-impedance-hold-s",
        type=float,
        default=0.15,
        help="Onset-mode low-Kp hold duration in seconds.",
    )
    parser.add_argument(
        "--force-impedance-recovery-s",
        type=float,
        default=0.10,
        help="Linear ramp duration back to nominal Kp/Kd after yielding.",
    )
    parser.add_argument("--force-impedance-tail-kp-scale", type=float, default=1.0)
    parser.add_argument("--force-impedance-tail-kd-scale", type=float, default=1.0)
    parser.add_argument(
        "--force-reference-governor-mode",
        choices=("off", "onset", "active", "two_phase"),
        default="off",
        help="Eval-time event-triggered reference governor for force yielding.",
    )
    parser.add_argument(
        "--force-reference-governor-admittance",
        type=float,
        default=0.0,
        help="Horizontal governor velocity gain in m/s per N of external force.",
    )
    parser.add_argument(
        "--force-reference-governor-damping",
        type=float,
        default=5.0,
        help="Linear damping on the governor position offset.",
    )
    parser.add_argument(
        "--force-reference-governor-offset-clip",
        type=float,
        default=0.0,
        help="Horizontal governor position offset clip in meters.",
    )
    parser.add_argument(
        "--force-reference-governor-velocity-clip",
        type=float,
        default=0.0,
        help="Horizontal governor command velocity clip in m/s.",
    )
    parser.add_argument(
        "--force-reference-governor-delay-s",
        type=float,
        default=0.10,
        help="Delay after force onset before the reference governor activates.",
    )
    parser.add_argument(
        "--force-reference-governor-hold-s",
        type=float,
        default=0.20,
        help="Onset-mode governor hold duration before recovery.",
    )
    parser.add_argument(
        "--force-reference-governor-recovery-s",
        type=float,
        default=0.10,
        help="Onset-mode governor gate recovery duration.",
    )
    parser.add_argument("--force-reference-governor-tail-admittance-scale", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-tail-offset-clip-scale", type=float, default=1.0)
    parser.add_argument("--force-reference-governor-tail-velocity-clip-scale", type=float, default=1.0)
    add_velocity_reward_args(parser)
    parser.add_argument("--randomize-commands", action="store_true")
    parser.add_argument("--fixed-command", action="store_true")
    parser.add_argument("--push-time", type=float)
    parser.add_argument("--push-vx", type=float)
    parser.add_argument("--push-vy", type=float)
    parser.add_argument("--push-vz", type=float)
    parser.add_argument("--push-wx", type=float)
    parser.add_argument("--push-wy", type=float)
    parser.add_argument("--push-wz", type=float)
    add_external_force_args(parser)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument(
        "--force-admittance-residual",
        action="store_true",
        help=(
            "Ignore PPO actions during evaluation and use an explicit external-force "
            "admittance residual action. Intended for onnx_residual controllability probes."
        ),
    )
    parser.add_argument(
        "--force-admittance-lateral-gain",
        type=float,
        default=0.0,
        help="Residual hip gain applied to lateral external force divided by force scale.",
    )
    parser.add_argument(
        "--force-admittance-forward-gain",
        type=float,
        default=0.0,
        help="Residual thigh gain applied to forward external force divided by force scale.",
    )
    parser.add_argument(
        "--force-admittance-force-scale",
        type=float,
        default=0.0,
        help="Force scale in N for explicit admittance residuals. Defaults to env force limits.",
    )
    parser.add_argument(
        "--force-admittance-action-clip",
        type=float,
        default=1.0,
        help="Absolute clip for explicit admittance residual actions before env residual scaling.",
    )
    parser.add_argument(
        "--force-admittance-hip-pattern",
        choices=("uniform", "left_right"),
        default="left_right",
        help="How lateral force residuals are mapped onto hip joints.",
    )
    parser.add_argument(
        "--force-action-probe",
        action="store_true",
        help="Add a deterministic force-conditioned delta to the policy action during evaluation.",
    )
    parser.add_argument(
        "--force-action-probe-gain",
        type=float,
        default=0.0,
        help="Probe action gain applied to external force divided by force scale.",
    )
    parser.add_argument(
        "--force-action-probe-sign",
        type=float,
        default=1.0,
        help="Signed multiplier for the force action probe; use +/-1 to test both directions.",
    )
    parser.add_argument(
        "--force-action-probe-force-scale",
        type=float,
        default=0.0,
        help="Force scale in N for force action probe. Defaults to env force limits.",
    )
    parser.add_argument(
        "--force-action-probe-action-clip",
        type=float,
        default=0.20,
        help="Absolute clip for each additive force action probe component.",
    )
    parser.add_argument(
        "--force-action-probe-joint-scope",
        choices=("hip", "thigh", "hip_thigh"),
        default="hip",
        help="Joint family affected by the additive force action probe.",
    )
    parser.add_argument(
        "--force-action-probe-window-start-s",
        type=float,
        default=0.10,
        help="Start time after force onset for the additive force action probe.",
    )
    parser.add_argument(
        "--force-action-probe-window-end-s",
        type=float,
        default=0.40,
        help="End time after force onset for the additive force action probe.",
    )
    parser.add_argument(
        "--force-action-probe-hip-pattern",
        choices=("uniform", "left_right"),
        default="left_right",
        help="How lateral force probe deltas are mapped onto hip joints.",
    )
    parser.add_argument(
        "--joint-pulse-probe",
        action="store_true",
        help="Add a deterministic joint-group pulse during a force-onset time window.",
    )
    parser.add_argument(
        "--joint-pulse-probe-group",
        choices=("hip_left_right", "front_rear_thigh", "all_hip", "stance_only"),
        default="hip_left_right",
        help="Joint group/pattern affected by the joint pulse controllability probe.",
    )
    parser.add_argument(
        "--joint-pulse-probe-amplitude",
        type=float,
        default=0.0,
        help="Raw action-space pulse amplitude before env residual scaling and smoothing.",
    )
    parser.add_argument(
        "--joint-pulse-probe-sign",
        type=float,
        default=1.0,
        help="Signed multiplier for the joint pulse probe; use +/-1 to test both directions.",
    )
    parser.add_argument(
        "--joint-pulse-probe-window-start-s",
        type=float,
        default=0.20,
        help="Start time after force onset for the joint pulse probe.",
    )
    parser.add_argument(
        "--joint-pulse-probe-window-end-s",
        type=float,
        default=0.30,
        help="End time after force onset for the joint pulse probe.",
    )
    parser.add_argument(
        "--safety-layer-action-override",
        type=float,
        nargs=4,
        metavar=("KP", "KD", "GOVERNOR", "CLIP"),
        help=(
            "Override onnx_safety_layer PPO output with a fixed 4D safety-layer action. "
            "Use for oracle/max-authority controller sweeps without PPO learning."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        help="VecNormalize stats path. Defaults to <checkpoint>_vecnormalize.pkl when present.",
    )
    parser.add_argument(
        "--no-vecnormalize",
        action="store_true",
        help="Do not load VecNormalize stats even if the default stats file exists.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation.json"),
        help="Output evaluation JSON report.",
    )
    parser.add_argument(
        "--trace-steps-jsonl",
        type=Path,
        help=(
            "Optional JSONL path for per-step force diagnostics. Rows include force excess, "
            "residual norms, and base motion projected onto the external-force axis."
        ),
    )
    parser.add_argument(
        "--replay-bank",
        type=Path,
        help="Optional force replay bank JSON. Each evaluation episode starts from one bank event state.",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        from stable_baselines3 import PPO
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
    velocity_reward_overrides = parse_reward_overrides(args.reward_term)
    env_config = make_task_config(
        task=args.task,
        control_dt=args.control_dt,
        episode_length_s=args.duration,
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
    env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
    vec_env = make_vec_env(model_path=args.model, config=env_config)
    normalizer, normalizer_path = load_vecnormalize_if_available(
        vec_env=vec_env,
        checkpoint_path=args.checkpoint,
        vecnormalize_path=args.vecnormalize,
        disabled=args.no_vecnormalize,
    )
    model = PPO.load(str(args.checkpoint), device="cpu")

    trace_rows: list[dict[str, Any]] | None = [] if args.trace_steps_jsonl else None
    replay_states: list[dict[str, Any]] | None = None
    if args.replay_bank is not None:
        replay_bank = json.loads(args.replay_bank.read_text(encoding="utf-8"))
        replay_states = _replay_states_from_bank(replay_bank, limit=max(1, args.episodes))
        if not replay_states:
            raise ValueError(f"replay bank {args.replay_bank} does not contain replayable events")

    episodes = []
    episode_count = len(replay_states) if replay_states is not None else max(1, args.episodes)
    for index in range(episode_count):
        episodes.append(
            _evaluate_episode(
                env=env,
                model=model,
                normalizer=normalizer if normalizer_path else None,
                deterministic=args.deterministic,
                seed=args.seed + index,
                episode=index,
                replay_state=replay_states[index] if replay_states is not None else None,
                trace_rows=trace_rows,
                force_admittance_residual=args.force_admittance_residual,
                force_admittance_lateral_gain=args.force_admittance_lateral_gain,
                force_admittance_forward_gain=args.force_admittance_forward_gain,
                force_admittance_force_scale=args.force_admittance_force_scale,
                force_admittance_action_clip=args.force_admittance_action_clip,
                force_admittance_hip_pattern=args.force_admittance_hip_pattern,
                force_action_probe=args.force_action_probe,
                force_action_probe_gain=args.force_action_probe_gain,
                force_action_probe_sign=args.force_action_probe_sign,
                force_action_probe_force_scale=args.force_action_probe_force_scale,
                force_action_probe_action_clip=args.force_action_probe_action_clip,
                force_action_probe_joint_scope=args.force_action_probe_joint_scope,
                force_action_probe_window_start_s=args.force_action_probe_window_start_s,
                force_action_probe_window_end_s=args.force_action_probe_window_end_s,
                force_action_probe_hip_pattern=args.force_action_probe_hip_pattern,
                joint_pulse_probe=args.joint_pulse_probe,
                joint_pulse_probe_group=args.joint_pulse_probe_group,
                joint_pulse_probe_amplitude=args.joint_pulse_probe_amplitude,
                joint_pulse_probe_sign=args.joint_pulse_probe_sign,
                joint_pulse_probe_window_start_s=args.joint_pulse_probe_window_start_s,
                joint_pulse_probe_window_end_s=args.joint_pulse_probe_window_end_s,
                safety_layer_action_override=args.safety_layer_action_override,
            )
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "model": str(args.model),
        "vecnormalize": str(normalizer_path) if normalizer_path else None,
        "deterministic": args.deterministic,
        "config": {
            "duration": args.duration,
            "replay_bank": str(args.replay_bank) if args.replay_bank else None,
            "replay_event_count": len(replay_states) if replay_states is not None else 0,
            "control_dt": args.control_dt,
            "target_forward_velocity": env_config.target_forward_velocity,
            "target_lateral_velocity": env_config.target_lateral_velocity,
            "target_yaw_rate": env_config.target_yaw_rate,
            "force_yielding_command_mode": env_config.force_yielding_command_mode,
            "force_yielding_command_velocity_per_n": env_config.force_yielding_command_velocity_per_n,
            "force_yielding_command_velocity_clip": env_config.force_yielding_command_velocity_clip,
            "force_yielding_command_pulse_start_s": env_config.force_yielding_command_pulse_start_s,
            "force_yielding_command_pulse_duration_s": env_config.force_yielding_command_pulse_duration_s,
            "force_yielding_command_pulse_recovery_s": env_config.force_yielding_command_pulse_recovery_s,
            "force_yielding_command_pulse_post_clip": env_config.force_yielding_command_pulse_post_clip,
            "force_impedance_mode": env_config.force_impedance_mode,
            "force_impedance_joint_scope": env_config.force_impedance_joint_scope,
            "force_impedance_kp_scale": env_config.force_impedance_kp_scale,
            "force_impedance_kd_scale": env_config.force_impedance_kd_scale,
            "force_impedance_delay_s": env_config.force_impedance_delay_s,
            "force_impedance_hold_s": env_config.force_impedance_hold_s,
            "force_impedance_recovery_s": env_config.force_impedance_recovery_s,
            "force_reference_governor_mode": env_config.force_reference_governor_mode,
            "force_reference_governor_admittance_mps_per_n": (
                env_config.force_reference_governor_admittance_mps_per_n
            ),
            "force_reference_governor_damping": env_config.force_reference_governor_damping,
            "force_reference_governor_offset_clip_m": env_config.force_reference_governor_offset_clip_m,
            "force_reference_governor_velocity_clip_mps": env_config.force_reference_governor_velocity_clip_mps,
            "force_reference_governor_delay_s": env_config.force_reference_governor_delay_s,
            "force_reference_governor_hold_s": env_config.force_reference_governor_hold_s,
            "force_reference_governor_recovery_s": env_config.force_reference_governor_recovery_s,
            "force_response_router_mode": env_config.force_response_router_mode,
            "force_response_profile": env_config.force_response_profile,
            "force_response_foot_kp_scale": env_config.force_response_foot_kp_scale,
            "force_response_foot_kd_scale": env_config.force_response_foot_kd_scale,
            "force_safety_trigger_source": env_config.force_safety_trigger_source,
            "force_safety_history_estimator_path": env_config.force_safety_history_estimator_path,
            "force_safety_detector_linear_acceleration_threshold": (
                env_config.force_safety_detector_linear_acceleration_threshold
            ),
            "force_safety_detector_angular_acceleration_threshold": (
                env_config.force_safety_detector_angular_acceleration_threshold
            ),
            "force_safety_detector_joint_error_threshold": env_config.force_safety_detector_joint_error_threshold,
            "force_safety_detector_joint_velocity_threshold": (
                env_config.force_safety_detector_joint_velocity_threshold
            ),
            "force_safety_detector_contact_loss": env_config.force_safety_detector_contact_loss,
            "force_safety_detector_enable_after_s": env_config.force_safety_detector_enable_after_s,
            "force_safety_detector_hold_s": env_config.force_safety_detector_hold_s,
            "force_safety_detector_recovery_s": env_config.force_safety_detector_recovery_s,
            "randomize_commands": env_config.randomize_commands,
            "action_scale": env_config.action_scale,
            "action_smoothing": env_config.action_smoothing,
            "reward_scale": env_config.reward_scale,
            "reset_settle_s": env_config.reset_settle_s,
            "velocity_pose_profile": env_config.velocity_pose_profile,
            "policy_leg_order": args.policy_leg_order,
            "velocity_reward_profile": env_config.velocity_reward_profile,
            "velocity_command_frame": env_config.velocity_command_frame,
            "velocity_reward_overrides": velocity_reward_overrides,
            "pd_kp": env_config.pd.kp,
            "pd_kd": env_config.pd.kd,
            "torque_limit": env_config.pd.torque_limit,
            "push_time_s": env_config.push_time_s,
            "push_linear_velocity": env_config.push_linear_velocity,
            "push_angular_velocity": env_config.push_angular_velocity,
            "external_force_mode": env_config.external_force_mode,
            "external_force_probability": env_config.external_force_probability,
            "external_force_body_names": env_config.external_force_body_names,
            "external_force_active_body_count": env_config.external_force_active_body_count,
            "external_force_event_count_range": env_config.external_force_event_count_range,
            "external_force_rest_s_range": env_config.external_force_rest_s_range,
            "external_force_start_s_range": env_config.external_force_start_s_range,
            "external_force_duration_s_range": env_config.external_force_duration_s_range,
            "external_force_min_n": env_config.external_force_min_n,
            "external_force_max_n": env_config.external_force_max_n,
            "external_force_z_fraction": env_config.external_force_z_fraction,
            "external_force_direction_angle_rad": env_config.external_force_direction_angle_rad,
            "external_force_direction_mode": env_config.external_force_direction_mode,
            "external_force_lateral_probability": env_config.external_force_lateral_probability,
            "external_force_torque_max_nm": env_config.external_force_torque_max_nm,
            "external_force_spring_stiffness_range": env_config.external_force_spring_stiffness_range,
            "external_force_spring_damping": env_config.external_force_spring_damping,
            "external_force_guiding_probability": env_config.external_force_guiding_probability,
            "external_force_transition_s": env_config.external_force_transition_s,
            "external_force_net_force_limit_n": env_config.external_force_net_force_limit_n,
            "external_force_net_torque_limit_nm": env_config.external_force_net_torque_limit_nm,
            "external_force_reference_mass": env_config.external_force_reference_mass,
            "external_force_reference_damping": env_config.external_force_reference_damping,
            "external_force_reference_velocity_clip": env_config.external_force_reference_velocity_clip,
            "external_force_reference_acceleration_clip": env_config.external_force_reference_acceleration_clip,
            "external_force_safe_limit_min_n": env_config.external_force_safe_limit_min_n,
            "external_force_safe_limit_max_n": env_config.external_force_safe_limit_max_n,
            "external_force_safe_margin_n": env_config.external_force_safe_margin_n,
            "include_external_force_observation": env_config.include_external_force_observation,
            "observation_mode": env_config.observation_mode,
            "policy_action_mode": env_config.policy_action_mode,
            "onnx_policy_path": env_config.onnx_policy_path,
            "onnx_normalizer_checkpoint": env_config.onnx_normalizer_checkpoint,
            "residual_action_scale": env_config.residual_action_scale,
            "force_admittance_residual": args.force_admittance_residual,
            "force_admittance_lateral_gain": args.force_admittance_lateral_gain,
            "force_admittance_forward_gain": args.force_admittance_forward_gain,
            "force_admittance_force_scale": args.force_admittance_force_scale,
            "force_admittance_action_clip": args.force_admittance_action_clip,
            "force_admittance_hip_pattern": args.force_admittance_hip_pattern,
            "force_action_probe": args.force_action_probe,
            "force_action_probe_gain": args.force_action_probe_gain,
            "force_action_probe_sign": args.force_action_probe_sign,
            "force_action_probe_force_scale": args.force_action_probe_force_scale,
            "force_action_probe_action_clip": args.force_action_probe_action_clip,
            "force_action_probe_joint_scope": args.force_action_probe_joint_scope,
            "force_action_probe_window_start_s": args.force_action_probe_window_start_s,
            "force_action_probe_window_end_s": args.force_action_probe_window_end_s,
            "force_action_probe_hip_pattern": args.force_action_probe_hip_pattern,
            "joint_pulse_probe": args.joint_pulse_probe,
            "joint_pulse_probe_group": args.joint_pulse_probe_group,
            "joint_pulse_probe_amplitude": args.joint_pulse_probe_amplitude,
            "joint_pulse_probe_sign": args.joint_pulse_probe_sign,
            "joint_pulse_probe_window_start_s": args.joint_pulse_probe_window_start_s,
            "joint_pulse_probe_window_end_s": args.joint_pulse_probe_window_end_s,
            "safety_layer_action_override": args.safety_layer_action_override,
            "fullbody_reference_mode": env_config.fullbody_reference_mode,
            "nominal_reference_dataset": env_config.nominal_reference_dataset,
            "gait_frequency_hz": env_config.gait_frequency_hz,
            "gait_step_length": env_config.gait_step_length,
            "gait_swing_height": env_config.gait_swing_height,
            "gait_joint_thigh_amplitude": env_config.gait_joint_thigh_amplitude,
            "gait_joint_calf_amplitude": env_config.gait_joint_calf_amplitude,
        },
        "aggregate": _aggregate_episodes(episodes),
        "episodes": episodes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.trace_steps_jsonl and trace_rows is not None:
        args.trace_steps_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.trace_steps_jsonl.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows),
            encoding="utf-8",
        )
        print(f"wrote {args.trace_steps_jsonl}")
    _print_summary(report)
    print(f"wrote {args.output}")
    return 0


def _evaluate_episode(
    *,
    env: Go2StandBalanceGymEnv,
    model: Any,
    normalizer: Any | None,
    deterministic: bool,
    seed: int,
    episode: int = 0,
    replay_state: dict[str, Any] | None = None,
    trace_rows: list[dict[str, Any]] | None = None,
    force_admittance_residual: bool = False,
    force_admittance_lateral_gain: float = 0.0,
    force_admittance_forward_gain: float = 0.0,
    force_admittance_force_scale: float = 0.0,
    force_admittance_action_clip: float = 1.0,
    force_admittance_hip_pattern: str = "left_right",
    force_action_probe: bool = False,
    force_action_probe_gain: float = 0.0,
    force_action_probe_sign: float = 1.0,
    force_action_probe_force_scale: float = 0.0,
    force_action_probe_action_clip: float = 0.20,
    force_action_probe_joint_scope: str = "hip",
    force_action_probe_window_start_s: float = 0.10,
    force_action_probe_window_end_s: float = 0.40,
    force_action_probe_hip_pattern: str = "left_right",
    joint_pulse_probe: bool = False,
    joint_pulse_probe_group: str = "hip_left_right",
    joint_pulse_probe_amplitude: float = 0.0,
    joint_pulse_probe_sign: float = 1.0,
    joint_pulse_probe_window_start_s: float = 0.20,
    joint_pulse_probe_window_end_s: float = 0.30,
    safety_layer_action_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    reset_options = {"replay_state": replay_state} if replay_state is not None else None
    obs, reset_info = env.reset(seed=seed, options=reset_options)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    base_z_values: list[float] = []
    base_positions: list[tuple[float, float, float]] = []
    forward_velocities: list[float] = []
    lateral_velocities: list[float] = []
    yaw_rates: list[float] = []
    base_yaws: list[float] = []
    post_force_forward_velocities: list[float] = []
    post_force_abs_lateral_velocities: list[float] = []
    post_force_abs_yaw_rates: list[float] = []
    force_base_z_values: list[float] = []
    post_force_base_z_values: list[float] = []
    force_downward_velocities: list[float] = []
    force_orientation_errors: list[float] = []
    force_contact_counts: list[int] = []
    speed_values: list[float] = []
    velocity_errors: list[float] = []
    yaw_rate_errors: list[float] = []
    orientation_errors: list[float] = []
    action_norms: list[float] = []
    onnx_action_norms: list[float] = []
    residual_action_norms: list[float] = []
    safety_layer_action_norms: list[float] = []
    final_action_norms: list[float] = []
    final_minus_onnx_action_norms: list[float] = []
    force_active_residual_action_norms: list[float] = []
    force_active_safety_layer_action_norms: list[float] = []
    force_active_final_minus_onnx_action_norms: list[float] = []
    force_action_probe_delta_norms: list[float] = []
    force_active_action_probe_delta_norms: list[float] = []
    joint_pulse_probe_delta_norms: list[float] = []
    force_active_joint_pulse_delta_norms: list[float] = []
    force_onset_residual_action_norms: list[float] = []
    force_onset_safety_layer_action_norms: list[float] = []
    force_onset_external_force_excess_values: list[float] = []
    action_balance_values = defaultdict(list)
    external_force_steps = 0
    external_force_seen = False
    external_force_event_counts: list[int] = []
    external_force_magnitudes: list[float] = []
    external_force_torque_magnitudes: list[float] = []
    external_force_excess_values: list[float] = []
    external_force_step_compliance: list[bool] = []
    external_force_safe_limit_n = 0.0
    contact_counts = defaultdict(int)
    reward_terms = defaultdict(float)
    reward_raw_terms = defaultdict(float)
    last_info: dict[str, Any] = dict(reset_info or {})
    force_onset_window_s = max(
        0.0,
        float(env.env.config.velocity_reward.unsafe_force_onset_window_s),
    )
    force_compliance = ForceComplianceTracker(
        dt_s=env.env.config.control_dt,
        onset_window_s=force_onset_window_s,
    )
    force_admittance_scale = (
        float(force_admittance_force_scale)
        if force_admittance_force_scale > 0.0
        else max(
            1.0,
            float(env.env.config.external_force_max_n),
            float(env.env.config.external_force_net_force_limit_n),
        )
    )
    force_action_probe_scale = (
        float(force_action_probe_force_scale)
        if force_action_probe_force_scale > 0.0
        else max(
            1.0,
            float(env.env.config.external_force_max_n),
            float(env.env.config.external_force_net_force_limit_n),
        )
    )

    while not terminated and not truncated:
        force_action_probe_delta_norm = None
        joint_pulse_probe_delta_norm = None
        if force_admittance_residual:
            action = _force_admittance_residual_action(
                info=last_info,
                action_names=last_info.get("action_names") or env.env.joint_names,
                lateral_gain=force_admittance_lateral_gain,
                forward_gain=force_admittance_forward_gain,
                force_scale_n=force_admittance_scale,
                action_clip=force_admittance_action_clip,
                hip_pattern=force_admittance_hip_pattern,
            )
        else:
            policy_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(policy_obs, deterministic=deterministic)
        if safety_layer_action_override is not None:
            action = _override_safety_layer_action(action, safety_layer_action_override)
        if force_action_probe:
            force_action_probe_delta = _force_action_probe_delta(
                info=last_info,
                action_names=last_info.get("action_names") or env.env.joint_names,
                gain=force_action_probe_gain,
                sign=force_action_probe_sign,
                force_scale_n=force_action_probe_scale,
                action_clip=force_action_probe_action_clip,
                joint_scope=force_action_probe_joint_scope,
                window_start_s=force_action_probe_window_start_s,
                window_end_s=force_action_probe_window_end_s,
                hip_pattern=force_action_probe_hip_pattern,
            )
            force_action_probe_delta_norm = _vector_norm(force_action_probe_delta)
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            delta_array = np.asarray(force_action_probe_delta, dtype=np.float64).reshape(-1)
            if action_array.shape != delta_array.shape:
                raise ValueError(
                    "force action probe delta shape does not match action: "
                    f"{delta_array.shape} vs {action_array.shape}"
                )
            action = action_array + delta_array
        if joint_pulse_probe:
            joint_pulse_delta = _joint_pulse_probe_delta(
                info=last_info,
                action_names=last_info.get("action_names") or env.env.joint_names,
                group=joint_pulse_probe_group,
                amplitude=joint_pulse_probe_amplitude,
                sign=joint_pulse_probe_sign,
                window_start_s=joint_pulse_probe_window_start_s,
                window_end_s=joint_pulse_probe_window_end_s,
            )
            joint_pulse_probe_delta_norm = _vector_norm(joint_pulse_delta)
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            delta_array = np.asarray(joint_pulse_delta, dtype=np.float64).reshape(-1)
            if action_array.shape != delta_array.shape:
                raise ValueError(
                    "joint pulse probe delta shape does not match action: "
                    f"{delta_array.shape} vs {action_array.shape}"
                )
            action = action_array + delta_array
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        force_compliance.observe(info)
        total_reward += float(reward)
        steps += 1

        base_z_values.append(float(info.get("base_z", 0.0)))
        base_linear_velocity_world = info.get("base_linear_velocity") or [0.0, 0.0, 0.0]
        base_linear_velocity = info.get("base_linear_velocity_tracking_frame") or base_linear_velocity_world
        base_angular_velocity = info.get("base_angular_velocity") or [0.0, 0.0, 0.0]
        command = info.get("command") or [0.0, 0.0, 0.0]
        projected_gravity = info.get("projected_gravity") or [0.0, 0.0, -1.0]
        contacts_info = info.get("contacts") or {}
        last_action = info.get("last_action") or []
        action_names = info.get("action_names") or []
        onnx_action = info.get("onnx_action")
        residual_action = info.get("residual_action")
        safety_layer_action = info.get("safety_layer_action")
        final_action = info.get("final_action")
        onnx_action_norm = _vector_norm(onnx_action)
        residual_action_norm = _vector_norm(residual_action)
        safety_layer_action_norm = _vector_norm(safety_layer_action)
        final_action_norm = _vector_norm(final_action)
        final_minus_onnx_action_norm = _vector_delta_norm(final_action, onnx_action)
        if force_action_probe_delta_norm is not None:
            force_action_probe_delta_norms.append(force_action_probe_delta_norm)
        if joint_pulse_probe_delta_norm is not None:
            joint_pulse_probe_delta_norms.append(joint_pulse_probe_delta_norm)
        if onnx_action_norm is not None:
            onnx_action_norms.append(onnx_action_norm)
        if residual_action_norm is not None:
            residual_action_norms.append(residual_action_norm)
        if safety_layer_action_norm is not None:
            safety_layer_action_norms.append(safety_layer_action_norm)
        if final_action_norm is not None:
            final_action_norms.append(final_action_norm)
        if final_minus_onnx_action_norm is not None:
            final_minus_onnx_action_norms.append(final_minus_onnx_action_norm)
        if trace_rows is not None:
            trace_rows.append(
                _force_trace_row(
                    episode=episode,
                    step=steps,
                    info=info,
                    residual_action_norm=residual_action_norm,
                    final_minus_onnx_action_norm=final_minus_onnx_action_norm,
                    force_action_probe_delta_norm=force_action_probe_delta_norm,
                    joint_pulse_probe_delta_norm=joint_pulse_probe_delta_norm,
                )
            )
        base_yaw = info.get("base_yaw")
        base_position = info.get("base_position") or []
        if len(base_position) >= 3:
            base_positions.append(
                (float(base_position[0]), float(base_position[1]), float(base_position[2]))
            )

        vx = float(base_linear_velocity[0])
        vy = float(base_linear_velocity[1])
        vz = float(base_linear_velocity[2])
        yaw_rate = float(base_angular_velocity[2])
        forward_velocities.append(vx)
        lateral_velocities.append(vy)
        yaw_rates.append(yaw_rate)
        if base_yaw is not None:
            base_yaws.append(float(base_yaw))
        speed_values.append(math.sqrt(vx * vx + vy * vy + vz * vz))
        velocity_errors.append(
            math.sqrt(
                (float(command[0]) - vx) ** 2
                + (float(command[1]) - vy) ** 2
            )
        )
        yaw_rate_errors.append(abs(float(command[2]) - yaw_rate))
        orientation_error = math.sqrt(float(projected_gravity[0]) ** 2 + float(projected_gravity[1]) ** 2)
        orientation_errors.append(orientation_error)
        if last_action:
            action_norms.append(math.sqrt(sum(float(value) ** 2 for value in last_action)))
            for name, value in action_balance_metrics(last_action, action_names).items():
                action_balance_values[name].append(value)
        force_active = bool(info.get("external_force_active", False))
        external_force_event_counts.append(int(info.get("external_force_event_count") or 0))
        if force_active:
            external_force_seen = True
            external_force_steps += 1
            external_force_safe_limit_n = float(info.get("external_force_safe_limit_n", 0.0))
            force_base_z_values.append(float(info.get("base_z", 0.0)))
            force_downward_velocities.append(max(0.0, -vz))
            force_orientation_errors.append(orientation_error)
            force_contact_counts.append(sum(1 for active in contacts_info.values() if active))
            external_force_magnitudes.append(float(info.get("external_force_magnitude", 0.0)))
            external_force_torque_magnitudes.append(float(info.get("external_force_torque_magnitude", 0.0)))
            external_force_excess_values.append(float(info.get("external_force_excess_n", 0.0)))
            if residual_action_norm is not None:
                force_active_residual_action_norms.append(residual_action_norm)
            if safety_layer_action_norm is not None:
                force_active_safety_layer_action_norms.append(safety_layer_action_norm)
            if final_minus_onnx_action_norm is not None:
                force_active_final_minus_onnx_action_norms.append(final_minus_onnx_action_norm)
            if force_action_probe_delta_norm is not None:
                force_active_action_probe_delta_norms.append(force_action_probe_delta_norm)
            if joint_pulse_probe_delta_norm is not None:
                force_active_joint_pulse_delta_norms.append(joint_pulse_probe_delta_norm)
            force_start_s = info.get("external_force_start_s")
            force_elapsed_s = (
                float(info.get("t", 0.0)) - float(force_start_s)
                if force_start_s is not None
                else None
            )
            if force_elapsed_s is not None and 0.0 <= force_elapsed_s <= force_onset_window_s:
                if residual_action_norm is not None:
                    force_onset_residual_action_norms.append(residual_action_norm)
                if safety_layer_action_norm is not None:
                    force_onset_safety_layer_action_norms.append(safety_layer_action_norm)
                force_onset_external_force_excess_values.append(
                    float(info.get("external_force_excess_n", 0.0))
                )
            compliant = info.get("external_force_step_compliant")
            if compliant is not None:
                external_force_step_compliance.append(bool(compliant))
        elif external_force_seen:
            post_force_base_z_values.append(float(info.get("base_z", 0.0)))
            post_force_forward_velocities.append(vx)
            post_force_abs_lateral_velocities.append(abs(vy))
            post_force_abs_yaw_rates.append(abs(yaw_rate))

        for foot, active in contacts_info.items():
            if active:
                contact_counts[str(foot)] += 1
        for name, value in (info.get("reward_terms") or {}).items():
            reward_terms[name] += float(value)
        for name, value in (info.get("reward_raw_terms") or {}).items():
            reward_raw_terms[name] += float(value)

    fall_base_z = env.env.config.fall_base_z
    fall_detected = bool(base_z_values and min(base_z_values) < fall_base_z)
    failure_terminated = bool(terminated and not truncated)
    mean_forward_velocity = _mean(forward_velocities)
    mean_abs_lateral_velocity = _mean([abs(value) for value in lateral_velocities])
    mean_abs_yaw_rate = _mean([abs(value) for value in yaw_rates])
    final_base_yaw = base_yaws[-1] if base_yaws else None
    final_base_x = base_positions[-1][0] if base_positions else None
    final_base_y = base_positions[-1][1] if base_positions else None
    final_lateral_displacement = (
        base_positions[-1][1] - base_positions[0][1]
        if len(base_positions) >= 2
        else None
    )
    forward_displacement = (
        base_positions[-1][0] - base_positions[0][0]
        if len(base_positions) >= 2
        else None
    )
    path_length_xy = _path_length_xy(base_positions)
    straightness_ratio = (
        max(0.0, float(forward_displacement)) / path_length_xy
        if forward_displacement is not None and path_length_xy > 1e-9
        else None
    )
    straightness_success = (
        not (fall_detected or failure_terminated)
        and mean_forward_velocity >= 0.35
        and mean_abs_lateral_velocity <= 0.08
        and mean_abs_yaw_rate <= 0.12
        and final_base_yaw is not None
        and abs(final_base_yaw) <= 0.35
        and final_lateral_displacement is not None
        and abs(final_lateral_displacement) <= 0.35
    )
    divisor = max(steps, 1)
    contact_ratio = {
        foot: contact_counts[foot] / divisor
        for foot in ("FR", "FL", "RR", "RL")
    }
    report = {
        "seed": seed,
        "steps": steps,
        "duration": steps * env.env.config.control_dt,
        "total_reward": total_reward,
        "mean_reward": total_reward / divisor,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "fall_detected": fall_detected or failure_terminated,
        "push_applied": bool(last_info.get("push_applied", False)),
        "external_force_applied": external_force_steps > 0,
        "external_force_steps": external_force_steps,
        "external_force_event_count": max(external_force_event_counts) if external_force_event_counts else 0,
        "external_force_max_magnitude": max(external_force_magnitudes) if external_force_magnitudes else 0.0,
        "external_force_max_torque_magnitude": (
            max(external_force_torque_magnitudes) if external_force_torque_magnitudes else 0.0
        ),
        "external_force_safe_limit_n": external_force_safe_limit_n,
        "external_force_mean_excess_n": _mean(external_force_excess_values),
        "external_force_max_excess_n": (
            max(external_force_excess_values) if external_force_excess_values else 0.0
        ),
        "force_active_step_compliance_rate": (
            sum(external_force_step_compliance) / len(external_force_step_compliance)
            if external_force_step_compliance
            else 1.0
        ),
        "episode_compliant": (
            sum(external_force_step_compliance) / len(external_force_step_compliance) >= 0.95
            if external_force_step_compliance
            else True
        ),
        "command": last_info.get("command") or [0.0, 0.0, 0.0],
        "base_z_min": min(base_z_values) if base_z_values else None,
        "base_z_mean": _mean(base_z_values),
        "min_base_z_during_force": min(force_base_z_values) if force_base_z_values else None,
        "min_base_z_after_force": min(post_force_base_z_values) if post_force_base_z_values else None,
        "max_downward_velocity_during_force": max(force_downward_velocities) if force_downward_velocities else None,
        "max_orientation_error_during_force": max(force_orientation_errors) if force_orientation_errors else None,
        "min_contact_count_during_force": min(force_contact_counts) if force_contact_counts else None,
        "mean_forward_velocity": mean_forward_velocity,
        "mean_lateral_velocity": _mean(lateral_velocities),
        "mean_abs_lateral_velocity": mean_abs_lateral_velocity,
        "mean_yaw_rate": _mean(yaw_rates),
        "mean_abs_yaw_rate": mean_abs_yaw_rate,
        "mean_abs_base_yaw": _mean([abs(value) for value in base_yaws]),
        "final_base_yaw": final_base_yaw,
        "final_base_x": final_base_x,
        "final_base_y": final_base_y,
        "final_lateral_displacement": final_lateral_displacement,
        "abs_final_lateral_displacement": (
            abs(final_lateral_displacement) if final_lateral_displacement is not None else None
        ),
        "forward_displacement": forward_displacement,
        "path_length_xy": path_length_xy,
        "straightness_ratio": straightness_ratio,
        "straightness_success": straightness_success,
        "mean_speed": _mean(speed_values),
        "post_force_mean_forward_velocity": (
            _mean(post_force_forward_velocities)
            if post_force_forward_velocities
            else None
        ),
        "post_force_mean_abs_lateral_velocity": (
            _mean(post_force_abs_lateral_velocities)
            if post_force_abs_lateral_velocities
            else None
        ),
        "post_force_mean_abs_yaw_rate": (
            _mean(post_force_abs_yaw_rates)
            if post_force_abs_yaw_rates
            else None
        ),
        "mean_velocity_error_xy": _mean(velocity_errors),
        "mean_yaw_rate_error": _mean(yaw_rate_errors),
        "mean_orientation_error": _mean(orientation_errors),
        "mean_action_norm": _mean(action_norms),
        "mean_onnx_action_norm": _mean(onnx_action_norms),
        "mean_residual_action_norm": _mean(residual_action_norms),
        "mean_safety_layer_action_norm": _mean(safety_layer_action_norms),
        "mean_final_action_norm": _mean(final_action_norms),
        "mean_final_minus_onnx_action_norm": _mean(final_minus_onnx_action_norms),
        "mean_force_action_probe_delta_norm": _mean(force_action_probe_delta_norms),
        "mean_joint_pulse_probe_delta_norm": _mean(joint_pulse_probe_delta_norms),
        "force_active_mean_residual_action_norm": _mean_or_none(force_active_residual_action_norms),
        "force_active_max_residual_action_norm": _max_or_none(force_active_residual_action_norms),
        "force_active_mean_safety_layer_action_norm": _mean_or_none(
            force_active_safety_layer_action_norms
        ),
        "force_active_max_safety_layer_action_norm": _max_or_none(
            force_active_safety_layer_action_norms
        ),
        "force_active_mean_final_minus_onnx_action_norm": _mean_or_none(
            force_active_final_minus_onnx_action_norms
        ),
        "force_active_mean_action_probe_delta_norm": _mean_or_none(
            force_active_action_probe_delta_norms
        ),
        "force_active_mean_joint_pulse_delta_norm": _mean_or_none(
            force_active_joint_pulse_delta_norms
        ),
        "force_onset_mean_residual_action_norm": _mean_or_none(force_onset_residual_action_norms),
        "force_onset_mean_safety_layer_action_norm": _mean_or_none(force_onset_safety_layer_action_norms),
        "force_onset_max_external_force_excess_n": _max_or_none(force_onset_external_force_excess_values),
        "contact_ratio": contact_ratio,
        "contact_imbalance": _contact_imbalance(contact_ratio),
        "reward_terms_mean": {
            name: value / divisor
            for name, value in sorted(reward_terms.items())
        },
        "reward_raw_terms_mean": {
            name: value / divisor
            for name, value in sorted(reward_raw_terms.items())
        },
    }
    report.update(contact_balance_metrics(contact_ratio))
    report.update(force_compliance.summary())
    report.update({
        name: _mean(action_balance_values[name])
        for name in ACTION_BALANCE_KEYS
    })
    return report


RECOVERY_FIRST_MIN_BASE_Z_DURING_FORCE = 0.22
RECOVERY_FIRST_MIN_BASE_Z_AFTER_FORCE = 0.26
RECOVERY_FIRST_MAX_ORIENTATION_ERROR_DURING_FORCE = 0.65
RECOVERY_FIRST_MIN_POST_FORCE_FORWARD_VELOCITY = 0.25
RECOVERY_FIRST_MAX_POST_FORCE_ABS_LATERAL_VELOCITY = 0.35
RECOVERY_FIRST_MAX_POST_FORCE_ABS_YAW_RATE = 0.45


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _episode_force_recovery_survived(episode: dict[str, Any]) -> bool:
    return bool(episode.get("external_force_applied", False)) and not bool(
        episode.get("fall_detected", False)
    )


def _episode_force_recovery_posture_success(episode: dict[str, Any]) -> bool:
    if not _episode_force_recovery_survived(episode):
        return False
    min_during = _optional_float(episode.get("min_base_z_during_force"))
    min_after = _optional_float(episode.get("min_base_z_after_force"))
    max_orientation = _optional_float(episode.get("max_orientation_error_during_force"))
    return (
        (min_during is None or min_during >= RECOVERY_FIRST_MIN_BASE_Z_DURING_FORCE)
        and (min_after is None or min_after >= RECOVERY_FIRST_MIN_BASE_Z_AFTER_FORCE)
        and (
            max_orientation is None
            or max_orientation <= RECOVERY_FIRST_MAX_ORIENTATION_ERROR_DURING_FORCE
        )
    )


def _episode_force_recovery_locomotion_success(episode: dict[str, Any]) -> bool:
    if not _episode_force_recovery_posture_success(episode):
        return False
    post_forward = _optional_float(episode.get("post_force_mean_forward_velocity"))
    post_lateral = _optional_float(episode.get("post_force_mean_abs_lateral_velocity"))
    post_yaw = _optional_float(episode.get("post_force_mean_abs_yaw_rate"))
    return (
        post_forward is not None
        and post_forward >= RECOVERY_FIRST_MIN_POST_FORCE_FORWARD_VELOCITY
        and (
            post_lateral is None
            or post_lateral <= RECOVERY_FIRST_MAX_POST_FORCE_ABS_LATERAL_VELOCITY
        )
        and (post_yaw is None or post_yaw <= RECOVERY_FIRST_MAX_POST_FORCE_ABS_YAW_RATE)
    )


def _aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(episodes), 1)
    force_recovery_survival_rate = sum(
        _episode_force_recovery_survived(episode) for episode in episodes
    ) / count
    force_recovery_posture_success_rate = sum(
        _episode_force_recovery_posture_success(episode) for episode in episodes
    ) / count
    force_recovery_locomotion_success_rate = sum(
        _episode_force_recovery_locomotion_success(episode) for episode in episodes
    ) / count
    reward_names = sorted({name for episode in episodes for name in episode["reward_terms_mean"]})
    aggregate = {
        "episodes": len(episodes),
        "success_rate": sum(not episode["fall_detected"] for episode in episodes) / count,
        "fall_rate": sum(bool(episode["fall_detected"]) for episode in episodes) / count,
        "straightness_success_rate": sum(bool(episode["straightness_success"]) for episode in episodes) / count,
        "mean_steps": _mean([episode["steps"] for episode in episodes]),
        "mean_total_reward": _mean([episode["total_reward"] for episode in episodes]),
        "mean_reward": _mean([episode["mean_reward"] for episode in episodes]),
        "base_z_min": min(
            episode["base_z_min"]
            for episode in episodes
            if episode["base_z_min"] is not None
        )
        if any(episode["base_z_min"] is not None for episode in episodes)
        else None,
        "min_base_z_during_force": _min_or_none([
            episode.get("min_base_z_during_force")
            for episode in episodes
        ]),
        "min_base_z_after_force": _min_or_none([
            episode.get("min_base_z_after_force")
            for episode in episodes
        ]),
        "max_downward_velocity_during_force": _max_or_none([
            episode.get("max_downward_velocity_during_force")
            for episode in episodes
        ]),
        "max_orientation_error_during_force": _max_or_none([
            episode.get("max_orientation_error_during_force")
            for episode in episodes
        ]),
        "min_contact_count_during_force": _min_or_none([
            episode.get("min_contact_count_during_force")
            for episode in episodes
        ]),
        "mean_forward_velocity": _mean([episode["mean_forward_velocity"] for episode in episodes]),
        "mean_lateral_velocity": _mean([episode["mean_lateral_velocity"] for episode in episodes]),
        "mean_abs_lateral_velocity": _mean([episode["mean_abs_lateral_velocity"] for episode in episodes]),
        "mean_yaw_rate": _mean([episode["mean_yaw_rate"] for episode in episodes]),
        "mean_abs_yaw_rate": _mean([episode["mean_abs_yaw_rate"] for episode in episodes]),
        "mean_abs_base_yaw": _mean([episode["mean_abs_base_yaw"] for episode in episodes]),
        "mean_final_base_yaw": _mean([episode["final_base_yaw"] for episode in episodes]),
        "mean_final_base_x": _mean([episode["final_base_x"] for episode in episodes]),
        "mean_final_base_y": _mean([episode["final_base_y"] for episode in episodes]),
        "mean_abs_final_lateral_displacement": _mean([
            episode["abs_final_lateral_displacement"]
            for episode in episodes
        ]),
        "mean_forward_displacement": _mean([episode["forward_displacement"] for episode in episodes]),
        "mean_path_length_xy": _mean([episode["path_length_xy"] for episode in episodes]),
        "mean_straightness_ratio": _mean([episode["straightness_ratio"] for episode in episodes]),
        "mean_velocity_error_xy": _mean([episode["mean_velocity_error_xy"] for episode in episodes]),
        "mean_yaw_rate_error": _mean([episode["mean_yaw_rate_error"] for episode in episodes]),
        "mean_orientation_error": _mean([episode["mean_orientation_error"] for episode in episodes]),
        "mean_action_norm": _mean([episode["mean_action_norm"] for episode in episodes]),
        "mean_onnx_action_norm": _mean([
            episode.get("mean_onnx_action_norm")
            for episode in episodes
        ]),
        "mean_residual_action_norm": _mean([
            episode.get("mean_residual_action_norm")
            for episode in episodes
        ]),
        "mean_safety_layer_action_norm": _mean([
            episode.get("mean_safety_layer_action_norm")
            for episode in episodes
        ]),
        "mean_final_action_norm": _mean([
            episode.get("mean_final_action_norm")
            for episode in episodes
        ]),
        "mean_final_minus_onnx_action_norm": _mean([
            episode.get("mean_final_minus_onnx_action_norm")
            for episode in episodes
        ]),
        "mean_force_active_residual_action_norm": _mean_or_none([
            episode.get("force_active_mean_residual_action_norm")
            for episode in episodes
        ]),
        "max_force_active_residual_action_norm": _max_or_none([
            episode.get("force_active_max_residual_action_norm")
            for episode in episodes
        ]),
        "mean_force_active_safety_layer_action_norm": _mean_or_none([
            episode.get("force_active_mean_safety_layer_action_norm")
            for episode in episodes
        ]),
        "max_force_active_safety_layer_action_norm": _max_or_none([
            episode.get("force_active_max_safety_layer_action_norm")
            for episode in episodes
        ]),
        "mean_force_active_final_minus_onnx_action_norm": _mean_or_none([
            episode.get("force_active_mean_final_minus_onnx_action_norm")
            for episode in episodes
        ]),
        "mean_force_action_probe_delta_norm": _mean([
            episode.get("mean_force_action_probe_delta_norm")
            for episode in episodes
        ]),
        "mean_force_active_action_probe_delta_norm": _mean_or_none([
            episode.get("force_active_mean_action_probe_delta_norm")
            for episode in episodes
        ]),
        "mean_joint_pulse_probe_delta_norm": _mean([
            episode.get("mean_joint_pulse_probe_delta_norm")
            for episode in episodes
        ]),
        "mean_force_active_joint_pulse_delta_norm": _mean_or_none([
            episode.get("force_active_mean_joint_pulse_delta_norm")
            for episode in episodes
        ]),
        "mean_force_onset_residual_action_norm": _mean_or_none([
            episode.get("force_onset_mean_residual_action_norm")
            for episode in episodes
        ]),
        "mean_force_onset_safety_layer_action_norm": _mean_or_none([
            episode.get("force_onset_mean_safety_layer_action_norm")
            for episode in episodes
        ]),
        "max_force_onset_external_force_excess_n": _max_or_none([
            episode.get("force_onset_max_external_force_excess_n")
            for episode in episodes
        ]),
        "mean_contact_imbalance": _mean([episode["contact_imbalance"] for episode in episodes]),
        "push_recovery_success_rate": sum(
            bool(episode["push_applied"]) and not bool(episode["fall_detected"])
            for episode in episodes
        )
        / count,
        "external_force_recovery_success_rate": sum(
            bool(episode["external_force_applied"]) and not bool(episode["fall_detected"])
            for episode in episodes
        )
        / count,
        "mean_external_force_max_magnitude": _mean([
            episode["external_force_max_magnitude"]
            for episode in episodes
        ]),
        "mean_external_force_max_torque_magnitude": _mean([
            episode["external_force_max_torque_magnitude"]
            for episode in episodes
        ]),
        "mean_external_force_safe_limit_n": _mean([
            episode.get("external_force_safe_limit_n")
            for episode in episodes
        ]),
        "mean_external_force_excess_n": _mean([
            episode.get("external_force_mean_excess_n")
            for episode in episodes
        ]),
        "max_external_force_excess_n": max(
            float(episode.get("external_force_max_excess_n", 0.0))
            for episode in episodes
        ),
        "force_active_step_compliance_rate": _mean([
            episode.get("force_active_step_compliance_rate")
            for episode in episodes
        ]),
        "episode_compliance_rate": sum(
            bool(episode.get("episode_compliant", True))
            for episode in episodes
        )
        / count,
        "mean_external_force_event_count": _mean([
            episode["external_force_event_count"]
            for episode in episodes
        ]),
        "post_force_mean_forward_velocity": _mean_or_none([
            episode["post_force_mean_forward_velocity"]
            for episode in episodes
        ]),
        "post_force_mean_abs_lateral_velocity": _mean_or_none([
            episode["post_force_mean_abs_lateral_velocity"]
            for episode in episodes
        ]),
        "post_force_mean_abs_yaw_rate": _mean_or_none([
            episode["post_force_mean_abs_yaw_rate"]
            for episode in episodes
        ]),
        "reward_terms_mean": {
            name: _mean([
                episode["reward_terms_mean"].get(name, 0.0)
                for episode in episodes
            ])
            for name in reward_names
        },
    }
    aggregate.update(aggregate_force_compliance_v2(episodes))
    aggregate.update({
        f"mean_{name}": _mean([episode[name] for episode in episodes])
        for name in CONTACT_BALANCE_KEYS
    })
    aggregate.update({
        f"mean_{name}": _mean([episode[name] for episode in episodes])
        for name in ACTION_BALANCE_KEYS
    })
    aggregate.update(
        {
            "force_recovery_success_rate": aggregate["external_force_recovery_success_rate"],
            "force_recovery_fall_rate": aggregate["fall_rate"],
            "force_recovery_post_force_mean_forward_velocity": aggregate[
                "post_force_mean_forward_velocity"
            ],
            "force_recovery_post_force_mean_abs_lateral_velocity": aggregate[
                "post_force_mean_abs_lateral_velocity"
            ],
            "force_recovery_post_force_mean_abs_yaw_rate": aggregate[
                "post_force_mean_abs_yaw_rate"
            ],
            "force_recovery_mean_abs_final_lateral_displacement": aggregate[
                "mean_abs_final_lateral_displacement"
            ],
            "force_recovery_survival_rate": force_recovery_survival_rate,
            "force_recovery_posture_success_rate": force_recovery_posture_success_rate,
            "force_recovery_locomotion_success_rate": force_recovery_locomotion_success_rate,
            "force_recovery_min_base_z_during_force": _min_or_none([
                episode.get("min_base_z_during_force") for episode in episodes
            ]),
            "force_recovery_min_base_z_after_force": _min_or_none([
                episode.get("min_base_z_after_force") for episode in episodes
            ]),
            "force_recovery_max_orientation_error_during_force": _max_or_none([
                episode.get("max_orientation_error_during_force") for episode in episodes
            ]),
        }
    )
    return aggregate


def _force_trace_row(
    *,
    episode: int,
    step: int,
    info: dict[str, Any],
    residual_action_norm: float | None,
    final_minus_onnx_action_norm: float | None,
    force_action_probe_delta_norm: float | None = None,
    joint_pulse_probe_delta_norm: float | None = None,
) -> dict[str, Any]:
    force_vector = _as_float3(info.get("external_force_vector"))
    base_position = _as_float3(info.get("base_position"))
    base_velocity = _as_float3(info.get("base_linear_velocity"))
    base_angular_velocity = _as_float3(info.get("base_angular_velocity"))
    projected_gravity = _as_float3(info.get("projected_gravity"))
    force_norm = math.sqrt(sum(value * value for value in force_vector))
    force_direction = (
        [value / force_norm for value in force_vector]
        if force_norm > 1e-9
        else [0.0, 0.0, 0.0]
    )
    force_elapsed_s = None
    force_start_s = info.get("external_force_start_s")
    if force_start_s is not None:
        force_elapsed_s = float(info.get("t", 0.0)) - float(force_start_s)

    return {
        "episode": int(episode),
        "step": int(step),
        "t": float(info.get("t", 0.0)),
        "qpos": _as_float_list(info.get("qpos")),
        "qvel": _as_float_list(info.get("qvel")),
        "ctrl": _as_float_list(info.get("ctrl")),
        "external_force_active": bool(info.get("external_force_active", False)),
        "external_force_mode": info.get("external_force_mode"),
        "external_force_body": info.get("external_force_body"),
        "external_force_schedule": info.get("external_force_schedule") or {"pulses": [], "springs": []},
        "external_force_vector": force_vector,
        "external_force_direction": force_direction,
        "external_force_magnitude_n": float(info.get("external_force_magnitude", force_norm)),
        "external_force_safe_limit_n": float(info.get("external_force_safe_limit_n", 0.0)),
        "external_force_safe_margin_n": float(info.get("external_force_safe_margin_n", 0.0)),
        "external_force_allowed_n": float(info.get("external_force_allowed_n", 0.0)),
        "external_force_excess_n": float(info.get("external_force_excess_n", 0.0)),
        "external_force_step_compliant": info.get("external_force_step_compliant"),
        "force_impedance_kp_scale": float(info.get("force_impedance_kp_scale", 1.0)),
        "force_impedance_kd_scale": float(info.get("force_impedance_kd_scale", 1.0)),
        "force_response_router_mode": info.get("force_response_router_mode", "off"),
        "force_response_class": info.get("force_response_class", "nominal"),
        "force_response_impacted_body": info.get("force_response_impacted_body"),
        "force_response_impacted_leg": info.get("force_response_impacted_leg"),
        "force_response_impacted_leg_is_stance": info.get("force_response_impacted_leg_is_stance"),
        "force_response_governor_enabled": bool(info.get("force_response_governor_enabled", False)),
        "force_response_joint_kp_scales": _as_float_list(info.get("force_response_joint_kp_scales")),
        "force_response_joint_kd_scales": _as_float_list(info.get("force_response_joint_kd_scales")),
        "command": _as_float3(info.get("command")),
        "force_yielding_command_offset": _as_float3(info.get("force_yielding_command_offset")),
        "force_reference_governor_offset": _as_float3(info.get("force_reference_governor_offset")),
        "force_reference_governor_velocity": _as_float3(info.get("force_reference_governor_velocity")),
        "force_reference_governor_gate": float(info.get("force_reference_governor_gate", 0.0)),
        "force_safety_trigger_source": info.get("force_safety_trigger_source"),
        "force_safety_detector_active": bool(info.get("force_safety_detector_active", False)),
        "force_safety_detector_gate": float(info.get("force_safety_detector_gate", 0.0)),
        "force_safety_detector_score": float(info.get("force_safety_detector_score", 0.0)),
        "force_safety_detector_force_proxy": _as_float3(info.get("force_safety_detector_force_proxy")),
        "force_elapsed_s": force_elapsed_s,
        "base_position": base_position,
        "base_linear_velocity": base_velocity,
        "base_angular_velocity": base_angular_velocity,
        "projected_gravity": projected_gravity,
        "base_position_along_force": sum(
            position * direction for position, direction in zip(base_position, force_direction)
        ),
        "base_velocity_along_force": sum(
            velocity * direction for velocity, direction in zip(base_velocity, force_direction)
        ),
        "residual_action_norm": residual_action_norm,
        "safety_layer_action_norm": _vector_norm(info.get("safety_layer_action")),
        "final_minus_onnx_action_norm": final_minus_onnx_action_norm,
        "force_action_probe_delta_norm": force_action_probe_delta_norm,
        "joint_pulse_probe_delta_norm": joint_pulse_probe_delta_norm,
        "joint_target_positions": _as_float_list(info.get("joint_target_positions")),
        "joint_positions": _as_float_list(info.get("joint_positions")),
        "joint_velocities": _as_float_list(info.get("joint_velocities")),
        "joint_torques": _as_float_list(info.get("joint_torques")),
        "last_action": _as_float_list(info.get("last_action")),
        "onnx_action": _as_float_list(info.get("onnx_action")),
        "residual_action": _as_float_list(info.get("residual_action")),
        "safety_layer_action": _as_float_list(info.get("safety_layer_action")),
        "final_action": _as_float_list(info.get("final_action")),
        "applied_action": _as_float_list(info.get("applied_action")),
        "contacts": {
            str(name): bool(value)
            for name, value in (info.get("contacts") or {}).items()
        },
    }


def _override_safety_layer_action(action: Any, override: Sequence[float]) -> list[float]:
    values = [float(value) for value in override]
    if len(values) != 4:
        raise ValueError(f"safety layer action override requires 4 values, got {len(values)}")
    return values


def _replay_states_from_bank(bank: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    events = bank.get("events") or []
    states: list[dict[str, Any]] = []
    for event in events:
        state = event.get("initial_state")
        if isinstance(state, dict):
            states.append(dict(state))
        if limit is not None and len(states) >= max(0, int(limit)):
            break
    return states


def _force_admittance_residual_action(
    *,
    info: dict[str, Any],
    action_names: Sequence[str],
    lateral_gain: float,
    forward_gain: float,
    force_scale_n: float,
    action_clip: float,
    hip_pattern: str,
) -> list[float]:
    names = [str(name) for name in action_names]
    residual = [0.0 for _ in names]
    if not bool(info.get("external_force_active", False)):
        return residual

    force = info.get("external_force_vector") or [0.0, 0.0, 0.0]
    if len(force) < 2:
        return residual

    scale_n = max(1e-6, float(force_scale_n))
    clip = max(0.0, float(action_clip))
    fx = float(force[0])
    fy = float(force[1])
    lateral_value = _clip_scalar(float(lateral_gain) * fy / scale_n, -clip, clip)
    forward_value = _clip_scalar(float(forward_gain) * fx / scale_n, -clip, clip)

    for index, name in enumerate(names):
        if name.endswith("_hip_joint"):
            sign = 1.0
            if hip_pattern == "left_right":
                leg = name.split("_", 1)[0]
                sign = 1.0 if leg in {"FL", "RL"} else -1.0
            residual[index] += sign * lateral_value
        elif name.endswith("_thigh_joint"):
            residual[index] += forward_value

    return [_clip_scalar(value, -clip, clip) for value in residual]


def _force_action_probe_delta(
    *,
    info: dict[str, Any],
    action_names: Sequence[str],
    gain: float,
    sign: float,
    force_scale_n: float,
    action_clip: float,
    joint_scope: str,
    window_start_s: float,
    window_end_s: float,
    hip_pattern: str,
) -> list[float]:
    names = [str(name) for name in action_names]
    delta = [0.0 for _ in names]
    if not bool(info.get("external_force_active", False)):
        return delta

    elapsed_s = info.get("external_force_current_window_elapsed_s")
    if elapsed_s is None:
        force_start_s = info.get("external_force_start_s")
        if force_start_s is not None:
            elapsed_s = float(info.get("t", 0.0)) - float(force_start_s)
    if elapsed_s is None:
        return delta
    elapsed = float(elapsed_s)
    if elapsed < float(window_start_s) or elapsed > float(window_end_s):
        return delta

    force = info.get("external_force_vector") or [0.0, 0.0, 0.0]
    if len(force) < 2:
        return delta

    scope = str(joint_scope)
    use_hips = scope in {"hip", "hip_thigh"}
    use_thighs = scope in {"thigh", "hip_thigh"}
    if not use_hips and not use_thighs:
        return delta

    scale_n = max(1e-6, float(force_scale_n))
    clip = max(0.0, float(action_clip))
    signed_gain = float(sign) * float(gain)
    fx = float(force[0])
    fy = float(force[1])
    lateral_value = _clip_scalar(signed_gain * fy / scale_n, -clip, clip)
    forward_value = _clip_scalar(signed_gain * fx / scale_n, -clip, clip)

    for index, name in enumerate(names):
        if use_hips and name.endswith("_hip_joint"):
            hip_sign = 1.0
            if hip_pattern == "left_right":
                leg = name.split("_", 1)[0]
                hip_sign = 1.0 if leg in {"FL", "RL"} else -1.0
            delta[index] += hip_sign * lateral_value
        elif use_thighs and name.endswith("_thigh_joint"):
            delta[index] += forward_value

    return [_clip_scalar(value, -clip, clip) for value in delta]


def _joint_pulse_probe_delta(
    *,
    info: dict[str, Any],
    action_names: Sequence[str],
    group: str,
    amplitude: float,
    sign: float,
    window_start_s: float,
    window_end_s: float,
) -> list[float]:
    names = [str(name) for name in action_names]
    delta = [0.0 for _ in names]
    if not bool(info.get("external_force_active", False)):
        return delta

    elapsed_s = info.get("external_force_current_window_elapsed_s")
    if elapsed_s is None:
        force_start_s = info.get("external_force_start_s")
        if force_start_s is not None:
            elapsed_s = float(info.get("t", 0.0)) - float(force_start_s)
    if elapsed_s is None:
        return delta
    elapsed = float(elapsed_s)
    if elapsed < float(window_start_s) or elapsed > float(window_end_s):
        return delta

    value = float(sign) * float(amplitude)
    contacts = {
        str(name): bool(active)
        for name, active in (info.get("contacts") or {}).items()
    }
    for index, name in enumerate(names):
        leg = name.split("_", 1)[0]
        if group == "hip_left_right" and name.endswith("_hip_joint"):
            delta[index] = value if leg in {"FL", "RL"} else -value
        elif group == "front_rear_thigh" and name.endswith("_thigh_joint"):
            delta[index] = value if leg in {"FL", "FR"} else -value
        elif group == "all_hip" and name.endswith("_hip_joint"):
            delta[index] = value
        elif group == "stance_only" and name.endswith("_hip_joint") and contacts.get(leg, False):
            delta[index] = value
    return delta


def _print_summary(report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    print(f"task: {report['task']}")
    print(f"checkpoint: {report['checkpoint']}")
    print(f"vecnormalize: {report['vecnormalize'] or 'none'}")
    print(f"velocity_command_frame: {report['config']['velocity_command_frame']}")
    print(f"episodes: {aggregate['episodes']}")
    print(f"success_rate: {aggregate['success_rate']:.2f}")
    print(f"fall_rate: {aggregate['fall_rate']:.2f}")
    print(f"straightness_success_rate: {aggregate['straightness_success_rate']:.2f}")
    print(f"base_z_min: {_fmt(aggregate['base_z_min'])} m")
    print(f"min_base_z_during_force: {_fmt(aggregate['min_base_z_during_force'])} m")
    print(f"min_base_z_after_force: {_fmt(aggregate['min_base_z_after_force'])} m")
    print(f"max_downward_velocity_during_force: {_fmt(aggregate['max_downward_velocity_during_force'])} m/s")
    print(f"max_orientation_error_during_force: {_fmt(aggregate['max_orientation_error_during_force'])}")
    print(f"min_contact_count_during_force: {_fmt(aggregate['min_contact_count_during_force'])}")
    print(f"mean_forward_velocity: {aggregate['mean_forward_velocity']:.3f} m/s")
    print(f"mean_lateral_velocity: {aggregate['mean_lateral_velocity']:.3f} m/s")
    print(f"mean_abs_lateral_velocity: {aggregate['mean_abs_lateral_velocity']:.3f} m/s")
    print(f"mean_yaw_rate: {aggregate['mean_yaw_rate']:.3f} rad/s")
    print(f"mean_abs_yaw_rate: {aggregate['mean_abs_yaw_rate']:.3f} rad/s")
    print(f"mean_final_base_yaw: {aggregate['mean_final_base_yaw']:.3f} rad")
    print(f"mean_abs_base_yaw: {aggregate['mean_abs_base_yaw']:.3f} rad")
    print(f"mean_abs_final_lateral_displacement: {aggregate['mean_abs_final_lateral_displacement']:.3f} m")
    print(f"mean_straightness_ratio: {aggregate['mean_straightness_ratio']:.3f}")
    print(f"mean_velocity_error_xy: {aggregate['mean_velocity_error_xy']:.3f} m/s")
    print(f"mean_yaw_rate_error: {aggregate['mean_yaw_rate_error']:.3f} rad/s")
    print(f"external_force_recovery_success_rate: {aggregate['external_force_recovery_success_rate']:.2f}")
    print(f"force_recovery_survival_rate: {aggregate.get('force_recovery_survival_rate', 0.0):.2f}")
    print(f"force_recovery_posture_success_rate: {aggregate.get('force_recovery_posture_success_rate', 0.0):.2f}")
    print(f"force_recovery_locomotion_success_rate: {aggregate.get('force_recovery_locomotion_success_rate', 0.0):.2f}")
    print(f"force_recovery_min_base_z_after_force: {_fmt(aggregate.get('force_recovery_min_base_z_after_force'))} m")
    print(
        "force_recovery_max_orientation_error_during_force: "
        f"{_fmt(aggregate.get('force_recovery_max_orientation_error_during_force'))}"
    )
    print(f"mean_external_force_event_count: {aggregate['mean_external_force_event_count']:.2f}")
    print(
        "mean_external_force_event_compliance_rate: "
        f"{_fmt(aggregate.get('mean_external_force_event_compliance_rate'))}"
    )
    print(
        "episode_all_events_compliance_rate: "
        f"{_fmt(aggregate.get('episode_all_events_compliance_rate'))}"
    )
    print(
        "max_external_force_event_onset_max_excess_n: "
        f"{_fmt(aggregate.get('max_external_force_event_onset_max_excess_n'))} N"
    )
    print(
        "max_external_force_event_adaptation_max_excess_n: "
        f"{_fmt(aggregate.get('max_external_force_event_adaptation_max_excess_n'))} N"
    )
    print(
        "mean_external_force_event_adaptation_compliance_rate: "
        f"{_fmt(aggregate.get('mean_external_force_event_adaptation_compliance_rate'))}"
    )
    print(
        "max_external_force_event_excess_impulse_ns: "
        f"{_fmt(aggregate.get('max_external_force_event_excess_impulse_ns'))} N*s"
    )
    print(f"mean_external_force_max_magnitude: {aggregate['mean_external_force_max_magnitude']:.2f} N")
    print(f"mean_external_force_max_torque_magnitude: {aggregate['mean_external_force_max_torque_magnitude']:.2f} N*m")
    print(f"mean_residual_action_norm: {aggregate['mean_residual_action_norm']:.3f}")
    print(f"mean_safety_layer_action_norm: {aggregate['mean_safety_layer_action_norm']:.3f}")
    print(f"mean_final_minus_onnx_action_norm: {aggregate['mean_final_minus_onnx_action_norm']:.3f}")
    print(
        "mean_force_active_residual_action_norm: "
        f"{_fmt(aggregate['mean_force_active_residual_action_norm'])}"
    )
    print(
        "max_force_active_residual_action_norm: "
        f"{_fmt(aggregate['max_force_active_residual_action_norm'])}"
    )
    print(
        "mean_force_active_safety_layer_action_norm: "
        f"{_fmt(aggregate['mean_force_active_safety_layer_action_norm'])}"
    )
    print(
        "max_force_active_safety_layer_action_norm: "
        f"{_fmt(aggregate['max_force_active_safety_layer_action_norm'])}"
    )
    print(
        "mean_force_onset_residual_action_norm: "
        f"{_fmt(aggregate['mean_force_onset_residual_action_norm'])}"
    )
    print(
        "mean_force_onset_safety_layer_action_norm: "
        f"{_fmt(aggregate['mean_force_onset_safety_layer_action_norm'])}"
    )
    print(
        "max_force_onset_external_force_excess_n: "
        f"{_fmt(aggregate['max_force_onset_external_force_excess_n'])} N"
    )
    print(f"post_force_mean_forward_velocity: {_fmt(aggregate['post_force_mean_forward_velocity'])} m/s")
    print(f"post_force_mean_abs_lateral_velocity: {_fmt(aggregate['post_force_mean_abs_lateral_velocity'])} m/s")
    print(f"post_force_mean_abs_yaw_rate: {_fmt(aggregate['post_force_mean_abs_yaw_rate'])} rad/s")
    print(f"mean_orientation_error: {aggregate['mean_orientation_error']:.3f}")
    print(f"mean_contact_imbalance: {aggregate['mean_contact_imbalance']:.3f}")
    print(f"mean_left_right_contact_delta: {aggregate['mean_left_right_contact_delta']:.3f}")
    print(f"mean_diagonal_contact_delta: {aggregate['mean_diagonal_contact_delta']:.3f}")
    print(f"mean_left_right_action_energy_delta: {aggregate['mean_left_right_action_energy_delta']:.5f}")
    print(f"mean_action_energy_imbalance_abs: {aggregate['mean_action_energy_imbalance_abs']:.5f}")
    print("reward_terms:")
    for name, value in aggregate["reward_terms_mean"].items():
        print(f"  {name}: {value:.5f}")


def _mean(values: list[float | int | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / max(len(clean), 1)


def _mean_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _min_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def _max_or_none(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _clip_scalar(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _as_float3(values: Any) -> list[float]:
    if values is None:
        return [0.0, 0.0, 0.0]
    try:
        clean = [float(value) for value in values]
    except TypeError:
        return [0.0, 0.0, 0.0]
    return (clean + [0.0, 0.0, 0.0])[:3]


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    try:
        return [float(value) for value in values]
    except TypeError:
        return []


def _vector_norm(values: Any) -> float | None:
    if values is None:
        return None
    try:
        clean = [float(value) for value in values]
    except TypeError:
        return None
    if not clean:
        return None
    return math.sqrt(sum(value * value for value in clean))


def _vector_delta_norm(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except TypeError:
        return None
    if not left_values or len(left_values) != len(right_values):
        return None
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left_values, right_values)))


def _path_length_xy(positions: list[tuple[float, float, float]]) -> float:
    if len(positions) < 2:
        return 0.0
    length = 0.0
    for previous, current in zip(positions, positions[1:]):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        length += math.sqrt(dx * dx + dy * dy)
    return length


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _contact_imbalance(contact_ratio: dict[str, float]) -> float:
    values = list(contact_ratio.values())
    return max(values) - min(values) if values else 0.0


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
