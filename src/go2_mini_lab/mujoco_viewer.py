from __future__ import annotations

import argparse
import os
import platform
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cli import (
    add_force_response_router_args,
    add_force_safety_trigger_args,
    add_fullbody_reference_args,
    add_observation_mode_arg,
    add_policy_action_mode_args,
    add_policy_leg_order_arg,
    add_velocity_reward_args,
    parse_reward_overrides,
    resolve_policy_leg_order,
)
from .controller import PDConfig
from .gym_env import Go2StandBalanceGymEnv
from .onnx_rollout import DEFAULT_REFERENCE_POLICY, _load_actor_obs_normalizer
from .rl_env import RL_TASKS, _compose_policy_action, make_task_config
from .sb3_tools import load_vecnormalize_if_available, make_vec_env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open a live MuJoCo viewer for the Go2 model, ONNX teacher, or PPO checkpoint."
    )
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--onnx-policy",
        type=Path,
        help="Run an ONNX teacher policy in the live MuJoCo viewer.",
    )
    policy_group.add_argument(
        "--ppo-checkpoint",
        type=Path,
        help="Run a Stable-Baselines3 PPO checkpoint in the live MuJoCo viewer.",
    )
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
        help="Path to the Go2 MuJoCo MJCF scene.",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. 0 means run until the viewer is closed.")
    parser.add_argument("--control-dt", type=float, default=0.02)
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
    add_observation_mode_arg(parser)
    add_policy_action_mode_args(
        parser,
        onnx_policy_arg="--residual-onnx-policy",
        onnx_policy_dest="residual_onnx_policy",
        onnx_normalizer_arg="--residual-onnx-normalizer-checkpoint",
        onnx_normalizer_dest="residual_onnx_normalizer_checkpoint",
    )
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
        help="PD impedance modulation mode for viewer force-safety probes.",
    )
    parser.add_argument(
        "--force-impedance-joint-scope",
        choices=("all", "hip", "thigh", "calf", "hip_calf", "stance_hip_calf"),
        default="all",
        help="Joint family affected by force impedance modulation.",
    )
    parser.add_argument("--force-impedance-kp-scale", type=float, default=1.0)
    parser.add_argument("--force-impedance-kd-scale", type=float, default=1.0)
    parser.add_argument("--force-impedance-delay-s", type=float, default=0.0)
    parser.add_argument("--force-impedance-hold-s", type=float, default=0.15)
    parser.add_argument("--force-impedance-recovery-s", type=float, default=0.10)
    parser.add_argument("--force-impedance-tail-kp-scale", type=float, default=1.0)
    parser.add_argument("--force-impedance-tail-kd-scale", type=float, default=1.0)
    parser.add_argument(
        "--force-reference-governor-mode",
        choices=("off", "onset", "active", "two_phase"),
        default="off",
        help="Event-triggered reference governor mode for viewer force-safety probes.",
    )
    parser.add_argument("--force-reference-governor-admittance", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-damping", type=float, default=5.0)
    parser.add_argument("--force-reference-governor-offset-clip", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-velocity-clip", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-delay-s", type=float, default=0.10)
    parser.add_argument("--force-reference-governor-hold-s", type=float, default=0.20)
    parser.add_argument("--force-reference-governor-recovery-s", type=float, default=0.10)
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
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PPO actions.")
    parser.add_argument(
        "--safety-layer-action-override",
        nargs=4,
        type=float,
        metavar=("KP_YIELD", "KD_BOOST", "GOV_GAIN", "GOV_CLIP"),
        help=(
            "Use a fixed 4D onnx_safety_layer action instead of loading PPO. "
            "Use with --policy-action-mode onnx_safety_layer and --residual-onnx-policy."
        ),
    )
    parser.add_argument("--vecnormalize", type=Path, help="VecNormalize stats for PPO. Defaults to <checkpoint>_vecnormalize.pkl.")
    parser.add_argument("--no-vecnormalize", action="store_true")
    parser.add_argument(
        "--normalizer-checkpoint",
        type=Path,
        help="Optional RSL-RL actor_obs_normalizer checkpoint for ONNX policies.",
    )
    parser.add_argument("--no-normalizer", action="store_true", help="Disable ONNX observation normalizer.")
    parser.add_argument("--realtime-rate", type=float, default=1.0, help="1.0 is real time; 0 disables sleeping.")
    parser.add_argument("--stop-on-done", action="store_true", help="Close when an episode terminates/truncates instead of resetting.")
    parser.add_argument(
        "--mouse-perturb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MuJoCo native mouse perturbation forces in the live viewer.",
    )
    parser.add_argument(
        "--max-mouse-force",
        type=float,
        default=30.0,
        help="Clip native mouse perturbation force magnitude in Newtons. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--external-force-observation",
        action="store_true",
        help="Append normalized mouse perturbation phase/force/torque signals to PPO observations.",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("MuJoCo viewer is missing. Install with: pip install -e '.[sim]'") from exc

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
        episode_length_s=args.duration if args.duration > 0 else 20.0,
        target_forward_velocity=args.target_forward_velocity,
        target_lateral_velocity=args.target_lateral_velocity,
        target_yaw_rate=args.target_yaw_rate,
        action_scale=args.action_scale,
        action_smoothing=args.action_smoothing,
        reward_scale=args.reward_scale,
        reset_settle_s=args.reset_settle,
        randomize_commands=randomize_commands,
        push_time_s=args.push_time,
        push_linear_velocity=push_linear_velocity,
        push_angular_velocity=push_angular_velocity,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
        velocity_reward_overrides=velocity_reward_overrides,
        observation_mode=args.observation_mode,
        policy_action_mode=args.policy_action_mode,
        onnx_policy_path=args.residual_onnx_policy,
        onnx_normalizer_checkpoint=args.residual_onnx_normalizer_checkpoint,
        residual_action_scale=args.residual_action_scale,
        fullbody_reference_mode=args.fullbody_reference_mode,
        nominal_reference_dataset=args.nominal_reference_dataset,
        gait_frequency_hz=args.gait_frequency,
        gait_step_length=args.gait_step_length,
        gait_swing_height=args.gait_swing_height,
        gait_joint_thigh_amplitude=args.gait_thigh_amplitude,
        gait_joint_calf_amplitude=args.gait_calf_amplitude,
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
        force_safety_detector_linear_acceleration_threshold=args.force_safety_detector_linear_acceleration_threshold,
        force_safety_detector_angular_acceleration_threshold=args.force_safety_detector_angular_acceleration_threshold,
        force_safety_detector_joint_error_threshold=args.force_safety_detector_joint_error_threshold,
        force_safety_detector_joint_velocity_threshold=args.force_safety_detector_joint_velocity_threshold,
        force_safety_detector_contact_loss=args.force_safety_detector_contact_loss,
        force_safety_detector_enable_after_s=args.force_safety_detector_enable_after_s,
        force_safety_detector_hold_s=args.force_safety_detector_hold_s,
        force_safety_detector_recovery_s=args.force_safety_detector_recovery_s,
        external_force_max_n=max(0.0, float(args.max_mouse_force)),
        external_force_torque_max_nm=max(0.0, float(args.max_mouse_force)),
        include_external_force_observation=args.external_force_observation,
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
    policy = _load_policy(args, env_config=env_config)
    obs, _ = env.reset()

    print(f"model: {args.model}")
    print(f"task: {args.task}")
    print(f"policy: {policy.label}")
    print(f"target_forward_velocity: {env_config.target_forward_velocity}")
    print(f"velocity_pose_profile: {env_config.velocity_pose_profile}")
    print(f"policy_leg_order: {args.policy_leg_order}")
    print(f"observation_mode: {env_config.observation_mode}")
    print(f"policy_action_mode: {env_config.policy_action_mode}")
    if env_config.onnx_policy_path:
        print(f"residual_onnx_policy: {env_config.onnx_policy_path}")
    print(f"residual_action_scale: {env_config.residual_action_scale}")
    print(f"velocity_command_frame: {env_config.velocity_command_frame}")
    print(f"fullbody_reference_mode: {env_config.fullbody_reference_mode}")
    if env_config.nominal_reference_dataset:
        print(f"nominal_reference_dataset: {env_config.nominal_reference_dataset}")
    print(f"action_scale: {env_config.action_scale}")
    print(f"action_smoothing: {env_config.action_smoothing}")
    print(f"pd_kp: {env_config.pd.kp}")
    print(f"pd_kd: {env_config.pd.kd}")
    print(f"torque_limit: {env_config.pd.torque_limit}")
    if args.mouse_perturb:
        print(f"mouse_perturb: enabled, max_force={args.max_mouse_force} N")
        print("Mouse: double-click a Go2 body, then Ctrl+right-drag to apply force; Ctrl+left-drag applies torque.")
        print("MuJoCo draws the perturbation arrow when the force is active. Press F1 in the viewer for controls.")
    else:
        print("mouse_perturb: disabled")
    print("Close the MuJoCo viewer window to stop.")

    start_wall = time.monotonic()
    try:
        with mujoco.viewer.launch_passive(env.env.model, env.env.data) as viewer:
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 1
            _update_viewer_text(
                viewer,
                env=env.env,
                selected_body_name="none",
                force_norm=0.0,
                max_force=args.max_mouse_force if args.mouse_perturb else 0.0,
            )
            while viewer.is_running():
                step_start = time.monotonic()
                action = policy.act(obs)
                obs, _reward, terminated, truncated, info = _step_with_viewer_perturb(
                    env=env.env,
                    action=action,
                    mujoco_module=mujoco,
                    perturb=viewer.perturb,
                    enable_mouse_perturb=bool(args.mouse_perturb),
                    max_mouse_force=float(args.max_mouse_force),
                )
                selected_body_name = _selected_body_name(env.env.model, viewer.perturb.select)
                force_norm = _selected_force_norm(env.env.data, viewer.perturb.select)
                _update_viewer_text(
                    viewer,
                    env=env.env,
                    selected_body_name=selected_body_name,
                    force_norm=force_norm,
                    max_force=args.max_mouse_force if args.mouse_perturb else 0.0,
                )
                viewer.sync()
                if terminated or truncated:
                    print(
                        "episode done: "
                        f"terminated={str(terminated).lower()} "
                        f"truncated={str(truncated).lower()} "
                        f"t={env.env.elapsed_s:.3f}s "
                        f"base_z={float(info.get('base_z', 0.0)):.3f}"
                    )
                    if args.stop_on_done:
                        break
                    obs, _ = env.reset()
                if args.duration > 0 and time.monotonic() - start_wall >= args.duration:
                    break
                _sleep_for_realtime(step_start, env_config.control_dt, args.realtime_rate)
    except RuntimeError as exc:
        message = str(exc)
        if platform.system() == "Darwin" and "mjpython" in message:
            command = ".venv/bin/mjpython -m go2_mini_lab.mujoco_viewer"
            if sys.argv[1:]:
                command = f"{command} {shlex.join(sys.argv[1:])}"
            raise RuntimeError(
                "MuJoCo's native viewer on macOS must be launched with mjpython.\n"
                f"Run this command instead:\n  {command}\n\n"
                "If mjpython reports that libpython3.12.dylib is missing in this uv venv, run:\n"
                "  ln -s /Users/loganluo/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/libpython3.12.dylib "
                ".venv/lib/libpython3.12.dylib"
            ) from exc
        raise
    return 0


class _Policy:
    def __init__(self, *, label: str, actuator: Any) -> None:
        self.label = label
        self._actuator = actuator

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._actuator(obs)


def _load_policy(args: argparse.Namespace, *, env_config: Any) -> _Policy:
    if args.safety_layer_action_override is not None:
        action = np.asarray(args.safety_layer_action_override, dtype=np.float32)

        def _fixed_safety_action(obs: np.ndarray) -> np.ndarray:
            return action.copy()

        return _Policy(label=f"fixed safety-layer action:{action.tolist()}", actuator=_fixed_safety_action)
    if args.onnx_policy is not None:
        return _load_onnx_policy(args)
    if args.ppo_checkpoint is not None:
        return _load_ppo_policy(args, env_config=env_config)

    def _zero_action(obs: np.ndarray) -> np.ndarray:
        return np.zeros(_default_action_size_for_policy_mode(env_config.policy_action_mode), dtype=np.float32)

    return _Policy(label="zero-action default pose", actuator=_zero_action)


def _default_action_size_for_policy_mode(policy_action_mode: str) -> int:
    if str(policy_action_mode) == "onnx_safety_layer":
        return 4
    return 12


def _load_onnx_policy(args: argparse.Namespace) -> _Policy:
    policy_path = args.onnx_policy or DEFAULT_REFERENCE_POLICY
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is missing. Install with: pip install -e '.[reference]'") from exc
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
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

    return _Policy(label=f"onnx:{policy_path}", actuator=_act)


def _load_ppo_policy(args: argparse.Namespace, *, env_config: Any) -> _Policy:
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


def _optional_vec(
    x: float | None,
    y: float | None,
    z: float | None,
) -> tuple[float, float, float] | None:
    if x is None and y is None and z is None:
        return None
    return (float(x or 0.0), float(y or 0.0), float(z or 0.0))


def _step_with_viewer_perturb(
    *,
    env: Any,
    action: np.ndarray,
    mujoco_module: Any,
    perturb: Any,
    enable_mouse_perturb: bool,
    max_mouse_force: float,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    action_array = env.np.asarray(action, dtype=env.np.float64)
    if action_array.shape != (env.action_size,):
        raise ValueError(f"action must have shape ({env.action_size},), got {action_array.shape}")
    previous_action = env._last_action.copy()
    previous_residual_action = env._last_residual_action.copy()
    previous_safety_layer_action = env._last_safety_layer_action.copy()
    previous_joint_vel = env._joint_vel_array()
    env._update_force_safety_detector()
    onnx_action = (
        env._onnx_expert_action()
        if env.config.policy_action_mode in ("onnx_residual", "onnx_safety_layer")
        else None
    )
    safety_layer_action = env.np.zeros_like(previous_safety_layer_action)
    compose_action = action_array
    if env.config.policy_action_mode == "onnx_safety_layer":
        safety_layer_action = action_array.copy()
        compose_action = env.np.zeros(env.joint_action_size, dtype=env.np.float64)
    final_action, base_action, residual_action = _compose_policy_action(
        compose_action,
        mode=env.config.policy_action_mode,
        onnx_action=onnx_action,
        residual_scale=env.config.residual_action_scale,
        np_module=env.np,
    )
    smoothing = max(0.0, min(0.98, float(env.config.action_smoothing)))
    filtered_action = smoothing * previous_action + (1.0 - smoothing) * final_action
    env._last_safety_layer_action = env.np.asarray(safety_layer_action, dtype=env.np.float64).copy()
    env._last_safety_layer_action_rate = env._last_safety_layer_action - previous_safety_layer_action
    env._apply_action(filtered_action)
    env._maybe_apply_push()
    env._last_onnx_action = env.np.asarray(base_action, dtype=env.np.float64).copy()
    env._last_residual_action = env.np.asarray(residual_action, dtype=env.np.float64).copy()
    env._last_policy_action = env.np.asarray(final_action, dtype=env.np.float64).copy()
    env._last_residual_action_rate = env._last_residual_action - previous_residual_action
    env._last_action = filtered_action
    env._external_force_vector[:] = 0.0
    env._external_force_torque_vector[:] = 0.0
    env._external_force_applied_this_step = False

    for _ in range(env._control_steps):
        env.data.xfrc_applied[:] = 0.0
        if enable_mouse_perturb:
            mujoco_module.mjv_applyPerturbForce(env.model, env.data, perturb)
            _clip_external_forces(env.data.xfrc_applied, max_force=max_mouse_force)
            active_body_id = _active_perturb_body_id(env.data.xfrc_applied, int(perturb.select))
            force = env.np.sum(env.data.xfrc_applied[:, :3], axis=0)
            torque = env.np.sum(env.data.xfrc_applied[:, 3:], axis=0)
            env._external_force_vector[:] = force
            env._external_force_torque_vector[:] = torque
            env._external_force_applied_this_step = bool(
                env.np.linalg.norm(force) > 1e-9 or env.np.linalg.norm(torque) > 1e-9
            )
            if env._external_force_applied_this_step and active_body_id >= 0:
                env._external_force_body_id = int(active_body_id)
                env._external_force_body_name = _selected_body_name(env.model, active_body_id)
                if (
                    not np.isfinite(float(env._external_force_start_s))
                    or float(env._external_force_end_s) < float(env._elapsed_s) - float(env.config.control_dt)
                ):
                    env._external_force_start_s = float(env._elapsed_s)
                env._external_force_end_s = float(env._elapsed_s) + float(env.config.control_dt)
        mujoco_module.mj_step(env.model, env.data)
    env._elapsed_s += env._control_steps * env.model.opt.timestep
    env._update_force_reference_governor()

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
    return env.observation(), float(reward), bool(terminated), bool(truncated), env._info(
        reward_terms=reward_terms,
        reward_raw_terms=raw_reward_terms,
    )


def _clip_external_forces(xfrc_applied: np.ndarray, *, max_force: float) -> None:
    if max_force <= 0:
        return
    for index in range(xfrc_applied.shape[0]):
        force = xfrc_applied[index, :3]
        force_norm = float(np.linalg.norm(force))
        if force_norm > max_force:
            xfrc_applied[index, :3] = force * (max_force / force_norm)


def _active_perturb_body_id(xfrc_applied: np.ndarray, selected_body_id: int) -> int:
    if 0 <= int(selected_body_id) < xfrc_applied.shape[0]:
        wrench = xfrc_applied[int(selected_body_id), :6]
        if float(np.linalg.norm(wrench)) > 1e-9:
            return int(selected_body_id)
    for index in range(xfrc_applied.shape[0]):
        if float(np.linalg.norm(xfrc_applied[index, :6])) > 1e-9:
            return int(index)
    return -1


def _selected_body_name(model: Any, body_id: int) -> str:
    if body_id < 0 or body_id >= model.nbody:
        return "none"
    try:
        import mujoco
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"body:{body_id}"
    except Exception:
        return f"body:{body_id}"


def _selected_force_norm(data: Any, body_id: int) -> float:
    if body_id < 0 or body_id >= data.xfrc_applied.shape[0]:
        return 0.0
    return float(np.linalg.norm(data.xfrc_applied[int(body_id), :3]))


def _update_viewer_text(
    viewer: Any,
    *,
    env: Any,
    selected_body_name: str,
    force_norm: float,
    max_force: float,
) -> None:
    limit_text = "unlimited" if max_force <= 0 else f"{max_force:.1f} N max"
    viewer.set_texts((
        None,
        None,
        "Go2 live perturb",
        (
            f"t={env.elapsed_s:.2f}s  base_z={float(env.data.qpos[2]):.3f}m  "
            f"selected={selected_body_name}  force={force_norm:.1f}N ({limit_text})\n"
            "Double-click body; Ctrl+right-drag force; Ctrl+left-drag torque; F1 help"
        ),
    ))


def _sleep_for_realtime(step_start: float, control_dt: float, realtime_rate: float) -> None:
    if realtime_rate <= 0:
        return
    remaining = control_dt / realtime_rate - (time.monotonic() - step_start)
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
