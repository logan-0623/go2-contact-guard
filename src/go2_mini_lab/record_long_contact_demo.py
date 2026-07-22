from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
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
from .onnx_rollout import DEFAULT_REFERENCE_POLICY
from .rl_env import RL_TASKS, make_task_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a GitHub-ready GIF/MP4 of Go2 long-contact force recovery."
    )
    parser.add_argument("--task", choices=RL_TASKS, default="velocity_flat")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml"),
        help="Path to the Go2 MuJoCo MJCF scene.",
    )
    parser.add_argument("--output-gif", type=Path, default=Path("docs/assets/go2_long_contact_foot_guard.gif"))
    parser.add_argument("--output-mp4", type=Path, help="Optional MP4 output path. Defaults to output-gif with .mp4.")
    parser.add_argument("--summary-output", type=Path, help="Optional JSON summary path. Defaults to output-gif with .json.")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--render-fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--demo-title", default="Go2 semantic contact guard")
    parser.add_argument(
        "--demo-subtitle",
        default="Frozen ONNX gait + event-triggered impedance / governor guard",
    )
    parser.add_argument("--target-forward-velocity", type=float, default=0.40)
    parser.add_argument("--target-lateral-velocity", type=float, default=0.0)
    parser.add_argument("--target-yaw-rate", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--action-smoothing", type=float, default=0.15)
    parser.add_argument("--reward-scale", type=float)
    parser.add_argument("--reset-settle", type=float, default=0.0)
    parser.add_argument("--velocity-pose-profile", choices=("mjlab", "official"), default="official")
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
    add_velocity_reward_args(parser)
    parser.add_argument(
        "--force-impedance-mode",
        choices=("off", "onset", "active", "two_phase"),
        default="off",
        help="PD impedance modulation mode for the long-contact demo.",
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
        help="Event-triggered reference governor mode for the long-contact demo.",
    )
    parser.add_argument("--force-reference-governor-admittance", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-damping", type=float, default=5.0)
    parser.add_argument("--force-reference-governor-offset-clip", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-velocity-clip", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-delay-s", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-hold-s", type=float, default=0.20)
    parser.add_argument("--force-reference-governor-recovery-s", type=float, default=0.10)
    parser.add_argument("--force-reference-governor-tail-admittance-scale", type=float, default=0.0)
    parser.add_argument("--force-reference-governor-tail-offset-clip-scale", type=float, default=1.0)
    parser.add_argument("--force-reference-governor-tail-velocity-clip-scale", type=float, default=1.0)
    parser.add_argument("--safety-layer-action-override", nargs=4, type=float, default=(0.0, 0.0, 0.0, 0.0))
    parser.add_argument("--pd-kp", type=float, default=20.0)
    parser.add_argument("--pd-kd", type=float, default=1.0)
    parser.add_argument("--torque-limit", type=float, default=23.5)
    parser.add_argument("--force-body", default="FR_foot")
    parser.add_argument("--force-n", type=float, default=22.0)
    parser.add_argument("--force-direction-angle", type=float, default=math.pi / 2.0)
    parser.add_argument("--contact-start", type=float, default=2.0)
    parser.add_argument("--contact-duration", type=float, default=3.0)
    parser.add_argument("--force-transition", type=float, default=0.20)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--camera-distance", type=float, default=3.1)
    parser.add_argument("--keep-frames", action="store_true", help="Keep rendered PNG frames next to output-gif.")
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/go2_mini_lab_matplotlib")

    output_gif = args.output_gif
    output_mp4 = args.output_mp4 or output_gif.with_suffix(".mp4")
    summary_output = args.summary_output or output_gif.with_suffix(".json")
    if args.residual_onnx_policy is None and str(args.policy_action_mode) == "onnx_safety_layer":
        args.residual_onnx_policy = str(DEFAULT_REFERENCE_POLICY)

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
        randomize_commands=False,
        velocity_pose_profile=args.velocity_pose_profile,
        policy_leg_order=resolve_policy_leg_order(args.policy_leg_order),
        velocity_reward_profile=args.velocity_reward_profile,
        velocity_command_frame=args.velocity_command_frame,
        velocity_reward_overrides=parse_reward_overrides(args.reward_term),
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
        external_force_mode="constant",
        external_force_probability=1.0,
        external_force_body_names=(str(args.force_body),),
        external_force_active_body_count=1,
        external_force_event_count_range=(1, 1),
        external_force_rest_s_range=(1.0, 1.0),
        external_force_start_s_range=(float(args.contact_start), float(args.contact_start)),
        external_force_duration_s_range=(float(args.contact_duration), float(args.contact_duration)),
        external_force_min_n=float(args.force_n),
        external_force_max_n=float(args.force_n),
        external_force_z_fraction=0.0,
        external_force_direction_angle_rad=float(args.force_direction_angle),
        external_force_torque_max_nm=0.0,
        external_force_transition_s=float(args.force_transition),
        external_force_net_force_limit_n=float(args.force_n),
        external_force_net_torque_limit_nm=0.0,
        include_external_force_observation=False,
    )
    env_config = replace(env_config, pd=PDConfig(kp=args.pd_kp, kd=args.pd_kd, torque_limit=args.torque_limit))

    env = Go2StandBalanceGymEnv(model_path=args.model, config=env_config)
    obs, _ = env.reset()
    action = np.asarray(args.safety_layer_action_override, dtype=np.float64)
    if action.shape != (env.env.action_size,):
        raise ValueError(f"safety-layer action must have shape ({env.env.action_size},), got {action.shape}")

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    temp_parent = output_gif.parent if args.keep_frames else None
    with tempfile.TemporaryDirectory(prefix="go2_long_contact_frames_", dir=temp_parent) as temp_dir_name:
        frame_dir = Path(temp_dir_name)
        frames, last_info, terminated, truncated = _render_rollout(
            env=env,
            action=action,
            frame_dir=frame_dir,
            render_fps=float(args.render_fps),
            width=int(args.width),
            height=int(args.height),
            camera_azimuth=float(args.camera_azimuth),
            camera_elevation=float(args.camera_elevation),
            camera_distance=float(args.camera_distance),
            contact_start=float(args.contact_start),
            contact_duration=float(args.contact_duration),
            force_n=float(args.force_n),
            force_body=str(args.force_body),
            demo_title=str(args.demo_title),
            demo_subtitle=str(args.demo_subtitle),
        )
        _encode_media(frame_dir, output_gif=output_gif, output_mp4=output_mp4, fps=float(args.render_fps))
        if args.keep_frames:
            kept = output_gif.with_suffix("")
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(frame_dir, kept)

    summary = {
        "output_gif": str(output_gif),
        "output_mp4": str(output_mp4),
        "frames": frames,
        "render_fps": float(args.render_fps),
        "duration_s": float(args.duration),
        "contact_start_s": float(args.contact_start),
        "contact_duration_s": float(args.contact_duration),
        "force_body": str(args.force_body),
        "force_n": float(args.force_n),
        "force_direction": [float(v) for v in _force_vector_xy(float(args.force_n), float(args.force_direction_angle))],
        "policy_action_mode": env_config.policy_action_mode,
        "force_safety_trigger_source": env_config.force_safety_trigger_source,
        "force_response_router_mode": env_config.force_response_router_mode,
        "force_response_profile": env_config.force_response_profile,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_base_z": float((last_info or {}).get("base_z", 0.0)),
        "final_forward_velocity": float(((last_info or {}).get("base_linear_velocity") or [0.0])[0]),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_gif}")
    print(f"wrote {output_mp4}")
    print(f"wrote {summary_output}")
    return 0


def _render_rollout(
    *,
    env: Go2StandBalanceGymEnv,
    action: np.ndarray,
    frame_dir: Path,
    render_fps: float,
    width: int,
    height: int,
    camera_azimuth: float,
    camera_elevation: float,
    camera_distance: float,
    contact_start: float,
    contact_duration: float,
    force_n: float,
    force_body: str,
    demo_title: str,
    demo_subtitle: str,
) -> tuple[int, dict[str, Any], bool, bool]:
    try:
        import mujoco
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Rendering requires mujoco and Pillow. Install the sim/demo dependencies first.") from exc

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE.value] = 1
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ.value] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = float(camera_azimuth)
    camera.elevation = float(camera_elevation)
    camera.distance = float(camera_distance)

    frame_dir.mkdir(parents=True, exist_ok=True)
    next_render_t = 0.0
    frame_index = 0
    terminated = False
    truncated = False
    last_info: dict[str, Any] = {}
    render_dt = 1.0 / max(1e-6, float(render_fps))

    renderer = mujoco.Renderer(env.env.model, height=height, width=width)
    try:
        while not terminated and not truncated:
            while env.env.elapsed_s + 1e-12 >= next_render_t:
                _write_frame(
                    renderer=renderer,
                    data=env.env.data,
                    camera=camera,
                    scene_option=scene_option,
                    frame_path=frame_dir / f"frame_{frame_index:05d}.png",
                    t_s=next_render_t,
                    total_duration=float(env.env.config.episode_length_s),
                    contact_start=contact_start,
                    contact_duration=contact_duration,
                    force_n=force_n,
                    force_body=force_body,
                    demo_title=demo_title,
                    demo_subtitle=demo_subtitle,
                    Image=Image,
                    ImageDraw=ImageDraw,
                )
                frame_index += 1
                next_render_t += render_dt
            _obs, _reward, terminated, truncated, last_info = env.step(action)
    finally:
        renderer.close()
    return frame_index, last_info, bool(terminated), bool(truncated)


def _write_frame(
    *,
    renderer: Any,
    data: Any,
    camera: Any,
    scene_option: Any,
    frame_path: Path,
    t_s: float,
    total_duration: float,
    contact_start: float,
    contact_duration: float,
    force_n: float,
    force_body: str,
    demo_title: str,
    demo_subtitle: str,
    Image: Any,
    ImageDraw: Any,
) -> None:
    base_position = np.asarray(data.qpos[:3], dtype=float)
    camera.lookat[:] = base_position + np.asarray([0.15, 0.0, 0.12])
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    pixels = renderer.render()
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image, "RGBA")
    active = contact_start <= t_s <= contact_start + contact_duration
    status = "LONG CONTACT ACTIVE" if active else ("RECOVERY" if t_s > contact_start + contact_duration else "NOMINAL WALK")
    accent = (230, 70, 50, 230) if active else ((80, 160, 80, 230) if "RECOVERY" in status else (55, 105, 180, 230))
    draw.rectangle((18, 18, 430, 116), fill=(0, 0, 0, 145))
    draw.rectangle((18, 18, 430, 24), fill=accent)
    draw.text((30, 34), f"{demo_title}  |  t={t_s:4.2f}s", fill=(255, 255, 255, 255))
    draw.text((30, 58), f"{status}: {force_n:.0f}N on {force_body}", fill=(255, 255, 255, 255))
    draw.text((30, 82), demo_subtitle, fill=(230, 230, 230, 255))
    _draw_timeline(
        draw,
        image.size,
        t_s=t_s,
        total_duration=total_duration,
        contact_start=contact_start,
        contact_duration=contact_duration,
    )
    if active:
        _draw_contact_arrow(draw, image.size, label=_contact_label(force_body))
    image.save(frame_path)


def _draw_timeline(
    draw: Any,
    image_size: tuple[int, int],
    *,
    t_s: float,
    total_duration: float,
    contact_start: float,
    contact_duration: float,
) -> None:
    width, height = image_size
    x0 = 46
    x1 = width - 46
    y = height - 30
    duration = max(1e-6, float(total_duration))
    contact_x0 = x0 + (x1 - x0) * max(0.0, min(1.0, float(contact_start) / duration))
    contact_x1 = x0 + (x1 - x0) * max(0.0, min(1.0, (float(contact_start) + float(contact_duration)) / duration))
    playhead_x = x0 + (x1 - x0) * max(0.0, min(1.0, float(t_s) / duration))
    draw.line((x0, y, x1, y), fill=(220, 220, 220, 160), width=3)
    draw.line((contact_x0, y, contact_x1, y), fill=(230, 70, 50, 230), width=6)
    draw.ellipse((playhead_x - 5, y - 5, playhead_x + 5, y + 5), fill=(255, 255, 255, 240))
    draw.text((x0, y - 22), "contact window", fill=(245, 245, 245, 220))


def _draw_contact_arrow(draw: Any, image_size: tuple[int, int], *, label: str) -> None:
    width, height = image_size
    start = (width - 166, height - 108)
    end = (width - 72, height - 108)
    color = (230, 70, 50, 235)
    draw.line((*start, *end), fill=color, width=8)
    draw.polygon(
        [
            (end[0], end[1]),
            (end[0] - 18, end[1] - 12),
            (end[0] - 18, end[1] + 12),
        ],
        fill=color,
    )
    draw.text((width - 184, height - 138), label, fill=(255, 255, 255, 235))


def _contact_label(force_body: str) -> str:
    body = str(force_body).lower()
    if "foot" in body:
        return "foot contact"
    if "thigh" in body or "calf" in body:
        return "leg contact"
    return "body contact"


def _encode_media(frame_dir: Path, *, output_gif: Path, output_mp4: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg to encode GIF/MP4 demo assets.")
    frame_pattern = str(frame_dir / "frame_%05d.png")
    fps_arg = f"{float(fps):.6g}"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            fps_arg,
            "-i",
            frame_pattern,
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(output_mp4),
        ],
        check=True,
    )
    palette = frame_dir / "palette.png"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            fps_arg,
            "-i",
            frame_pattern,
            "-vf",
            "palettegen",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            fps_arg,
            "-i",
            frame_pattern,
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=3",
            str(output_gif),
        ],
        check=True,
    )


def _window_scale(t_s: float, *, start_s: float, duration_s: float, transition_s: float) -> float:
    start = float(start_s)
    end = start + max(0.0, float(duration_s))
    transition = max(0.0, float(transition_s))
    t = float(t_s)
    if t < start or t > end:
        return 0.0
    if transition <= 0.0:
        return 1.0
    if t < start + transition:
        return max(0.0, min(1.0, (t - start) / transition))
    if t > end - transition:
        return max(0.0, min(1.0, (end - t) / transition))
    return 1.0


def _force_vector_xy(force_n: float, angle_rad: float) -> np.ndarray:
    return np.asarray(
        [
            float(force_n) * math.cos(float(angle_rad)),
            float(force_n) * math.sin(float(angle_rad)),
            0.0,
        ],
        dtype=np.float64,
    )


if __name__ == "__main__":
    raise SystemExit(main())
