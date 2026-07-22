from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .controller import UNITREE_GO2_JOINTS
from .export import _strip_raw_state_arrays
from .trajectory import BaseState, make_trajectory
from .trajectory_analysis import metrics_output_path, validate_trajectory


def make_backflip_reference(
    *,
    duration: float = 2.2,
    dt: float = 0.02,
) -> dict:
    frames = []
    steps = max(1, int(duration / dt))
    previous_joints = _pose("stand")

    for index in range(steps + 1):
        t = round(index * dt, 6)
        progress = min(1.0, t / duration)
        base_z, pitch, contacts, pose_name = _reference_state(progress)
        joints = _blend_pose(pose_name, progress)
        qvel_joints = [
            (joints[name] - previous_joints[name]) / dt if index else 0.0
            for name in UNITREE_GO2_JOINTS
        ]
        base = BaseState(
            position=[0.0, 0.0, base_z],
            quaternion=_quat_y(pitch),
            linear_velocity=[0.0, 0.0, 0.0],
            angular_velocity=[0.0, pitch_rate_reference(progress, duration), 0.0],
        )
        frames.append(
            {
                "t": t,
                "qpos": [
                    *base.position,
                    *base.quaternion,
                    *[joints[name] for name in UNITREE_GO2_JOINTS],
                ],
                "qvel": [
                    *base.linear_velocity,
                    *base.angular_velocity,
                    *qvel_joints,
                ],
                "ctrl": [0.0 for _ in UNITREE_GO2_JOINTS],
                "base": asdict(base),
                "joints": {name: round(joints[name], 6) for name in UNITREE_GO2_JOINTS},
                "contacts": contacts,
                "gait_phase": {foot: round(progress, 6) for foot in ("FR", "FL", "RR", "RL")},
            }
        )
        previous_joints = joints

    return make_trajectory(
        frames=frames,
        joint_order=list(UNITREE_GO2_JOINTS),
        dt=dt,
        source="kinematic-backflip-reference",
        extra_metadata={
            "mode": "kinematic-backflip-reference",
            "preset": "backflip_reference",
            "allow_flight_phase": True,
            "reference_motion": True,
        },
        notes=(
            "Kinematic backflip reference for future imitation or reference-tracking rewards. "
            "This is not a MuJoCo physics rollout and not a trained policy."
        ),
    )


def pitch_rate_reference(progress: float, duration: float) -> float:
    if 0.25 <= progress <= 0.72:
        return (2.0 * math.pi) / max(duration * 0.47, 1e-6)
    return 0.0


def _reference_state(progress: float) -> tuple[float, float, dict[str, bool], str]:
    if progress < 0.18:
        local = _smooth(progress / 0.18)
        return _lerp(0.36, 0.335, local), 0.0, _contacts(True), "crouch"
    if progress < 0.28:
        local = _smooth((progress - 0.18) / 0.10)
        return _lerp(0.335, 0.52, local), _lerp(0.0, -0.25, local), _contacts(True), "takeoff"
    if progress < 0.72:
        local = (progress - 0.28) / 0.44
        z = 0.52 + 0.28 * math.sin(math.pi * local)
        pitch = _lerp(-0.25, 2.0 * math.pi - 0.15, local)
        return z, pitch, _contacts(False), "tuck"
    if progress < 0.86:
        local = _smooth((progress - 0.72) / 0.14)
        return _lerp(0.52, 0.36, local), _lerp(2.0 * math.pi - 0.15, 2.0 * math.pi, local), _contacts(True), "landing"
    local = _smooth((progress - 0.86) / 0.14)
    return 0.36, 2.0 * math.pi, _contacts(True), "stand" if local >= 0.65 else "landing"


def _blend_pose(pose_name: str, progress: float) -> dict[str, float]:
    if progress < 0.18:
        return _interpolate_pose("stand", "crouch", _smooth(progress / 0.18))
    if progress < 0.28:
        return _interpolate_pose("crouch", "takeoff", _smooth((progress - 0.18) / 0.10))
    if progress < 0.72:
        return _pose("tuck")
    if progress < 0.86:
        return _interpolate_pose("tuck", "landing", _smooth((progress - 0.72) / 0.14))
    return _interpolate_pose("landing", "stand", _smooth((progress - 0.86) / 0.14))


def _pose(name: str) -> dict[str, float]:
    poses = {
        "stand": (0.0, 0.608813, -1.21763),
        "crouch": (0.0, 1.05, -2.05),
        "takeoff": (0.0, 0.25, -0.92),
        "tuck": (0.0, 1.20, -2.15),
        "landing": (0.0, 0.72, -1.45),
    }
    hip, thigh, calf = poses[name]
    result = {}
    for joint_name in UNITREE_GO2_JOINTS:
        if "_hip_" in joint_name:
            side = -1.0 if joint_name.startswith(("FR", "RR")) else 1.0
            result[joint_name] = hip + side * (0.08 if name in {"tuck", "landing"} else 0.0)
        elif "_thigh_" in joint_name:
            result[joint_name] = thigh
        elif "_calf_" in joint_name:
            result[joint_name] = calf
    return result


def _interpolate_pose(start: str, end: str, amount: float) -> dict[str, float]:
    a = _pose(start)
    b = _pose(end)
    return {name: _lerp(a[name], b[name], amount) for name in UNITREE_GO2_JOINTS}


def _quat_y(angle: float) -> list[float]:
    half = 0.5 * angle
    return [round(math.cos(half), 8), 0.0, round(math.sin(half), 8), 0.0]


def _contacts(value: bool) -> dict[str, bool]:
    return {foot: value for foot in ("FR", "FL", "RR", "RL")}


def _smooth(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a kinematic Go2 backflip reference trajectory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backflip_reference.json"),
        help="Output trajectory JSON path.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.2,
        help="Reference duration in seconds.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Reference timestep in seconds.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Omit raw qpos/qvel/ctrl arrays from the exported replay JSON.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not write metrics JSON.",
    )
    args = parser.parse_args(argv)

    trajectory = make_backflip_reference(duration=args.duration, dt=args.dt)
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
    print(f"wrote {args.output} ({len(trajectory['frames'])} frames)")
    print("mode: kinematic-backflip-reference")
    print(f"duration: {trajectory['metadata']['duration']:.3f}s")
    print(f"health: {result['metrics']['health_status']}")
    if result["warnings"]:
        print(f"validation warnings: {len(result['warnings'])}")
        for warning in result["warnings"][:5]:
            print(f"- {warning}")
    if result["errors"]:
        print(f"validation errors: {len(result['errors'])}")
        for error in result["errors"][:5]:
            print(f"- {error}")

    if not args.no_metrics:
        metrics_path = metrics_output_path(args.output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
        print(f"wrote {metrics_path}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
