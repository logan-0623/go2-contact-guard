from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .controller import LEGS
from .trajectory import SCHEMA


BASE_Z_WARNING_THRESHOLD = 0.32
BASE_Z_FALL_THRESHOLD = 0.25


def load_trajectory(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.name == "trajectory.json":
        return path.with_name("trajectory_metrics.json")
    return path.with_name(f"{path.stem}_metrics.json")


def analyze_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    frames = trajectory.get("frames") or []
    metadata = trajectory.get("metadata") or {}
    feet = metadata.get("feet") or list(LEGS)
    allow_flight_phase = bool(metadata.get("allow_flight_phase"))
    joint_order = metadata.get("joint_order") or _infer_joint_order(frames)
    timestamps = [_number(frame.get("t")) for frame in frames]
    dts = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
        if timestamps[i] is not None and timestamps[i - 1] is not None
    ]
    base_z = [
        _number(((frame.get("base") or {}).get("position") or [None, None, None])[2])
        for frame in frames
    ]
    base_z_values = [value for value in base_z if value is not None]
    base_z_min = min(base_z_values) if base_z_values else None
    base_z_min_frame = (
        next((index for index, value in enumerate(base_z) if value == base_z_min), None)
        if base_z_min is not None
        else None
    )
    linear_velocity = [
        (frame.get("base") or {}).get("linear_velocity") or [0.0, 0.0, 0.0]
        for frame in frames
    ]
    speeds = [
        math.sqrt(sum(float(component or 0.0) ** 2 for component in velocity[:3]))
        for velocity in linear_velocity
    ]
    vx_values = [float((velocity + [0.0])[0] or 0.0) for velocity in linear_velocity]

    contact_ratio = _contact_ratio(frames, feet)
    max_missing_contact_run = _max_missing_contact_run(frames, feet)
    fall_detected = bool(base_z_values and min(base_z_values) < BASE_Z_FALL_THRESHOLD)
    warning_events = _warning_events(
        base_z=base_z,
        timestamps=timestamps,
        max_missing_contact_run=max_missing_contact_run,
        fall_detected=fall_detected,
        allow_flight_phase=allow_flight_phase,
    )

    return {
        "schema": trajectory.get("schema"),
        "source": metadata.get("source"),
        "mode": metadata.get("mode"),
        "preset": metadata.get("preset"),
        "duration": timestamps[-1] if timestamps else 0.0,
        "frames": len(frames),
        "dt_mean": _mean(dts),
        "dt_min": min(dts) if dts else 0.0,
        "dt_max": max(dts) if dts else 0.0,
        "base_z_min": base_z_min,
        "base_z_mean": _mean(base_z_values),
        "base_z_max": max(base_z_values) if base_z_values else None,
        "base_z_min_frame": base_z_min_frame,
        "base_z_min_t": timestamps[base_z_min_frame] if base_z_min_frame is not None else None,
        "mean_forward_velocity": _mean(vx_values),
        "max_forward_velocity": max(vx_values) if vx_values else 0.0,
        "mean_speed": _mean(speeds),
        "max_speed": max(speeds) if speeds else 0.0,
        "fall_detected": fall_detected,
        "health_status": "fall" if fall_detected else "warning" if warning_events else "good",
        "warning_events": warning_events,
        "contact_ratio": contact_ratio,
        "max_missing_contact_run": max_missing_contact_run,
        "joint_angle_range": _joint_angle_range(frames, joint_order),
    }


def validate_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if trajectory.get("schema") != SCHEMA:
        errors.append(f"schema mismatch: expected {SCHEMA}, got {trajectory.get('schema')!r}")

    frames = trajectory.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("frames must be a non-empty list")
        frames = []

    metadata = trajectory.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        metadata = {}

    joint_order = metadata.get("joint_order")
    if not isinstance(joint_order, list) or not joint_order:
        warnings.append("metadata.joint_order is missing or empty")
        joint_order = _infer_joint_order(frames)

    feet = metadata.get("feet") or list(LEGS)
    if not isinstance(feet, list) or not feet:
        warnings.append("metadata.feet is missing or empty")
        feet = list(LEGS)

    previous_t: float | None = None
    base_z_values: list[float] = []
    missing_joints: set[str] = set()
    missing_contact_fields: set[str] = set()

    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            errors.append(f"frame {index} must be an object")
            continue

        for key in ("t", "qpos", "qvel", "ctrl", "base", "joints", "contacts", "gait_phase"):
            if key not in frame:
                errors.append(f"frame {index} missing {key}")

        t = _number(frame.get("t"))
        if t is None:
            errors.append(f"frame {index} has invalid timestamp")
        elif previous_t is not None and t <= previous_t:
            errors.append(f"timestamp is not strictly increasing at frame {index}: {t} <= {previous_t}")
        previous_t = t

        finite_path = _first_non_finite_path(frame)
        if finite_path:
            errors.append(f"frame {index} contains non-finite value at {finite_path}")

        base = frame.get("base") or {}
        position = base.get("position") or []
        if len(position) < 3:
            errors.append(f"frame {index} base.position must contain x, y, z")
        else:
            base_z = _number(position[2])
            if base_z is None:
                errors.append(f"frame {index} base.position[2] is invalid")
            else:
                base_z_values.append(base_z)

        joints = frame.get("joints") or {}
        for joint_name in joint_order:
            if joint_name not in joints:
                missing_joints.add(joint_name)

        contacts = frame.get("contacts") or {}
        for foot in feet:
            if foot not in contacts:
                missing_contact_fields.add(str(foot))

    if missing_joints:
        preview = ", ".join(sorted(missing_joints)[:8])
        errors.append(f"missing joint samples: {preview}")

    if missing_contact_fields:
        preview = ", ".join(sorted(missing_contact_fields)[:8])
        warnings.append(f"missing contact fields: {preview}")

    metrics = analyze_trajectory(trajectory)

    if frames and len(frames) < 2:
        warnings.append("trajectory has fewer than 2 frames")

    if metrics["duration"] and metrics["duration"] < 0.5:
        warnings.append(f"trajectory duration is very short: {metrics['duration']:.3f}s")

    base_z_min = metrics.get("base_z_min")
    if base_z_min is not None and base_z_min < BASE_Z_WARNING_THRESHOLD:
        warnings.append(
            f"base_z dropped below {BASE_Z_WARNING_THRESHOLD:.2f}m at frame "
            f"{metrics.get('base_z_min_frame')}: min={base_z_min:.3f}m"
        )

    if metrics.get("fall_detected"):
        warnings.append("possible fall detected")

    if not metadata.get("allow_flight_phase"):
        for foot, missing_run in metrics["max_missing_contact_run"].items():
            if missing_run >= 10:
                warnings.append(f"{foot} foot contact missing for {missing_run} consecutive frames")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def format_validation_report(result: dict[str, Any], trajectory_path: str | Path | None = None) -> str:
    metrics = result["metrics"]
    lines = [
        f"Trajectory: {trajectory_path}" if trajectory_path else None,
        f"Schema: {metrics.get('schema')}",
        f"Frames: {metrics['frames']}",
        f"Duration: {_fmt(metrics['duration'])}s",
        f"dt mean: {_fmt(metrics['dt_mean'])}s",
    ]
    lines = [line for line in lines if line is not None]

    lines.extend(
        [
            "",
            "Base height:",
            f"  min: {_fmt(metrics['base_z_min'])} m",
            f"  mean: {_fmt(metrics['base_z_mean'])} m",
            f"  max: {_fmt(metrics['base_z_max'])} m",
            f"  min frame: {metrics.get('base_z_min_frame')} at {_fmt(metrics.get('base_z_min_t'))}s",
            "",
            "Forward velocity:",
            f"  mean: {_fmt(metrics['mean_forward_velocity'])} m/s",
            f"  max: {_fmt(metrics['max_forward_velocity'])} m/s",
        ]
    )

    contact_ratio = metrics.get("contact_ratio") or {}
    if contact_ratio:
        lines.append("")
        lines.append("Contact ratio:")
        lines.extend(f"  {foot}: {ratio:.2f}" for foot, ratio in contact_ratio.items())

    lines.extend(
        [
            "",
            "Health:",
            f"  status: {metrics.get('health_status')}",
            f"  fall_detected: {str(metrics['fall_detected']).lower()}",
            "  nan_check: " + ("passed" if not any("non-finite" in e for e in result["errors"]) else "failed"),
            "  joint_coverage: " + ("passed" if not any("missing joint" in e for e in result["errors"]) else "failed"),
            "  contact_coverage: " + ("passed" if not any("missing contact" in w for w in result["warnings"]) else "warnings"),
        ]
    )

    if result["errors"]:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result["errors"])

    if result["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result["warnings"])
    else:
        lines.append("")
        lines.append("Warnings:")
        lines.append("none")

    return "\n".join(lines)


def _infer_joint_order(frames: list[dict[str, Any]]) -> list[str]:
    for frame in frames:
        joints = frame.get("joints")
        if isinstance(joints, dict) and joints:
            return list(joints.keys())
    return []


def _contact_ratio(frames: list[dict[str, Any]], feet: list[str]) -> dict[str, float]:
    if not frames:
        return {foot: 0.0 for foot in feet}
    return {
        str(foot): sum(1 for frame in frames if (frame.get("contacts") or {}).get(foot)) / len(frames)
        for foot in feet
    }


def _max_missing_contact_run(frames: list[dict[str, Any]], feet: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for foot in feet:
        current = 0
        longest = 0
        for frame in frames:
            if (frame.get("contacts") or {}).get(foot):
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        result[str(foot)] = longest
    return result


def _warning_events(
    *,
    base_z: list[float | None],
    timestamps: list[float | None],
    max_missing_contact_run: dict[str, int],
    fall_detected: bool,
    allow_flight_phase: bool = False,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    low_index = next(
        (index for index, value in enumerate(base_z) if value is not None and value < BASE_Z_WARNING_THRESHOLD),
        None,
    )
    if low_index is not None:
        events.append(
            {
                "type": "base_z_low",
                "severity": "warning",
                "frame": low_index,
                "t": timestamps[low_index],
                "message": f"base_z dropped below {BASE_Z_WARNING_THRESHOLD:.2f}m",
            }
        )

    fall_index = next(
        (index for index, value in enumerate(base_z) if value is not None and value < BASE_Z_FALL_THRESHOLD),
        None,
    )
    if fall_detected and fall_index is not None:
        events.append(
            {
                "type": "fall_detected",
                "severity": "error",
                "frame": fall_index,
                "t": timestamps[fall_index],
                "message": f"base_z dropped below {BASE_Z_FALL_THRESHOLD:.2f}m",
            }
        )

    if not allow_flight_phase:
        for foot, missing_run in max_missing_contact_run.items():
            if missing_run >= 10:
                events.append(
                    {
                        "type": "contact_gap",
                        "severity": "warning",
                        "frame": None,
                        "t": None,
                        "message": f"{foot} foot contact missing for {missing_run} consecutive frames",
                    }
                )

    return events


def _joint_angle_range(frames: list[dict[str, Any]], joint_order: list[str]) -> dict[str, list[float]]:
    ranges: dict[str, list[float]] = {}
    for joint_name in joint_order:
        values = [
            _number((frame.get("joints") or {}).get(joint_name))
            for frame in frames
        ]
        values = [value for value in values if value is not None]
        if values:
            ranges[joint_name] = [min(values), max(values)]
    return ranges


def _first_non_finite_path(value: Any, path: str = "$") -> str | None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    if isinstance(value, int):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else path
    if isinstance(value, list):
        for index, item in enumerate(value):
            child = _first_non_finite_path(item, f"{path}[{index}]")
            if child:
                return child
    if isinstance(value, dict):
        for key, item in value.items():
            child = _first_non_finite_path(item, f"{path}.{key}")
            if child:
                return child
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
