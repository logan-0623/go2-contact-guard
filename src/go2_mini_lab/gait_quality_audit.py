from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


def analyze_gait_quality(trajectory: dict[str, Any]) -> dict[str, Any]:
    frames = list(trajectory.get("frames") or [])
    metadata = trajectory.get("metadata") or {}
    joint_order = list(metadata.get("joint_order") or _infer_joint_order(frames))
    feet = list(metadata.get("feet") or ["FR", "FL", "RR", "RL"])
    timestamps = [_number(frame.get("t")) for frame in frames]
    dts = _dts(timestamps)

    linear_velocities = [
        _vec(((frame.get("base") or {}).get("linear_velocity") or []), 3)
        for frame in frames
    ]
    angular_velocities = [
        _vec(((frame.get("base") or {}).get("angular_velocity") or []), 3)
        for frame in frames
    ]
    base_positions = [
        _vec(((frame.get("base") or {}).get("position") or []), 3)
        for frame in frames
    ]
    base_quaternions = [
        _vec(((frame.get("base") or {}).get("quaternion") or []), 4)
        for frame in frames
    ]
    roll_pitch_yaw = [_quat_to_rpy(quat) for quat in base_quaternions]
    joint_values = _joint_matrix(frames, joint_order)
    ctrl_values = _ctrl_matrix(frames)

    return {
        "frames": len(frames),
        "duration": _duration(timestamps),
        "dt_mean": _mean(dts),
        "mean_forward_velocity": _mean([velocity[0] for velocity in linear_velocities]),
        "mean_abs_lateral_velocity": _mean([abs(velocity[1]) for velocity in linear_velocities]),
        "mean_abs_yaw_rate": _mean([abs(velocity[2]) for velocity in angular_velocities]),
        "mean_abs_roll": _mean([abs(rpy[0]) for rpy in roll_pitch_yaw]),
        "mean_abs_pitch": _mean([abs(rpy[1]) for rpy in roll_pitch_yaw]),
        "base_z_min": min([position[2] for position in base_positions], default=0.0),
        "base_z_mean": _mean([position[2] for position in base_positions]),
        "base_z_std": _std([position[2] for position in base_positions]),
        "joint_position_std_mean": _mean(_column_stds(joint_values)),
        "joint_velocity_rms": _finite_difference_rms(joint_values, dts, order=1),
        "joint_acceleration_rms": _finite_difference_rms(joint_values, dts, order=2),
        "ctrl_rms": _matrix_rms(ctrl_values),
        "ctrl_rate_rms": _finite_difference_rms(ctrl_values, dts, order=1),
        "contact_ratio": _contact_ratio(frames, feet),
        "contact_switches_per_second": _contact_switches_per_second(frames, feet, _duration(timestamps)),
        "left_right_contact_delta": _left_right_contact_delta(frames),
        "diagonal_contact_delta": _diagonal_contact_delta(frames),
    }


def compare_to_baseline(
    trajectory: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    frames = list(trajectory.get("frames") or [])
    baseline_frames = list(baseline.get("frames") or [])
    count = min(len(frames), len(baseline_frames))
    joint_order = list((trajectory.get("metadata") or {}).get("joint_order") or _infer_joint_order(frames))
    joint_order = [
        joint_name
        for joint_name in joint_order
        if all(joint_name in (frame.get("joints") or {}) for frame in frames[:count])
        and all(joint_name in (frame.get("joints") or {}) for frame in baseline_frames[:count])
    ]

    joint_errors: list[float] = []
    ctrl_errors: list[float] = []
    yaw_rate_errors: list[float] = []
    base_z_errors: list[float] = []
    contact_matches = 0
    contact_total = 0
    feet = list((trajectory.get("metadata") or {}).get("feet") or ["FR", "FL", "RR", "RL"])

    for frame, baseline_frame in zip(frames[:count], baseline_frames[:count]):
        for joint_name in joint_order:
            joint_errors.append(
                abs(
                    float((frame.get("joints") or {}).get(joint_name, 0.0))
                    - float((baseline_frame.get("joints") or {}).get(joint_name, 0.0))
                )
            )
        for value, baseline_value in zip(frame.get("ctrl") or [], baseline_frame.get("ctrl") or []):
            ctrl_errors.append(abs(float(value) - float(baseline_value)))

        angular_velocity = _vec(((frame.get("base") or {}).get("angular_velocity") or []), 3)
        baseline_angular_velocity = _vec(((baseline_frame.get("base") or {}).get("angular_velocity") or []), 3)
        yaw_rate_errors.append(abs(angular_velocity[2] - baseline_angular_velocity[2]))

        position = _vec(((frame.get("base") or {}).get("position") or []), 3)
        baseline_position = _vec(((baseline_frame.get("base") or {}).get("position") or []), 3)
        base_z_errors.append(abs(position[2] - baseline_position[2]))

        contacts = frame.get("contacts") or {}
        baseline_contacts = baseline_frame.get("contacts") or {}
        for foot in feet:
            contact_total += 1
            if bool(contacts.get(foot)) == bool(baseline_contacts.get(foot)):
                contact_matches += 1

    return {
        "aligned_frames": count,
        "mean_abs_joint_position_error": _mean(joint_errors),
        "mean_abs_ctrl_error": _mean(ctrl_errors),
        "mean_abs_base_yaw_rate_error": _mean(yaw_rate_errors),
        "mean_abs_base_z_error": _mean(base_z_errors),
        "contact_agreement": contact_matches / contact_total if contact_total else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit gait quality metrics from a Go2 trajectory JSON, optionally against an ONNX baseline."
    )
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    trajectory = _load_json(args.trajectory)
    report: dict[str, Any] = {
        "trajectory": str(args.trajectory),
        "metrics": analyze_gait_quality(trajectory),
    }
    if args.baseline is not None:
        baseline = _load_json(args.baseline)
        report["baseline"] = str(args.baseline)
        report["baseline_metrics"] = analyze_gait_quality(baseline)
        report["comparison"] = compare_to_baseline(trajectory, baseline)

    text = json.dumps(report, indent=2 if args.pretty else None)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_joint_order(frames: list[dict[str, Any]]) -> list[str]:
    for frame in frames:
        joints = frame.get("joints")
        if isinstance(joints, dict) and joints:
            return list(joints)
    return []


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vec(values: Sequence[Any], size: int) -> list[float]:
    result = [_number(value) for value in values[:size]]
    result = [0.0 if value is None else value for value in result]
    while len(result) < size:
        result.append(0.0)
    return result


def _dts(timestamps: list[float | None]) -> list[float]:
    return [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
        if timestamps[index] is not None
        and timestamps[index - 1] is not None
        and timestamps[index] > timestamps[index - 1]
    ]


def _duration(timestamps: list[float | None]) -> float:
    values = [value for value in timestamps if value is not None]
    if len(values) < 2:
        return 0.0
    return max(0.0, values[-1] - values[0])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _joint_matrix(frames: list[dict[str, Any]], joint_order: list[str]) -> list[list[float]]:
    return [
        [float((frame.get("joints") or {}).get(joint_name, 0.0)) for joint_name in joint_order]
        for frame in frames
    ]


def _ctrl_matrix(frames: list[dict[str, Any]]) -> list[list[float]]:
    width = max((len(frame.get("ctrl") or []) for frame in frames), default=0)
    matrix: list[list[float]] = []
    for frame in frames:
        values = [float(value) for value in (frame.get("ctrl") or [])[:width]]
        while len(values) < width:
            values.append(0.0)
        matrix.append(values)
    return matrix


def _column_stds(matrix: list[list[float]]) -> list[float]:
    if not matrix or not matrix[0]:
        return []
    return [
        _std([row[column] for row in matrix])
        for column in range(len(matrix[0]))
    ]


def _matrix_rms(matrix: list[list[float]]) -> float:
    values = [value for row in matrix for value in row]
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _finite_difference_rms(matrix: list[list[float]], dts: list[float], *, order: int) -> float:
    if len(matrix) < 2 or not matrix[0]:
        return 0.0
    current = [list(row) for row in matrix]
    current_dts = list(dts)
    for _ in range(order):
        if len(current) < 2:
            return 0.0
        next_matrix = []
        for index in range(1, len(current)):
            dt = current_dts[index - 1] if index - 1 < len(current_dts) else 0.0
            if dt <= 0:
                continue
            next_matrix.append(
                [
                    (current[index][column] - current[index - 1][column]) / dt
                    for column in range(len(current[index]))
                ]
            )
        current = next_matrix
        current_dts = current_dts[1:]
    return _matrix_rms(current)


def _quat_to_rpy(quat: list[float]) -> tuple[float, float, float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _contact_ratio(frames: list[dict[str, Any]], feet: list[str]) -> dict[str, float]:
    if not frames:
        return {foot: 0.0 for foot in feet}
    return {
        foot: sum(1 for frame in frames if bool((frame.get("contacts") or {}).get(foot))) / len(frames)
        for foot in feet
    }


def _contact_switches_per_second(
    frames: list[dict[str, Any]],
    feet: list[str],
    duration: float,
) -> dict[str, float]:
    if duration <= 0.0:
        return {foot: 0.0 for foot in feet}
    switches: dict[str, float] = {}
    for foot in feet:
        count = 0
        previous: bool | None = None
        for frame in frames:
            current = bool((frame.get("contacts") or {}).get(foot))
            if previous is not None and current != previous:
                count += 1
            previous = current
        switches[foot] = count / duration
    return switches


def _left_right_contact_delta(frames: list[dict[str, Any]]) -> float:
    if not frames:
        return 0.0
    deltas = []
    for frame in frames:
        contacts = frame.get("contacts") or {}
        left = float(bool(contacts.get("FL"))) + float(bool(contacts.get("RL")))
        right = float(bool(contacts.get("FR"))) + float(bool(contacts.get("RR")))
        deltas.append((left - right) / 2.0)
    return _mean(deltas)


def _diagonal_contact_delta(frames: list[dict[str, Any]]) -> float:
    if not frames:
        return 0.0
    deltas = []
    for frame in frames:
        contacts = frame.get("contacts") or {}
        diagonal_a = float(bool(contacts.get("FR"))) + float(bool(contacts.get("RL")))
        diagonal_b = float(bool(contacts.get("FL"))) + float(bool(contacts.get("RR")))
        deltas.append((diagonal_a - diagonal_b) / 2.0)
    return _mean(deltas)


if __name__ == "__main__":
    raise SystemExit(main())
