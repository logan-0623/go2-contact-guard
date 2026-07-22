from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..collect_nominal_gait_reference import _bin_phase_samples, _quat_wxyz_to_rpy
from ..rl_env import FEET, FULLBODY_TRACKING_KEYPOINTS
from .reference_audit import ReferenceAuditReport, audit_nominal_gait_file, write_audit_report


DEFAULT_MODEL = Path("external/unitree_mujoco/unitree_robots/go2/flat_scene.xml")


def convert_rich_rollout_file(
    source: str | Path,
    output: str | Path,
    *,
    audit_output: str | Path | None = None,
    phase_bins: int = 200,
    gait_frequency_hz: float = 1.7,
    discard_initial_s: float = 1.0,
    model_path: str | Path | None = DEFAULT_MODEL,
) -> ReferenceAuditReport:
    source_path = Path(source)
    output_path = Path(output)
    with np.load(source_path, allow_pickle=True) as data:
        arrays = {name: data[name] for name in data.files if name != "metadata"}
        metadata = _metadata_from_npz(data)

    raw_samples = _rich_arrays_to_samples(
        arrays,
        metadata=metadata,
        source_path=source_path,
        gait_frequency_hz=float(gait_frequency_hz),
        discard_initial_s=float(discard_initial_s),
        model_path=None if model_path is None else Path(model_path),
    )
    if not raw_samples:
        raise RuntimeError("rich rollout conversion produced no samples")

    binned = _bin_phase_samples(
        raw_samples,
        phase_bins=max(2, int(phase_bins)),
        gait_frequency_hz=float(gait_frequency_hz),
        np_module=np,
    )
    keypoint_names = list(FULLBODY_TRACKING_KEYPOINTS) if "keypoint_pos_ref" in binned else []
    output_metadata = {
        "source": "onnx_rollout",
        "policy_format": metadata.get("policy_format", "onnx"),
        "reference_policy": metadata.get("source_policy") or metadata.get("reference_policy"),
        "raw_dataset": str(source_path),
        "model": str(model_path) if model_path is not None else metadata.get("model"),
        "task": metadata.get("task", "velocity_flat"),
        "raw_samples": len(raw_samples),
        "phase_bins": max(2, int(phase_bins)),
        "discard_initial_s": float(discard_initial_s),
        "control_dt": float(metadata.get("control_dt", 0.02)),
        "target_forward_velocity": float(metadata.get("target_forward_velocity", 0.4)),
        "target_lateral_velocity": float(metadata.get("target_lateral_velocity", 0.0)),
        "target_yaw_rate": float(metadata.get("target_yaw_rate", 0.0)),
        "gait_frequency_hz": float(gait_frequency_hz),
        "action_names": list(metadata.get("action_names") or []),
        "foot_names": list(metadata.get("foot_names") or FEET),
        "keypoint_names": keypoint_names,
        "velocity_pose_profile": metadata.get("velocity_pose_profile"),
        "policy_leg_order": metadata.get("policy_leg_order"),
        "pd_kp": metadata.get("pd_kp"),
        "pd_kd": metadata.get("pd_kd"),
        "torque_limit": metadata.get("torque_limit"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **binned, metadata=json.dumps(output_metadata))
    report = audit_nominal_gait_file(output_path)
    if audit_output is not None:
        write_audit_report(audit_output, report)
    return report


def _metadata_from_npz(data: Any) -> dict[str, Any]:
    if "metadata" not in data.files:
        return {}
    value = data["metadata"]
    try:
        item = value.item()
    except ValueError:
        item = value
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if isinstance(item, str):
        return json.loads(item)
    if isinstance(item, Mapping):
        return dict(item)
    return {}


def _rich_arrays_to_samples(
    arrays: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
    source_path: Path,
    gait_frequency_hz: float,
    discard_initial_s: float,
    model_path: Path | None,
) -> list[dict[str, Any]]:
    actions = _required_array(arrays, "actions", source_path)
    joint_positions = _required_array(arrays, "joint_positions", source_path)
    joint_velocities = _required_array(arrays, "joint_velocities", source_path)
    joint_targets = _required_array(arrays, "joint_targets", source_path)
    foot_positions = _required_array(arrays, "foot_positions", source_path)
    foot_velocities = _required_array(arrays, "foot_velocities", source_path)
    contacts = _required_array(arrays, "contacts", source_path)
    base_positions = _required_array(arrays, "base_positions", source_path)
    base_quaternions = _required_array(arrays, "base_quaternions", source_path)
    base_linear_velocities = _required_array(arrays, "base_linear_velocities", source_path)
    base_angular_velocities = _required_array(arrays, "base_angular_velocities", source_path)
    sample_count = int(actions.shape[0])
    for name, value in {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "joint_targets": joint_targets,
        "foot_positions": foot_positions,
        "foot_velocities": foot_velocities,
        "contacts": contacts,
        "base_positions": base_positions,
        "base_quaternions": base_quaternions,
        "base_linear_velocities": base_linear_velocities,
        "base_angular_velocities": base_angular_velocities,
    }.items():
        if int(value.shape[0]) != sample_count:
            raise ValueError(f"{name} length {value.shape[0]} does not match actions length {sample_count}")

    qpos = arrays.get("qpos")
    qvel = arrays.get("qvel")
    keypoint_pos, keypoint_vel = _build_keypoint_references(
        qpos=qpos,
        qvel=qvel,
        model_path=model_path,
    )

    control_dt = float(metadata.get("control_dt", 0.02))
    selected_indices = _selected_indices(
        metadata=metadata,
        sample_count=sample_count,
        control_dt=control_dt,
        discard_initial_s=discard_initial_s,
    )
    samples: list[dict[str, Any]] = []
    for index, local_step in selected_indices:
        phase = (float(local_step) * control_dt * float(gait_frequency_hz)) % 1.0
        sample = {
            "phase_ref": phase,
            "q_ref": np.asarray(joint_positions[index], dtype=np.float32),
            "dq_ref": np.asarray(joint_velocities[index], dtype=np.float32),
            "action_ref": np.asarray(actions[index], dtype=np.float32),
            "joint_target_ref": np.asarray(joint_targets[index], dtype=np.float32),
            "foot_pos_ref": np.asarray(foot_positions[index], dtype=np.float32),
            "foot_vel_ref": np.asarray(foot_velocities[index], dtype=np.float32),
            "contact_ref": np.asarray(contacts[index], dtype=np.float32),
            "base_pos_ref": np.asarray(base_positions[index], dtype=np.float32),
            "base_quat_ref": np.asarray(base_quaternions[index], dtype=np.float32),
            "base_rpy_ref": np.asarray(_quat_wxyz_to_rpy(base_quaternions[index]), dtype=np.float32),
            "base_height_ref": float(base_positions[index][2]),
            "base_vel_ref": np.asarray(base_linear_velocities[index], dtype=np.float32),
            "base_angvel_ref": np.asarray(base_angular_velocities[index], dtype=np.float32),
        }
        if keypoint_pos is not None and keypoint_vel is not None:
            sample["keypoint_pos_ref"] = keypoint_pos[index]
            sample["keypoint_vel_ref"] = keypoint_vel[index]
        samples.append(sample)
    return samples


def _required_array(arrays: Mapping[str, np.ndarray], name: str, source_path: Path) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"rich rollout dataset missing {name}: {source_path}")
    return np.asarray(arrays[name], dtype=np.float32)


def _selected_indices(
    *,
    metadata: Mapping[str, Any],
    sample_count: int,
    control_dt: float,
    discard_initial_s: float,
) -> list[tuple[int, int]]:
    episode_lengths = metadata.get("episode_lengths")
    if not episode_lengths:
        episode_lengths = [sample_count]
    selected: list[tuple[int, int]] = []
    offset = 0
    discard_steps = max(0, int(math.ceil(float(discard_initial_s) / max(1e-6, control_dt))))
    for length_value in episode_lengths:
        length = int(length_value)
        end = min(sample_count, offset + length)
        for index in range(offset, end):
            local_step = index - offset
            if local_step >= discard_steps:
                selected.append((index, local_step))
        offset = end
        if offset >= sample_count:
            break
    return selected


def _build_keypoint_references(
    *,
    qpos: Any,
    qvel: Any,
    model_path: Path | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if model_path is None or qpos is None or qvel is None:
        return None, None
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required to reconstruct full-body keypoint references") from exc

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos_array = np.asarray(qpos, dtype=np.float32)
    qvel_array = np.asarray(qvel, dtype=np.float32)
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in FULLBODY_TRACKING_KEYPOINTS
    ]
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for index in range(int(qpos_array.shape[0])):
        data.qpos[: min(data.qpos.shape[0], qpos_array.shape[1])] = qpos_array[index, : data.qpos.shape[0]]
        data.qvel[: min(data.qvel.shape[0], qvel_array.shape[1])] = qvel_array[index, : data.qvel.shape[0]]
        mujoco.mj_forward(model, data)
        root_position = np.asarray(data.qpos[:3], dtype=np.float32)
        root_rotation = _quat_wxyz_to_matrix(data.qpos[3:7])
        keypoint_positions = []
        keypoint_velocities = []
        for body_id in body_ids:
            if body_id < 0:
                keypoint_positions.append(np.zeros(3, dtype=np.float32))
                keypoint_velocities.append(np.zeros(3, dtype=np.float32))
                continue
            position_world = np.asarray(data.xpos[body_id], dtype=np.float32)
            velocity_world = np.asarray(data.cvel[body_id][3:6], dtype=np.float32)
            keypoint_positions.append(root_rotation.T @ (position_world - root_position))
            keypoint_velocities.append(root_rotation.T @ velocity_world)
        positions.append(np.stack(keypoint_positions).astype(np.float32))
        velocities.append(np.stack(keypoint_velocities).astype(np.float32))
    return np.stack(positions).astype(np.float32), np.stack(velocities).astype(np.float32)


def _quat_wxyz_to_matrix(quat: Any) -> np.ndarray:
    w, x, y, z = [float(value) for value in quat]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a rich ONNX rollout dataset into nominal gait reference schema.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--phase-bins", type=int, default=200)
    parser.add_argument("--gait-frequency", type=float, default=1.7)
    parser.add_argument("--discard-initial-s", type=float, default=1.0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-model-keypoints", action="store_true")
    parser.add_argument("--allow-audit-fail", action="store_true")
    args = parser.parse_args(argv)

    model_path = None if args.no_model_keypoints else args.model
    report = convert_rich_rollout_file(
        args.source,
        args.output,
        audit_output=args.audit_output,
        phase_bins=args.phase_bins,
        gait_frequency_hz=args.gait_frequency,
        discard_initial_s=args.discard_initial_s,
        model_path=model_path,
    )
    print(f"wrote {args.output}")
    print(f"mean_vx: {report.metrics['mean_vx']:.3f}")
    print(f"mean_abs_vy: {report.metrics['mean_abs_vy']:.3f}")
    print(f"mean_abs_yaw_rate: {report.metrics['mean_abs_yaw_rate']:.3f}")
    print(f"base_z_min: {report.metrics['base_z_min']:.3f}")
    print(f"recommended_anti_collapse_z_safe: {report.metrics['recommended_anti_collapse_z_safe']:.3f}")
    print(f"audit: {'PASS' if report.passed else 'FAIL'}")
    if args.audit_output is not None:
        print(f"wrote {args.audit_output}")
    if not report.passed:
        for name, reason in report.failures.items():
            print(f"FAIL {name}: {reason}")
    return 0 if report.passed or args.allow_audit_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
