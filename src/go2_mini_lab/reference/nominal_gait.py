from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_REFERENCE_ARRAYS = (
    "phase_ref",
    "q_ref",
    "dq_ref",
    "action_ref",
    "joint_target_ref",
    "foot_pos_ref",
    "foot_vel_ref",
    "contact_ref",
    "base_pos_ref",
    "base_quat_ref",
    "base_rpy_ref",
    "base_height_ref",
    "base_vel_ref",
    "base_angvel_ref",
)


@dataclass(frozen=True)
class NominalGaitSample:
    phase: float
    q_ref: np.ndarray
    dq_ref: np.ndarray
    action_ref: np.ndarray
    joint_target_ref: np.ndarray
    foot_pos_ref: np.ndarray
    foot_vel_ref: np.ndarray
    contact_ref: np.ndarray
    base_pos_ref: np.ndarray
    base_quat_ref: np.ndarray
    base_rpy_ref: np.ndarray
    base_height_ref: float
    base_vel_ref: np.ndarray
    base_angvel_ref: np.ndarray
    keypoint_pos_ref: np.ndarray
    keypoint_vel_ref: np.ndarray


@dataclass(frozen=True)
class NominalGaitReference:
    phase_ref: np.ndarray
    q_ref: np.ndarray
    dq_ref: np.ndarray
    action_ref: np.ndarray
    joint_target_ref: np.ndarray
    foot_pos_ref: np.ndarray
    foot_vel_ref: np.ndarray
    contact_ref: np.ndarray
    base_pos_ref: np.ndarray
    base_quat_ref: np.ndarray
    base_rpy_ref: np.ndarray
    base_height_ref: np.ndarray
    base_vel_ref: np.ndarray
    base_angvel_ref: np.ndarray
    keypoint_pos_ref: np.ndarray
    keypoint_vel_ref: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        *,
        phase_ref: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        action_ref: np.ndarray,
        joint_target_ref: np.ndarray,
        foot_pos_ref: np.ndarray,
        foot_vel_ref: np.ndarray,
        contact_ref: np.ndarray,
        base_pos_ref: np.ndarray,
        base_quat_ref: np.ndarray,
        base_rpy_ref: np.ndarray,
        base_height_ref: np.ndarray,
        base_vel_ref: np.ndarray,
        base_angvel_ref: np.ndarray,
        keypoint_pos_ref: np.ndarray | None = None,
        keypoint_vel_ref: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "NominalGaitReference":
        arrays = {
            "phase_ref": _as_float_array(phase_ref),
            "q_ref": _as_float_array(q_ref),
            "dq_ref": _as_float_array(dq_ref),
            "action_ref": _as_float_array(action_ref),
            "joint_target_ref": _as_float_array(joint_target_ref),
            "foot_pos_ref": _as_float_array(foot_pos_ref),
            "foot_vel_ref": _as_float_array(foot_vel_ref),
            "contact_ref": _as_float_array(contact_ref),
            "base_pos_ref": _as_float_array(base_pos_ref),
            "base_quat_ref": _as_float_array(base_quat_ref),
            "base_rpy_ref": _as_float_array(base_rpy_ref),
            "base_height_ref": _as_float_array(base_height_ref),
            "base_vel_ref": _as_float_array(base_vel_ref),
            "base_angvel_ref": _as_float_array(base_angvel_ref),
        }
        sample_count = _validate_reference_arrays(arrays)
        if keypoint_pos_ref is None:
            keypoint_pos_ref = np.zeros((sample_count, 0, 3), dtype=np.float32)
        if keypoint_vel_ref is None:
            keypoint_vel_ref = np.zeros((sample_count, 0, 3), dtype=np.float32)
        arrays["keypoint_pos_ref"] = _as_float_array(keypoint_pos_ref)
        arrays["keypoint_vel_ref"] = _as_float_array(keypoint_vel_ref)
        _validate_first_dim("keypoint_pos_ref", arrays["keypoint_pos_ref"], sample_count)
        _validate_first_dim("keypoint_vel_ref", arrays["keypoint_vel_ref"], sample_count)

        sort_index = np.argsort(arrays["phase_ref"] % 1.0)
        sorted_arrays = {
            name: (value[sort_index] if value.shape[:1] == (sample_count,) else value)
            for name, value in arrays.items()
        }
        sorted_arrays["phase_ref"] = sorted_arrays["phase_ref"] % 1.0
        if np.any(np.diff(sorted_arrays["phase_ref"]) <= 0.0):
            raise ValueError("phase_ref values must be unique after wrapping into [0, 1)")

        return cls(
            phase_ref=sorted_arrays["phase_ref"],
            q_ref=sorted_arrays["q_ref"],
            dq_ref=sorted_arrays["dq_ref"],
            action_ref=sorted_arrays["action_ref"],
            joint_target_ref=sorted_arrays["joint_target_ref"],
            foot_pos_ref=sorted_arrays["foot_pos_ref"],
            foot_vel_ref=sorted_arrays["foot_vel_ref"],
            contact_ref=sorted_arrays["contact_ref"],
            base_pos_ref=sorted_arrays["base_pos_ref"],
            base_quat_ref=sorted_arrays["base_quat_ref"],
            base_rpy_ref=sorted_arrays["base_rpy_ref"],
            base_height_ref=sorted_arrays["base_height_ref"],
            base_vel_ref=sorted_arrays["base_vel_ref"],
            base_angvel_ref=sorted_arrays["base_angvel_ref"],
            keypoint_pos_ref=sorted_arrays["keypoint_pos_ref"],
            keypoint_vel_ref=sorted_arrays["keypoint_vel_ref"],
            metadata=dict(metadata or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NominalGaitReference":
        with np.load(Path(path), allow_pickle=False) as data:
            missing = sorted(set(REQUIRED_REFERENCE_ARRAYS) - set(data.files))
            if missing:
                raise ValueError(f"nominal gait reference missing arrays: {', '.join(missing)}")
            metadata: dict[str, Any] = {}
            if "metadata" in data.files:
                metadata = json.loads(str(data["metadata"].item()))
            keypoint_pos_ref = data["keypoint_pos_ref"] if "keypoint_pos_ref" in data.files else None
            keypoint_vel_ref = data["keypoint_vel_ref"] if "keypoint_vel_ref" in data.files else None
            return cls.from_arrays(
                phase_ref=data["phase_ref"],
                q_ref=data["q_ref"],
                dq_ref=data["dq_ref"],
                action_ref=data["action_ref"],
                joint_target_ref=data["joint_target_ref"],
                foot_pos_ref=data["foot_pos_ref"],
                foot_vel_ref=data["foot_vel_ref"],
                contact_ref=data["contact_ref"],
                base_pos_ref=data["base_pos_ref"],
                base_quat_ref=data["base_quat_ref"],
                base_rpy_ref=data["base_rpy_ref"],
                base_height_ref=data["base_height_ref"],
                base_vel_ref=data["base_vel_ref"],
                base_angvel_ref=data["base_angvel_ref"],
                keypoint_pos_ref=keypoint_pos_ref,
                keypoint_vel_ref=keypoint_vel_ref,
                metadata=metadata,
            )

    def sample(self, phase: float) -> NominalGaitSample:
        wrapped_phase = float(phase) % 1.0
        lo, hi, alpha = _periodic_interp_indices(self.phase_ref, wrapped_phase)
        return NominalGaitSample(
            phase=wrapped_phase,
            q_ref=_lerp(self.q_ref[lo], self.q_ref[hi], alpha),
            dq_ref=_lerp(self.dq_ref[lo], self.dq_ref[hi], alpha),
            action_ref=_lerp(self.action_ref[lo], self.action_ref[hi], alpha),
            joint_target_ref=_lerp(self.joint_target_ref[lo], self.joint_target_ref[hi], alpha),
            foot_pos_ref=_lerp(self.foot_pos_ref[lo], self.foot_pos_ref[hi], alpha),
            foot_vel_ref=_lerp(self.foot_vel_ref[lo], self.foot_vel_ref[hi], alpha),
            contact_ref=_lerp(self.contact_ref[lo], self.contact_ref[hi], alpha),
            base_pos_ref=_lerp(self.base_pos_ref[lo], self.base_pos_ref[hi], alpha),
            base_quat_ref=_normalize_quat(_lerp(self.base_quat_ref[lo], self.base_quat_ref[hi], alpha)),
            base_rpy_ref=_lerp(self.base_rpy_ref[lo], self.base_rpy_ref[hi], alpha),
            base_height_ref=float(_lerp(self.base_height_ref[lo], self.base_height_ref[hi], alpha)),
            base_vel_ref=_lerp(self.base_vel_ref[lo], self.base_vel_ref[hi], alpha),
            base_angvel_ref=_lerp(self.base_angvel_ref[lo], self.base_angvel_ref[hi], alpha),
            keypoint_pos_ref=_lerp(self.keypoint_pos_ref[lo], self.keypoint_pos_ref[hi], alpha),
            keypoint_vel_ref=_lerp(self.keypoint_vel_ref[lo], self.keypoint_vel_ref[hi], alpha),
        )

    @property
    def sample_count(self) -> int:
        return int(self.phase_ref.shape[0])


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _validate_reference_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    sample_count = int(arrays["phase_ref"].shape[0])
    if sample_count < 2:
        raise ValueError("nominal gait reference needs at least two samples")
    for name in REQUIRED_REFERENCE_ARRAYS:
        _validate_first_dim(name, arrays[name], sample_count)
    _validate_shape("q_ref", arrays["q_ref"], (sample_count, 12))
    _validate_shape("dq_ref", arrays["dq_ref"], (sample_count, 12))
    _validate_shape("action_ref", arrays["action_ref"], (sample_count, 12))
    _validate_shape("joint_target_ref", arrays["joint_target_ref"], (sample_count, 12))
    _validate_shape("foot_pos_ref", arrays["foot_pos_ref"], (sample_count, 4, 3))
    _validate_shape("foot_vel_ref", arrays["foot_vel_ref"], (sample_count, 4, 3))
    _validate_shape("contact_ref", arrays["contact_ref"], (sample_count, 4))
    _validate_shape("base_pos_ref", arrays["base_pos_ref"], (sample_count, 3))
    _validate_shape("base_quat_ref", arrays["base_quat_ref"], (sample_count, 4))
    _validate_shape("base_rpy_ref", arrays["base_rpy_ref"], (sample_count, 3))
    _validate_shape("base_height_ref", arrays["base_height_ref"], (sample_count,))
    _validate_shape("base_vel_ref", arrays["base_vel_ref"], (sample_count, 3))
    _validate_shape("base_angvel_ref", arrays["base_angvel_ref"], (sample_count, 3))
    return sample_count


def _validate_first_dim(name: str, value: np.ndarray, sample_count: int) -> None:
    if value.shape[:1] != (sample_count,):
        raise ValueError(
            f"{name} first dimension must match phase_ref length {sample_count}, got {value.shape}"
        )


def _validate_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def _periodic_interp_indices(phases: np.ndarray, phase: float) -> tuple[int, int, float]:
    count = int(phases.shape[0])
    if count == 1:
        return 0, 0, 0.0
    hi = int(np.searchsorted(phases, phase, side="right"))
    if hi == 0:
        lo = count - 1
        wrapped_hi_phase = float(phases[0]) + 1.0
        wrapped_phase = phase + 1.0
        span = max(1e-8, wrapped_hi_phase - float(phases[lo]))
        return lo, 0, (wrapped_phase - float(phases[lo])) / span
    if hi >= count:
        lo = count - 1
        span = max(1e-8, float(phases[0]) + 1.0 - float(phases[lo]))
        return lo, 0, (phase - float(phases[lo])) / span
    lo = hi - 1
    span = max(1e-8, float(phases[hi]) - float(phases[lo]))
    return lo, hi, (phase - float(phases[lo])) / span


def _lerp(a: np.ndarray | np.float32, b: np.ndarray | np.float32, alpha: float) -> np.ndarray:
    return (1.0 - float(alpha)) * a + float(alpha) * b


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray(quat / norm, dtype=np.float32)

