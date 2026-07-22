from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_Z_SAFE_MARGIN = 0.02


@dataclass(frozen=True)
class ReferenceAuditThresholds:
    min_mean_vx: float = 0.35
    max_mean_vx: float = 0.45
    max_mean_abs_vy: float = 0.08
    max_mean_abs_yaw_rate: float = 0.12
    min_base_z: float = 0.26
    max_joint_abs: float = 3.2
    max_action_rate_mean: float = 4.0
    min_foot_clearance: float = 0.015
    min_contact_duty: float = 0.10
    max_contact_duty: float = 0.90
    max_left_right_contact_delta: float = 0.35


@dataclass(frozen=True)
class ReferenceAuditReport:
    passed: bool
    metrics: dict[str, float]
    failures: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "failures": self.failures,
        }


def audit_nominal_gait_file(
    path: str | Path,
    *,
    thresholds: ReferenceAuditThresholds | None = None,
) -> ReferenceAuditReport:
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != "metadata"}
    return audit_nominal_gait_arrays(arrays, thresholds=thresholds)


def audit_nominal_gait_arrays(
    arrays: Mapping[str, Any],
    *,
    thresholds: ReferenceAuditThresholds | None = None,
) -> ReferenceAuditReport:
    thresholds = thresholds or ReferenceAuditThresholds()
    base_vel = _required_array(arrays, "base_vel_ref")
    base_angvel = _required_array(arrays, "base_angvel_ref")
    base_height = _required_array(arrays, "base_height_ref")
    q_ref = _required_array(arrays, "q_ref")
    action_ref = _required_array(arrays, "action_ref")
    foot_pos = _required_array(arrays, "foot_pos_ref")
    contact_ref = _required_array(arrays, "contact_ref")

    metrics = {
        "samples": float(base_vel.shape[0]),
        "mean_vx": float(np.mean(base_vel[:, 0])),
        "mean_abs_vy": float(np.mean(np.abs(base_vel[:, 1]))),
        "mean_abs_yaw_rate": float(np.mean(np.abs(base_angvel[:, 2]))),
        "base_z_min": float(np.min(base_height)),
        "base_z_p05": float(np.percentile(base_height, 5.0)),
        "base_z_mean": float(np.mean(base_height)),
        "recommended_anti_collapse_z_safe": float(
            np.percentile(base_height, 5.0) - DEFAULT_Z_SAFE_MARGIN
        ),
        "joint_abs_max": float(np.max(np.abs(q_ref))) if q_ref.size else 0.0,
        "action_rate_mean": _mean_action_rate(action_ref),
        "foot_clearance_max": float(np.max(foot_pos[:, :, 2])) if foot_pos.size else 0.0,
    }
    contact_duty = np.mean(contact_ref, axis=0) if contact_ref.size else np.zeros(4, dtype=np.float32)
    metrics.update({
        "contact_duty_fr": float(contact_duty[0]),
        "contact_duty_fl": float(contact_duty[1]),
        "contact_duty_rr": float(contact_duty[2]),
        "contact_duty_rl": float(contact_duty[3]),
        "left_right_contact_delta": float(abs((contact_duty[1] + contact_duty[3]) - (contact_duty[0] + contact_duty[2])) / 2.0),
        "diagonal_contact_delta": float(((contact_duty[0] + contact_duty[3]) - (contact_duty[1] + contact_duty[2])) / 2.0),
    })

    failures: dict[str, str] = {}
    _check_range(
        failures,
        "mean_vx",
        metrics["mean_vx"],
        lower=thresholds.min_mean_vx,
        upper=thresholds.max_mean_vx,
    )
    _check_upper(failures, "mean_abs_vy", metrics["mean_abs_vy"], thresholds.max_mean_abs_vy)
    _check_upper(
        failures,
        "mean_abs_yaw_rate",
        metrics["mean_abs_yaw_rate"],
        thresholds.max_mean_abs_yaw_rate,
    )
    _check_lower(failures, "base_z_min", metrics["base_z_min"], thresholds.min_base_z)
    _check_upper(failures, "joint_abs_max", metrics["joint_abs_max"], thresholds.max_joint_abs)
    _check_upper(failures, "action_rate_mean", metrics["action_rate_mean"], thresholds.max_action_rate_mean)
    _check_lower(failures, "foot_clearance_max", metrics["foot_clearance_max"], thresholds.min_foot_clearance)
    for key in ("contact_duty_fr", "contact_duty_fl", "contact_duty_rr", "contact_duty_rl"):
        _check_range(
            failures,
            key,
            metrics[key],
            lower=thresholds.min_contact_duty,
            upper=thresholds.max_contact_duty,
        )
    _check_upper(
        failures,
        "left_right_contact_delta",
        metrics["left_right_contact_delta"],
        thresholds.max_left_right_contact_delta,
    )
    return ReferenceAuditReport(passed=not failures, metrics=metrics, failures=failures)


def write_audit_report(path: str | Path, report: ReferenceAuditReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _required_array(arrays: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"reference audit missing required array: {name}")
    return np.asarray(arrays[name], dtype=np.float32)


def _mean_action_rate(action_ref: np.ndarray) -> float:
    if action_ref.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(action_ref, axis=0), axis=1)))


def _check_lower(failures: dict[str, str], name: str, value: float, lower: float) -> None:
    if value < lower:
        failures[name] = f"{value:.6g} < {lower:.6g}"


def _check_upper(failures: dict[str, str], name: str, value: float, upper: float) -> None:
    if value > upper:
        failures[name] = f"{value:.6g} > {upper:.6g}"


def _check_range(
    failures: dict[str, str],
    name: str,
    value: float,
    *,
    lower: float,
    upper: float,
) -> None:
    if value < lower or value > upper:
        failures[name] = f"{value:.6g} outside [{lower:.6g}, {upper:.6g}]"
