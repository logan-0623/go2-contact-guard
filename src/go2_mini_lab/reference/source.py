from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .nominal_gait import NominalGaitReference


@dataclass(frozen=True)
class ReferenceSourceSummary:
    kind: str
    label: str
    artifact: str | None


@dataclass(frozen=True)
class ReferenceFrame:
    phase: float
    q: np.ndarray
    dq: np.ndarray
    action: np.ndarray
    joint_target: np.ndarray
    foot_pos: np.ndarray
    foot_vel: np.ndarray
    contact: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray
    base_rpy: np.ndarray
    base_height: float
    base_vel: np.ndarray
    base_angvel: np.ndarray
    keypoint_pos: np.ndarray
    keypoint_vel: np.ndarray


class RolloutReferenceSource:
    def __init__(self, reference: NominalGaitReference) -> None:
        self.reference = reference
        self.summary = reference_source_summary(reference.metadata)

    def sample(self, phase: float) -> ReferenceFrame:
        sample = self.reference.sample(phase)
        return ReferenceFrame(
            phase=sample.phase,
            q=sample.q_ref,
            dq=sample.dq_ref,
            action=sample.action_ref,
            joint_target=sample.joint_target_ref,
            foot_pos=sample.foot_pos_ref,
            foot_vel=sample.foot_vel_ref,
            contact=sample.contact_ref,
            base_pos=sample.base_pos_ref,
            base_quat=sample.base_quat_ref,
            base_rpy=sample.base_rpy_ref,
            base_height=sample.base_height_ref,
            base_vel=sample.base_vel_ref,
            base_angvel=sample.base_angvel_ref,
            keypoint_pos=sample.keypoint_pos_ref,
            keypoint_vel=sample.keypoint_vel_ref,
        )


def reference_source_summary(metadata: Mapping[str, Any]) -> ReferenceSourceSummary:
    source = str(metadata.get("source") or "")
    policy_format = str(metadata.get("policy_format") or "")
    reference_policy = metadata.get("reference_policy")
    source_checkpoint = metadata.get("source_checkpoint")

    if policy_format == "onnx" or source == "onnx_rollout":
        return ReferenceSourceSummary(
            kind="onnx_policy_rollout",
            label="ONNX policy rollout",
            artifact=str(reference_policy or metadata.get("policy") or "") or None,
        )
    if source == "ppo_rollout":
        return ReferenceSourceSummary(
            kind="ppo_rollout_self",
            label="self PPO rollout",
            artifact=str(source_checkpoint or "") or None,
        )
    if source == "unitree_controller_rollout":
        return ReferenceSourceSummary(
            kind="unitree_controller_rollout",
            label="Unitree controller rollout",
            artifact=str(metadata.get("controller") or "") or None,
        )
    return ReferenceSourceSummary(
        kind=source or "unknown",
        label=source or "unknown reference source",
        artifact=str(source_checkpoint or reference_policy or "") or None,
    )
