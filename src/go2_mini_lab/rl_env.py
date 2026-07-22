from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .controller import GaitConfig, PDConfig, TrotGaitController, compute_pd_torque
from .history_event_estimator import (
    LinearHistoryEventEstimator,
    build_history_feature,
    build_proprioceptive_feature,
)
from .mujoco_rollout import (
    _find_actuators_for_joints,
    _find_joint_addresses,
    _foot_contacts,
    _set_initial_pose,
)
from .trajectory import BaseState


RL_TASKS = ("stand", "slow_walk", "velocity_flat", "push_recovery")
VELOCITY_TASKS = ("slow_walk", "velocity_flat")
FEET = ("FR", "FL", "RR", "RL")
POLICY_LEG_ORDER_PROFILES = {
    "actuator": FEET,
    "mjcf": ("FL", "FR", "RL", "RR"),
}
DEFAULT_SLOW_WALK_VELOCITY = 0.18
MJLAB_BAD_ORIENTATION_LIMIT_RAD = 0.8726646259971648
EXTERNAL_FORCE_MODES = ("constant", "spring")
EXTERNAL_FORCE_DIRECTION_MODES = ("default", "lateral_bimodal", "lateral_mixed", "yielding_tail_mixed")
EXTERNAL_FORCE_SPRING_TYPES = ("resistive", "guiding")
FORCE_IMPEDANCE_MODES = ("off", "onset", "active", "two_phase")
FORCE_IMPEDANCE_JOINT_SCOPES = ("all", "hip", "thigh", "calf", "hip_calf", "stance_hip_calf")
FORCE_YIELDING_COMMAND_MODES = ("scaled", "unit", "unit_pulse")
FORCE_REFERENCE_GOVERNOR_MODES = ("off", "onset", "active", "two_phase")
FORCE_RESPONSE_ROUTER_MODES = ("off", "semantic_oracle", "deployable_semantic")
FORCE_RESPONSE_PROFILES = ("body_yield_foot_guard",)
FORCE_SAFETY_TRIGGER_SOURCES = (
    "oracle",
    "deployable",
    "deployable_v2",
    "deployable_history",
    "deployable_hybrid",
    "deployable_history_or_v2",
    "deployable_history_gated_v2",
    "oracle_or_deployable",
    "oracle_or_deployable_v2",
    "oracle_or_deployable_history",
    "oracle_or_deployable_hybrid",
    "oracle_or_deployable_history_or_v2",
    "oracle_or_deployable_history_gated_v2",
)
ORACLE_FORCE_SAFETY_TRIGGER_SOURCES = {
    "oracle",
    "oracle_or_deployable",
    "oracle_or_deployable_v2",
    "oracle_or_deployable_history",
    "oracle_or_deployable_hybrid",
    "oracle_or_deployable_history_or_v2",
    "oracle_or_deployable_history_gated_v2",
}
DEPLOYABLE_FORCE_SAFETY_TRIGGER_SOURCES = {
    "deployable",
    "oracle_or_deployable",
}
DEPLOYABLE_V2_FORCE_SAFETY_TRIGGER_SOURCES = {
    "deployable_v2",
    "oracle_or_deployable_v2",
}
DEPLOYABLE_HISTORY_FORCE_SAFETY_TRIGGER_SOURCES = {
    "deployable_history",
    "oracle_or_deployable_history",
}
DEPLOYABLE_HYBRID_FORCE_SAFETY_TRIGGER_SOURCES = {
    "deployable_hybrid",
    "oracle_or_deployable_hybrid",
}
DEPLOYABLE_HISTORY_OR_V2_FORCE_SAFETY_TRIGGER_SOURCES = {
    "deployable_history_or_v2",
    "oracle_or_deployable_history_or_v2",
}
DEPLOYABLE_HISTORY_GATED_V2_FORCE_SAFETY_TRIGGER_SOURCES = {
    "deployable_history_gated_v2",
    "oracle_or_deployable_history_gated_v2",
}
ANY_DEPLOYABLE_FORCE_SAFETY_TRIGGER_SOURCES = (
    DEPLOYABLE_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_V2_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_HISTORY_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_HYBRID_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_HISTORY_OR_V2_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_HISTORY_GATED_V2_FORCE_SAFETY_TRIGGER_SOURCES
)
HISTORY_ESTIMATOR_FORCE_SAFETY_TRIGGER_SOURCES = (
    DEPLOYABLE_HISTORY_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_HYBRID_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_HISTORY_OR_V2_FORCE_SAFETY_TRIGGER_SOURCES
    | DEPLOYABLE_HISTORY_GATED_V2_FORCE_SAFETY_TRIGGER_SOURCES
)
FORCE_SAFETY_DETECTOR_V2_BASELINE_ALPHA = 0.12
FORCE_SAFETY_DETECTOR_V2_WARMUP_STEPS = 10
FORCE_SAFETY_DETECTOR_V2_RELEASE_RATIO = 0.55
FORCE_SAFETY_HISTORY_GATED_V2_RATIO = 0.85
VELOCITY_COMMAND_FRAMES = ("world", "body")
OBSERVATION_MODES = ("policy", "privileged")
POLICY_ACTION_MODES = ("full_action", "onnx_residual", "onnx_safety_layer")
ONNX_POLICY_ACTION_MODES = ("onnx_residual", "onnx_safety_layer")
SAFETY_LAYER_ACTION_NAMES = (
    "safety_kp_yield",
    "safety_kd_boost",
    "safety_governor_gain",
    "safety_governor_clip",
)
FULLBODY_REFERENCE_MODES = ("static", "phase_trot", "bounded_compliant", "rollout", "rollout_compliant")
FULLBODY_TRACKING_KEYPOINTS = (
    "base_link",
    "FR_hip",
    "FL_hip",
    "RR_hip",
    "RL_hip",
    "FR_calf",
    "FL_calf",
    "RR_calf",
    "RL_calf",
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
)


@dataclass(frozen=True)
class BoundedCompliantReferenceStep:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    motion_velocity: tuple[float, float, float]
    offset: tuple[float, float, float]
    compliant_velocity: tuple[float, float, float]
    driving_force: tuple[float, float, float]
    interaction_force: tuple[float, float, float]


@dataclass(frozen=True)
class ForceReferenceGovernorStep:
    offset: tuple[float, float, float]
    velocity: tuple[float, float, float]
    gate: float


@dataclass(frozen=True)
class ForceSafetyDetectorStep:
    score: float
    active: bool
    force_proxy: tuple[float, float, float]


@dataclass(frozen=True)
class ForceResponseRouterStep:
    response_class: str
    impacted_body: str | None
    impacted_leg: str | None
    impacted_leg_is_stance: bool | None
    governor_enabled: bool
    joint_kp_scales: tuple[float, ...]
    joint_kd_scales: tuple[float, ...]


def _integrate_bounded_compliant_reference(
    *,
    motion_target: Any,
    previous_motion_target: Any,
    reference_position: Any,
    reference_velocity: Any,
    interaction_force: Any,
    dt: float,
    admittance: float,
    stiffness: float,
    damping: float,
    offset_clip: float,
    velocity_clip: float,
    driving_force_limit: float = 0.0,
    horizontal_only: bool = False,
    z_offset_clip: float | None = None,
    z_velocity_clip: float | None = None,
) -> BoundedCompliantReferenceStep:
    dt = max(1e-6, float(dt))
    motion_target = _tuple3(motion_target)
    previous_motion_target = _tuple3(previous_motion_target)
    reference_position = _tuple3(reference_position)
    reference_velocity = _tuple3(reference_velocity)
    interaction_force = _tuple3(interaction_force)
    if horizontal_only:
        interaction_force = (interaction_force[0], interaction_force[1], 0.0)

    motion_velocity = _vec_scale(_vec_sub(motion_target, previous_motion_target), 1.0 / dt)
    driving_force = _vec_add(
        _vec_scale(_vec_sub(motion_target, reference_position), max(0.0, float(stiffness))),
        _vec_scale(_vec_sub(motion_velocity, reference_velocity), max(0.0, float(damping))),
    )
    if driving_force_limit > 0.0:
        driving_force = _clip_tuple_norm(driving_force, driving_force_limit)
    acceleration = _vec_add(
        driving_force,
        _vec_scale(interaction_force, max(0.0, float(admittance))),
    )
    next_velocity = _vec_add(reference_velocity, _vec_scale(acceleration, dt))
    compliant_velocity = _vec_sub(next_velocity, motion_velocity)
    if velocity_clip > 0.0:
        compliant_velocity = _clip_tuple_norm(compliant_velocity, velocity_clip)
        next_velocity = _vec_add(motion_velocity, compliant_velocity)
    if horizontal_only or z_velocity_clip is not None:
        z_limit = 0.0 if z_velocity_clip is None else max(0.0, float(z_velocity_clip))
        clipped_z = max(-z_limit, min(z_limit, compliant_velocity[2]))
        compliant_velocity = (compliant_velocity[0], compliant_velocity[1], clipped_z)
        next_velocity = _vec_add(motion_velocity, compliant_velocity)

    next_position = _vec_add(reference_position, _vec_scale(next_velocity, dt))
    offset = _vec_sub(next_position, motion_target)
    if offset_clip > 0.0:
        offset = _clip_tuple_norm(offset, offset_clip)
        next_position = _vec_add(motion_target, offset)
    if horizontal_only or z_offset_clip is not None:
        z_limit = 0.0 if z_offset_clip is None else max(0.0, float(z_offset_clip))
        clipped_z = max(-z_limit, min(z_limit, offset[2]))
        offset = (offset[0], offset[1], clipped_z)
        next_position = _vec_add(motion_target, offset)
    compliant_velocity = _vec_sub(next_velocity, motion_velocity)

    return BoundedCompliantReferenceStep(
        position=next_position,
        velocity=next_velocity,
        motion_velocity=motion_velocity,
        offset=offset,
        compliant_velocity=compliant_velocity,
        driving_force=driving_force,
        interaction_force=interaction_force,
    )


def _sample_external_force_safe_limit(
    rng: Any,
    *,
    min_n: float,
    max_n: float,
) -> float:
    min_n = float(min_n)
    max_n = float(max_n)
    if max_n <= 0.0:
        return 0.0
    if min_n <= 0.0 or max_n < min_n:
        raise ValueError("external force safe limit requires 0 < min_n <= max_n")
    return float(rng.uniform(min_n, max_n))


def _normalize_external_force_safe_limit(value_n: float, *, max_n: float) -> float:
    max_n = float(max_n)
    if max_n <= 0.0:
        return 0.0
    return max(0.0, min(float(value_n) / max_n, 1.0))


def _force_excess_cost(*, force_n: float, safe_limit_n: float, margin_n: float) -> float:
    safe_limit_n = max(1e-6, float(safe_limit_n))
    allowed_force_n = safe_limit_n + max(0.0, float(margin_n))
    excess_n = max(0.0, float(force_n) - allowed_force_n)
    return (excess_n / safe_limit_n) ** 2


def _force_step_is_compliant(*, force_n: float, safe_limit_n: float, margin_n: float) -> bool:
    return float(force_n) <= float(safe_limit_n) + max(0.0, float(margin_n)) + 1e-9


def _force_yielding_command_value(
    *,
    np_module: Any,
    command: Any,
    force_tracking_frame: Any,
    active: bool,
    mode: str = "scaled",
    velocity_per_n: float,
    velocity_clip: float,
    elapsed_since_force_start_s: float | None = None,
    pulse_start_s: float = 0.10,
    pulse_duration_s: float = 0.20,
    pulse_recovery_s: float = 0.10,
    pulse_post_clip: float = 0.0,
) -> Any:
    command_array = np_module.asarray(command, dtype=np_module.float64).copy()
    if not active:
        return command_array
    gain = float(velocity_per_n)
    clip_mps = max(0.0, float(velocity_clip))
    if gain == 0.0 or clip_mps <= 0.0:
        return command_array
    mode = str(mode)
    if mode not in FORCE_YIELDING_COMMAND_MODES:
        choices = ", ".join(FORCE_YIELDING_COMMAND_MODES)
        raise ValueError(f"unknown force yielding command mode {mode!r}; choose one of: {choices}")

    force = np_module.asarray(force_tracking_frame, dtype=np_module.float64)
    if len(command_array) < 2 or len(force) < 2:
        return command_array
    force_xy = np_module.asarray(force[:2], dtype=np_module.float64)
    if mode in {"unit", "unit_pulse"}:
        force_norm = float(np_module.linalg.norm(force_xy))
        if force_norm <= 1e-12:
            return command_array
        active_clip_mps = clip_mps
        if mode == "unit_pulse":
            if elapsed_since_force_start_s is None:
                return command_array
            elapsed_s = max(0.0, float(elapsed_since_force_start_s))
            start_s = max(0.0, float(pulse_start_s))
            duration_s = max(0.0, float(pulse_duration_s))
            recovery_s = max(0.0, float(pulse_recovery_s))
            post_clip_mps = max(0.0, float(pulse_post_clip))
            hold_end_s = start_s + duration_s
            if elapsed_s < start_s:
                active_clip_mps = 0.0
            elif elapsed_s <= hold_end_s:
                active_clip_mps = clip_mps
            elif recovery_s > 0.0 and elapsed_s <= hold_end_s + recovery_s:
                alpha = (elapsed_s - hold_end_s) / recovery_s
                active_clip_mps = clip_mps + alpha * (post_clip_mps - clip_mps)
            else:
                active_clip_mps = post_clip_mps
            if active_clip_mps <= 0.0:
                return command_array
        offset_xy = (1.0 if gain > 0.0 else -1.0) * active_clip_mps * force_xy / force_norm
    else:
        offset_xy = gain * force_xy
        offset_xy = _clip_vector_norm(offset_xy, clip_mps, np_module)
    command_array[:2] += offset_xy
    return command_array


def _force_impedance_scale_value(
    *,
    t_s: float,
    windows: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    mode: str,
    kp_scale: float,
    kd_scale: float,
    delay_s: float = 0.0,
    hold_s: float,
    recovery_s: float,
    tail_kp_scale: float = 1.0,
    tail_kd_scale: float = 1.0,
) -> tuple[float, float]:
    mode = str(mode)
    if mode not in FORCE_IMPEDANCE_MODES:
        choices = ", ".join(FORCE_IMPEDANCE_MODES)
        raise ValueError(f"unknown force impedance mode {mode!r}; choose one of: {choices}")
    if mode == "off":
        return 1.0, 1.0

    target_kp = max(0.0, float(kp_scale))
    target_kd = max(0.0, float(kd_scale))
    tail_kp = max(0.0, float(tail_kp_scale))
    tail_kd = max(0.0, float(tail_kd_scale))
    delay_s = max(0.0, float(delay_s))
    hold_s = max(0.0, float(hold_s))
    recovery_s = max(0.0, float(recovery_s))
    t_s = float(t_s)

    best_kp = 1.0
    best_kd = 1.0
    for start_s, end_s in windows:
        start_s = float(start_s) + delay_s
        end_s = float(end_s)
        if end_s < start_s:
            continue
        hold_end_s = end_s if mode == "active" else min(end_s, start_s + hold_s)
        if t_s < start_s:
            continue
        if mode == "two_phase" and t_s > hold_end_s and t_s <= end_s:
            current_kp = tail_kp
            current_kd = tail_kd
        elif t_s <= hold_end_s:
            current_kp = target_kp
            current_kd = target_kd
        elif recovery_s > 0.0 and t_s <= (end_s if mode == "two_phase" else hold_end_s) + recovery_s:
            recovery_start_kp = tail_kp if mode == "two_phase" else target_kp
            recovery_start_kd = tail_kd if mode == "two_phase" else target_kd
            recovery_start_s = end_s if mode == "two_phase" else hold_end_s
            if t_s < recovery_start_s:
                continue
            alpha = (t_s - recovery_start_s) / recovery_s
            current_kp = recovery_start_kp + alpha * (1.0 - recovery_start_kp)
            current_kd = recovery_start_kd + alpha * (1.0 - recovery_start_kd)
        else:
            continue
        best_kp = min(best_kp, current_kp)
        best_kd = current_kd if current_kp <= best_kp else best_kd
    return best_kp, best_kd


def _force_impedance_joint_is_selected(
    joint_name: str,
    scope: str,
    contacts: dict[str, bool] | None = None,
) -> bool:
    scope = str(scope)
    if scope not in FORCE_IMPEDANCE_JOINT_SCOPES:
        choices = ", ".join(FORCE_IMPEDANCE_JOINT_SCOPES)
        raise ValueError(f"unknown force impedance joint scope {scope!r}; choose one of: {choices}")
    if scope == "all":
        return True
    name = str(joint_name)
    if scope == "hip_calf":
        return name.endswith("_hip_joint") or name.endswith("_calf_joint")
    if scope == "stance_hip_calf":
        leg = name.split("_", 1)[0]
        in_stance = bool((contacts or {}).get(leg, False))
        return in_stance and (name.endswith("_hip_joint") or name.endswith("_calf_joint"))
    return name.endswith(f"_{scope}_joint")


def _force_response_nominal_step(joint_names: tuple[str, ...]) -> ForceResponseRouterStep:
    return ForceResponseRouterStep(
        response_class="nominal",
        impacted_body=None,
        impacted_leg=None,
        impacted_leg_is_stance=None,
        governor_enabled=False,
        joint_kp_scales=tuple(1.0 for _ in joint_names),
        joint_kd_scales=tuple(1.0 for _ in joint_names),
    )


def _force_response_first_body_name(external_force_body: str | None) -> str | None:
    if external_force_body is None:
        return None
    for name in str(external_force_body).split(","):
        stripped = name.strip()
        if stripped:
            return stripped
    return None


def _force_response_leg_from_body_name(body_name: str | None) -> str | None:
    if not body_name:
        return None
    prefix = str(body_name).split("_", 1)[0]
    return prefix if prefix in FEET else None


def _force_response_body_kind(body_name: str | None) -> str:
    if not body_name or body_name == "base_link":
        return "body"
    if str(body_name).endswith("_foot"):
        return "foot"
    if _force_response_leg_from_body_name(body_name) is not None:
        return "leg"
    return "body"


def _force_response_joint_leg(joint_name: str) -> str | None:
    prefix = str(joint_name).split("_", 1)[0]
    return prefix if prefix in FEET else None


def _force_response_joint_is_hip_calf(joint_name: str) -> bool:
    return str(joint_name).endswith("_hip_joint") or str(joint_name).endswith("_calf_joint")


def _force_response_phase_scales_value(
    *,
    t_s: float,
    windows: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    mode: str,
    kp_scale: float,
    kd_scale: float,
    delay_s: float,
    hold_s: float,
    recovery_s: float,
    tail_kp_scale: float = 1.0,
    tail_kd_scale: float = 1.0,
) -> tuple[float, float, bool]:
    mode = str(mode)
    if mode not in FORCE_IMPEDANCE_MODES:
        choices = ", ".join(FORCE_IMPEDANCE_MODES)
        raise ValueError(f"unknown force impedance mode {mode!r}; choose one of: {choices}")
    if mode == "off":
        return 1.0, 1.0, False

    t_s = float(t_s)
    delay_s = max(0.0, float(delay_s))
    hold_s = max(0.0, float(hold_s))
    recovery_s = max(0.0, float(recovery_s))
    target_kp = max(0.0, float(kp_scale))
    target_kd = max(0.0, float(kd_scale))
    tail_kp = max(0.0, float(tail_kp_scale))
    tail_kd = max(0.0, float(tail_kd_scale))

    best_kp = 1.0
    best_kd = 1.0
    best_delta = 0.0
    for start_s, end_s in windows:
        start_s = float(start_s) + delay_s
        end_s = float(end_s)
        if end_s < start_s or t_s < start_s:
            continue
        hold_end_s = end_s if mode == "active" else min(end_s, start_s + hold_s)
        if mode == "two_phase" and t_s > hold_end_s and t_s <= end_s:
            current_kp = tail_kp
            current_kd = tail_kd
        elif t_s <= hold_end_s:
            current_kp = target_kp
            current_kd = target_kd
        elif recovery_s > 0.0 and t_s <= (end_s if mode == "two_phase" else hold_end_s) + recovery_s:
            recovery_start_s = end_s if mode == "two_phase" else hold_end_s
            if t_s < recovery_start_s:
                continue
            recovery_start_kp = tail_kp if mode == "two_phase" else target_kp
            recovery_start_kd = tail_kd if mode == "two_phase" else target_kd
            alpha = (t_s - recovery_start_s) / recovery_s
            current_kp = recovery_start_kp + alpha * (1.0 - recovery_start_kp)
            current_kd = recovery_start_kd + alpha * (1.0 - recovery_start_kd)
        else:
            continue

        delta = max(abs(current_kp - 1.0), abs(current_kd - 1.0))
        if delta >= best_delta:
            best_kp = current_kp
            best_kd = current_kd
            best_delta = delta
    return best_kp, best_kd, bool(best_delta > 1e-9)


def _semantic_force_response_router_value(
    *,
    t_s: float,
    windows: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    mode: str,
    profile: str,
    external_force_body: str | None,
    contacts: dict[str, bool] | None,
    joint_names: tuple[str, ...],
    base_joint_scope: str,
    base_kp_scale: float,
    base_kd_scale: float,
    foot_kp_scale: float,
    foot_kd_scale: float,
    delay_s: float,
    hold_s: float,
    recovery_s: float,
    tail_kp_scale: float,
    tail_kd_scale: float,
) -> ForceResponseRouterStep:
    mode = str(mode)
    if mode not in FORCE_RESPONSE_ROUTER_MODES:
        choices = ", ".join(FORCE_RESPONSE_ROUTER_MODES)
        raise ValueError(f"unknown force response router mode {mode!r}; choose one of: {choices}")
    profile = str(profile)
    if profile not in FORCE_RESPONSE_PROFILES:
        choices = ", ".join(FORCE_RESPONSE_PROFILES)
        raise ValueError(f"unknown force response profile {profile!r}; choose one of: {choices}")
    joint_names = tuple(str(name) for name in joint_names)
    if mode == "off":
        return _force_response_nominal_step(joint_names)

    body_name = _force_response_first_body_name(external_force_body)
    body_kind = _force_response_body_kind(body_name)
    impacted_leg = _force_response_leg_from_body_name(body_name)
    contact_map = contacts or {}
    impacted_leg_is_stance = (
        bool(contact_map.get(impacted_leg, False)) if impacted_leg is not None else None
    )

    if body_kind == "foot":
        phase_kp, phase_kd, phase_active = _force_response_phase_scales_value(
            t_s=t_s,
            windows=windows,
            mode="onset",
            kp_scale=foot_kp_scale,
            kd_scale=foot_kd_scale,
            delay_s=delay_s,
            hold_s=hold_s,
            recovery_s=recovery_s,
        )
    else:
        phase_kp, phase_kd, phase_active = _force_response_phase_scales_value(
            t_s=t_s,
            windows=windows,
            mode="onset",
            kp_scale=base_kp_scale,
            kd_scale=base_kd_scale,
            delay_s=delay_s,
            hold_s=hold_s,
            recovery_s=recovery_s,
            tail_kp_scale=tail_kp_scale,
            tail_kd_scale=tail_kd_scale,
        )
    if not phase_active:
        return _force_response_nominal_step(joint_names)

    joint_kp_scales: list[float] = []
    joint_kd_scales: list[float] = []
    response_class = "body_yield"
    governor_enabled = True

    if body_kind == "foot" and impacted_leg_is_stance:
        response_class = "stance_foot_guard"
        governor_enabled = False
        for joint_name in joint_names:
            joint_leg = _force_response_joint_leg(joint_name)
            selected = bool(contact_map.get(joint_leg, False)) and _force_response_joint_is_hip_calf(joint_name)
            joint_kp_scales.append(float(phase_kp) if selected else 1.0)
            joint_kd_scales.append(float(phase_kd) if selected else 1.0)
    elif body_kind == "foot":
        response_class = "swing_foot_restep"
        governor_enabled = False
        for joint_name in joint_names:
            selected = _force_response_joint_leg(joint_name) == impacted_leg
            joint_kp_scales.append(1.0)
            joint_kd_scales.append(float(phase_kd) if selected else 1.0)
    elif body_kind == "leg" and impacted_leg is not None:
        response_class = "leg_collision_guard"
        governor_enabled = True
        for joint_name in joint_names:
            selected = _force_response_joint_leg(joint_name) == impacted_leg and _force_response_joint_is_hip_calf(joint_name)
            joint_kp_scales.append(float(phase_kp) if selected else 1.0)
            joint_kd_scales.append(float(phase_kd) if selected else 1.0)
    else:
        for joint_name in joint_names:
            selected = _force_impedance_joint_is_selected(joint_name, base_joint_scope, contact_map)
            joint_kp_scales.append(float(phase_kp) if selected else 1.0)
            joint_kd_scales.append(float(phase_kd) if selected else 1.0)

    return ForceResponseRouterStep(
        response_class=response_class,
        impacted_body=body_name,
        impacted_leg=impacted_leg,
        impacted_leg_is_stance=impacted_leg_is_stance,
        governor_enabled=governor_enabled,
        joint_kp_scales=tuple(joint_kp_scales),
        joint_kd_scales=tuple(joint_kd_scales),
    )


def _force_safety_detector_step_value(
    *,
    np_module: Any,
    base_linear_acceleration: Any,
    base_angular_acceleration: Any,
    max_joint_tracking_error: float,
    max_joint_velocity: float,
    contact_count: int,
    previous_contact_count: int,
    linear_acceleration_threshold: float,
    angular_acceleration_threshold: float,
    joint_tracking_error_threshold: float,
    joint_velocity_threshold: float,
    contact_loss_enabled: bool,
) -> ForceSafetyDetectorStep:
    linear_acceleration = np_module.asarray(base_linear_acceleration, dtype=np_module.float64)
    angular_acceleration = np_module.asarray(base_angular_acceleration, dtype=np_module.float64)
    if len(linear_acceleration) < 3:
        linear_acceleration = np_module.pad(linear_acceleration, (0, 3 - len(linear_acceleration)))
    if len(angular_acceleration) < 3:
        angular_acceleration = np_module.pad(angular_acceleration, (0, 3 - len(angular_acceleration)))

    score = 0.0
    linear_threshold = max(0.0, float(linear_acceleration_threshold))
    angular_threshold = max(0.0, float(angular_acceleration_threshold))
    joint_error_threshold = max(0.0, float(joint_tracking_error_threshold))
    joint_velocity_threshold = max(0.0, float(joint_velocity_threshold))
    if linear_threshold > 0.0:
        score = max(score, float(np_module.linalg.norm(linear_acceleration[:3])) / linear_threshold)
    if angular_threshold > 0.0:
        score = max(score, float(np_module.linalg.norm(angular_acceleration[:3])) / angular_threshold)
    if joint_error_threshold > 0.0:
        score = max(score, abs(float(max_joint_tracking_error)) / joint_error_threshold)
    if joint_velocity_threshold > 0.0:
        score = max(score, abs(float(max_joint_velocity)) / joint_velocity_threshold)
    if contact_loss_enabled and int(previous_contact_count) >= 2 and int(contact_count) <= 1:
        score = max(score, 1.0)

    force_proxy = tuple(float(v) for v in linear_acceleration[:3])
    return ForceSafetyDetectorStep(
        score=float(score),
        active=bool(score >= 1.0),
        force_proxy=force_proxy,
    )


def _force_safety_detector_v2_step_value(
    *,
    np_module: Any,
    base_linear_acceleration: Any,
    base_angular_acceleration: Any,
    linear_acceleration_baseline: Any,
    angular_acceleration_baseline: Any,
    previous_active: bool,
    warmup: bool,
    linear_acceleration_threshold: float,
    angular_acceleration_threshold: float,
    baseline_alpha: float = FORCE_SAFETY_DETECTOR_V2_BASELINE_ALPHA,
) -> tuple[ForceSafetyDetectorStep, tuple[float, float, float], tuple[float, float, float]]:
    linear_acceleration = np_module.asarray(base_linear_acceleration, dtype=np_module.float64)
    angular_acceleration = np_module.asarray(base_angular_acceleration, dtype=np_module.float64)
    linear_baseline = np_module.asarray(linear_acceleration_baseline, dtype=np_module.float64)
    angular_baseline = np_module.asarray(angular_acceleration_baseline, dtype=np_module.float64)
    if len(linear_acceleration) < 3:
        linear_acceleration = np_module.pad(linear_acceleration, (0, 3 - len(linear_acceleration)))
    if len(angular_acceleration) < 3:
        angular_acceleration = np_module.pad(angular_acceleration, (0, 3 - len(angular_acceleration)))
    if len(linear_baseline) < 3:
        linear_baseline = np_module.pad(linear_baseline, (0, 3 - len(linear_baseline)))
    if len(angular_baseline) < 3:
        angular_baseline = np_module.pad(angular_baseline, (0, 3 - len(angular_baseline)))

    alpha = max(0.0, min(1.0, float(baseline_alpha)))
    if bool(warmup):
        updated_linear = (1.0 - alpha) * linear_baseline[:3] + alpha * linear_acceleration[:3]
        updated_angular = (1.0 - alpha) * angular_baseline[:3] + alpha * angular_acceleration[:3]
        if float(np_module.linalg.norm(linear_baseline[:3])) <= 1e-9:
            updated_linear = linear_acceleration[:3]
        if float(np_module.linalg.norm(angular_baseline[:3])) <= 1e-9:
            updated_angular = angular_acceleration[:3]
        return (
            ForceSafetyDetectorStep(score=0.0, active=False, force_proxy=(0.0, 0.0, 0.0)),
            tuple(float(v) for v in updated_linear),
            tuple(float(v) for v in updated_angular),
        )

    linear_residual = linear_acceleration[:3] - linear_baseline[:3]
    angular_residual = angular_acceleration[:3] - angular_baseline[:3]
    score = 0.0
    linear_threshold = max(0.0, float(linear_acceleration_threshold))
    angular_threshold = max(0.0, float(angular_acceleration_threshold))
    if linear_threshold > 0.0:
        score = max(score, float(np_module.linalg.norm(linear_residual)) / linear_threshold)
    if angular_threshold > 0.0:
        score = max(score, float(np_module.linalg.norm(angular_residual)) / angular_threshold)
    trigger_score = FORCE_SAFETY_DETECTOR_V2_RELEASE_RATIO if bool(previous_active) else 1.0
    active = bool(score >= trigger_score)

    if active:
        updated_linear = linear_baseline[:3]
        updated_angular = angular_baseline[:3]
    else:
        updated_linear = (1.0 - alpha) * linear_baseline[:3] + alpha * linear_acceleration[:3]
        updated_angular = (1.0 - alpha) * angular_baseline[:3] + alpha * angular_acceleration[:3]
    return (
        ForceSafetyDetectorStep(
            score=float(score),
            active=active,
            force_proxy=tuple(float(v) for v in linear_residual),
        ),
        tuple(float(v) for v in updated_linear),
        tuple(float(v) for v in updated_angular),
    )


def _hybrid_force_safety_detector_step_value(
    *,
    history_step: ForceSafetyDetectorStep,
    acceleration_step: ForceSafetyDetectorStep,
    history_threshold: float,
) -> ForceSafetyDetectorStep:
    history_threshold = max(1e-9, float(history_threshold))
    history_score = float(history_step.score) / history_threshold
    acceleration_score = float(acceleration_step.score)
    if bool(history_step.active):
        history_score = max(1.0, history_score)
    if bool(acceleration_step.active):
        acceleration_score = max(1.0, acceleration_score)
    score = min(history_score, acceleration_score)
    return ForceSafetyDetectorStep(
        score=float(score),
        active=bool(score >= 1.0),
        force_proxy=tuple(float(value) for value in acceleration_step.force_proxy),
    )


def _history_or_v2_force_safety_detector_step_value(
    *,
    history_step: ForceSafetyDetectorStep,
    acceleration_step: ForceSafetyDetectorStep,
    history_threshold: float,
) -> ForceSafetyDetectorStep:
    history_threshold = max(1e-9, float(history_threshold))
    history_score = float(history_step.score) / history_threshold
    acceleration_score = float(acceleration_step.score)
    if bool(history_step.active):
        history_score = max(1.0, history_score)
    if bool(acceleration_step.active):
        acceleration_score = max(1.0, acceleration_score)
    score = max(history_score, acceleration_score)
    if bool(acceleration_step.active):
        force_proxy = acceleration_step.force_proxy
    elif bool(history_step.active):
        force_proxy = history_step.force_proxy
    elif acceleration_score >= history_score:
        force_proxy = acceleration_step.force_proxy
    else:
        force_proxy = history_step.force_proxy
    return ForceSafetyDetectorStep(
        score=float(score),
        active=bool(score >= 1.0),
        force_proxy=tuple(float(value) for value in force_proxy),
    )


def _history_gated_v2_force_safety_detector_step_value(
    *,
    history_step: ForceSafetyDetectorStep,
    acceleration_step: ForceSafetyDetectorStep,
    history_threshold: float,
    history_gate_ratio: float = FORCE_SAFETY_HISTORY_GATED_V2_RATIO,
) -> ForceSafetyDetectorStep:
    history_threshold = max(1e-9, float(history_threshold))
    history_gate_ratio = max(0.0, min(1.0, float(history_gate_ratio)))
    history_score = float(history_step.score) / history_threshold
    acceleration_score = float(acceleration_step.score)
    if bool(history_step.active):
        history_score = max(1.0, history_score)
    if bool(acceleration_step.active):
        acceleration_score = max(1.0, acceleration_score)

    history_active = bool(history_score >= 1.0)
    gated_acceleration_active = bool(
        acceleration_score >= 1.0
        and history_score >= history_gate_ratio
    )
    active = bool(history_active or gated_acceleration_active)
    score = max(history_score, acceleration_score if gated_acceleration_active else min(acceleration_score, 0.999))
    if gated_acceleration_active:
        force_proxy = acceleration_step.force_proxy
    else:
        force_proxy = history_step.force_proxy
    return ForceSafetyDetectorStep(
        score=float(score if active else min(score, 0.999)),
        active=active,
        force_proxy=tuple(float(value) for value in force_proxy),
    )


def _force_safety_detector_gate_value(
    *,
    t_s: float,
    start_s: float,
    hold_s: float,
    recovery_s: float,
) -> float:
    start_s = float(start_s)
    if not math.isfinite(start_s):
        return 0.0
    t_s = float(t_s)
    hold_s = max(0.0, float(hold_s))
    recovery_s = max(0.0, float(recovery_s))
    if t_s < start_s:
        return 0.0
    hold_end_s = start_s + hold_s
    if t_s <= hold_end_s:
        return 1.0
    if recovery_s > 0.0 and t_s <= hold_end_s + recovery_s:
        return max(0.0, 1.0 - (t_s - hold_end_s) / recovery_s)
    return 0.0


def _force_safety_detector_enabled_value(*, t_s: float, enable_after_s: float) -> bool:
    return float(t_s) >= max(0.0, float(enable_after_s))


def _force_safety_windows_value(
    *,
    source: str,
    oracle_windows: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    detector_start_s: float,
    detector_hold_s: float,
) -> list[tuple[float, float]]:
    source = str(source)
    if source not in FORCE_SAFETY_TRIGGER_SOURCES:
        choices = ", ".join(FORCE_SAFETY_TRIGGER_SOURCES)
        raise ValueError(f"unknown force safety trigger source {source!r}; choose one of: {choices}")
    windows: list[tuple[float, float]] = []
    if source in ORACLE_FORCE_SAFETY_TRIGGER_SOURCES:
        windows.extend((float(start_s), float(end_s)) for start_s, end_s in oracle_windows)
    if source in ANY_DEPLOYABLE_FORCE_SAFETY_TRIGGER_SOURCES and math.isfinite(float(detector_start_s)):
        detector_hold_s = max(0.0, float(detector_hold_s))
        if detector_hold_s > 0.0:
            windows.append((float(detector_start_s), float(detector_start_s) + detector_hold_s))
    return windows


def _policy_action_size_value(mode: str, *, joint_action_size: int) -> int:
    mode = str(mode)
    if mode not in POLICY_ACTION_MODES:
        choices = ", ".join(POLICY_ACTION_MODES)
        raise ValueError(f"unknown policy action mode {mode!r}; choose one of: {choices}")
    if mode == "onnx_safety_layer":
        return len(SAFETY_LAYER_ACTION_NAMES)
    return int(joint_action_size)


def _force_reference_governor_phase_value(
    *,
    t_s: float,
    windows: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    mode: str,
    delay_s: float,
    hold_s: float,
    recovery_s: float,
    tail_admittance_scale: float = 0.0,
    tail_offset_clip_scale: float = 1.0,
    tail_velocity_clip_scale: float = 1.0,
) -> tuple[float, float, float]:
    mode = str(mode)
    if mode not in FORCE_REFERENCE_GOVERNOR_MODES:
        choices = ", ".join(FORCE_REFERENCE_GOVERNOR_MODES)
        raise ValueError(f"unknown force reference governor mode {mode!r}; choose one of: {choices}")
    if mode == "off":
        return 0.0, 1.0, 1.0

    t_s = float(t_s)
    delay_s = max(0.0, float(delay_s))
    hold_s = max(0.0, float(hold_s))
    recovery_s = max(0.0, float(recovery_s))
    tail_admittance_scale = max(0.0, float(tail_admittance_scale))
    tail_offset_clip_scale = max(0.0, float(tail_offset_clip_scale))
    tail_velocity_clip_scale = max(0.0, float(tail_velocity_clip_scale))
    best_gate = 0.0
    best_offset_clip_scale = 1.0
    best_velocity_clip_scale = 1.0
    for start_s, end_s in windows:
        start_s = float(start_s) + delay_s
        end_s = float(end_s)
        if end_s < start_s:
            continue
        hold_end_s = end_s if mode == "active" else min(end_s, start_s + hold_s)
        if t_s < start_s:
            continue
        offset_clip_scale = 1.0
        velocity_clip_scale = 1.0
        if mode == "two_phase" and t_s > hold_end_s and t_s <= end_s:
            gate = tail_admittance_scale
            offset_clip_scale = tail_offset_clip_scale
            velocity_clip_scale = tail_velocity_clip_scale
        elif t_s <= hold_end_s:
            gate = 1.0
        elif recovery_s > 0.0 and t_s <= (end_s if mode == "two_phase" else hold_end_s) + recovery_s:
            recovery_start_s = end_s if mode == "two_phase" else hold_end_s
            recovery_start_gate = tail_admittance_scale if mode == "two_phase" else 1.0
            alpha = (t_s - recovery_start_s) / recovery_s
            gate = recovery_start_gate * (1.0 - alpha)
            if mode == "two_phase":
                offset_clip_scale = tail_offset_clip_scale + alpha * (1.0 - tail_offset_clip_scale)
                velocity_clip_scale = tail_velocity_clip_scale + alpha * (1.0 - tail_velocity_clip_scale)
        else:
            gate = 0.0
        gate = max(0.0, gate)
        if gate >= best_gate:
            best_gate = gate
            best_offset_clip_scale = offset_clip_scale
            best_velocity_clip_scale = velocity_clip_scale
    return best_gate, best_offset_clip_scale, best_velocity_clip_scale


def _force_reference_governor_gate_value(
    *,
    t_s: float,
    windows: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    mode: str,
    delay_s: float,
    hold_s: float,
    recovery_s: float,
    tail_admittance_scale: float = 0.0,
) -> float:
    gate, _offset_clip_scale, _velocity_clip_scale = _force_reference_governor_phase_value(
        t_s=t_s,
        windows=windows,
        mode=mode,
        delay_s=delay_s,
        hold_s=hold_s,
        recovery_s=recovery_s,
        tail_admittance_scale=tail_admittance_scale,
    )
    return gate


def _integrate_force_reference_governor_value(
    *,
    np_module: Any,
    previous_offset: Any,
    force_tracking_frame: Any,
    t_s: float,
    windows: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    mode: str,
    admittance_mps_per_n: float,
    damping: float,
    offset_clip_m: float,
    velocity_clip_mps: float,
    dt: float,
    delay_s: float,
    hold_s: float,
    recovery_s: float,
    tail_admittance_scale: float = 0.0,
    tail_offset_clip_scale: float = 1.0,
    tail_velocity_clip_scale: float = 1.0,
) -> ForceReferenceGovernorStep:
    gate, offset_clip_scale, velocity_clip_scale = _force_reference_governor_phase_value(
        t_s=t_s,
        windows=windows,
        mode=mode,
        delay_s=delay_s,
        hold_s=hold_s,
        recovery_s=recovery_s,
        tail_admittance_scale=tail_admittance_scale,
        tail_offset_clip_scale=tail_offset_clip_scale,
        tail_velocity_clip_scale=tail_velocity_clip_scale,
    )
    previous = np_module.asarray(previous_offset, dtype=np_module.float64).copy()
    force = np_module.asarray(force_tracking_frame, dtype=np_module.float64)
    if len(previous) < 3:
        previous = np_module.pad(previous, (0, 3 - len(previous)))
    if len(force) < 3:
        force = np_module.pad(force, (0, 3 - len(force)))
    previous[2] = 0.0
    force_xy = force[:2]

    velocity = np_module.zeros(3, dtype=np_module.float64)
    velocity[:2] = (
        gate * float(admittance_mps_per_n) * force_xy
        - max(0.0, float(damping)) * previous[:2]
    )
    velocity = _clip_vector_norm(
        velocity,
        max(0.0, float(velocity_clip_mps)) * max(0.0, float(velocity_clip_scale)),
        np_module,
    )
    offset = previous + velocity * max(1e-6, float(dt))
    offset[2] = 0.0
    offset = _clip_vector_norm(
        offset,
        max(0.0, float(offset_clip_m)) * max(0.0, float(offset_clip_scale)),
        np_module,
    )
    return ForceReferenceGovernorStep(
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
        velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
        gate=float(gate),
    )


def _external_force_excess_reward_terms_value(
    *,
    force_n: float,
    safe_limit_n: float,
    margin_n: float,
    active: bool,
    elapsed_since_force_start_s: float,
    excess_penalty: float,
    onset_penalty: float,
    onset_window_s: float,
) -> dict[str, float]:
    if not active or safe_limit_n <= 0.0:
        return {}
    cost = _force_excess_cost(
        force_n=force_n,
        safe_limit_n=safe_limit_n,
        margin_n=margin_n,
    )
    if cost <= 0.0:
        return {}

    terms: dict[str, float] = {}
    if excess_penalty != 0.0:
        terms["external_force_excess_penalty"] = float(excess_penalty) * cost
    if (
        onset_penalty != 0.0
        and elapsed_since_force_start_s >= 0.0
        and elapsed_since_force_start_s <= max(0.0, float(onset_window_s))
    ):
        terms["external_force_excess_onset_penalty"] = float(onset_penalty) * cost
    return terms


def _external_force_tail_reward_terms_value(
    *,
    force_n: float,
    safe_limit_n: float,
    margin_n: float,
    active: bool,
    elapsed_since_force_start_s: float,
    previous_peak_cost: float,
    violation_penalty: float,
    violation_onset_penalty: float,
    peak_delta_penalty: float,
    peak_delta_onset_penalty: float,
    onset_window_s: float,
) -> tuple[dict[str, float], float]:
    previous_peak = max(0.0, float(previous_peak_cost))
    if not active or safe_limit_n <= 0.0:
        return {}, previous_peak

    cost = _force_excess_cost(
        force_n=force_n,
        safe_limit_n=safe_limit_n,
        margin_n=margin_n,
    )
    if cost <= 0.0:
        return {}, previous_peak

    new_peak = max(previous_peak, cost)
    peak_delta = max(0.0, new_peak - previous_peak)
    onset_active = (
        elapsed_since_force_start_s >= 0.0
        and elapsed_since_force_start_s <= max(0.0, float(onset_window_s))
    )

    terms: dict[str, float] = {}
    if violation_penalty != 0.0:
        terms["external_force_violation_penalty"] = float(violation_penalty)
    if violation_onset_penalty != 0.0 and onset_active:
        terms["external_force_violation_onset_penalty"] = float(violation_onset_penalty)
    if peak_delta > 0.0 and peak_delta_penalty != 0.0:
        terms["external_force_peak_delta_penalty"] = float(peak_delta_penalty) * peak_delta
    if peak_delta > 0.0 and peak_delta_onset_penalty != 0.0 and onset_active:
        terms["external_force_peak_delta_onset_penalty"] = float(peak_delta_onset_penalty) * peak_delta
    return terms, new_peak


def _force_active_support_reward_terms_value(
    *,
    contact_count: int,
    active: bool,
    elapsed_since_force_start_s: float,
    min_contact_count: float,
    active_penalty: float,
    onset_penalty: float,
    onset_window_s: float,
) -> dict[str, float]:
    if not active:
        return {}
    deficit = max(0.0, float(min_contact_count) - float(contact_count))
    if deficit <= 0.0:
        return {}

    terms: dict[str, float] = {}
    if active_penalty != 0.0:
        terms["force_active_contact_count_penalty"] = float(active_penalty) * deficit
    if (
        onset_penalty != 0.0
        and elapsed_since_force_start_s >= 0.0
        and elapsed_since_force_start_s <= max(0.0, float(onset_window_s))
    ):
        terms["force_onset_contact_count_penalty"] = float(onset_penalty) * deficit
    return terms


def _force_yielding_velocity_reward_terms_value(
    *,
    np_module: Any,
    base_linear_velocity: Any,
    command: Any,
    force_vector: Any,
    active: bool,
    elapsed_since_force_start_s: float,
    tracking_weight: float,
    tracking_sigma: float,
    shortfall_penalty: float,
    onset_start_s: float,
    onset_window_s: float,
) -> dict[str, float]:
    if not active:
        return {}
    elapsed_s = float(elapsed_since_force_start_s)
    start_s = max(0.0, float(onset_start_s))
    end_s = start_s + max(0.0, float(onset_window_s))
    if elapsed_s < start_s or elapsed_s > end_s:
        return {}
    if tracking_weight == 0.0 and shortfall_penalty == 0.0:
        return {}

    force_xy = np_module.asarray(force_vector[:2], dtype=np_module.float64)
    force_norm = float(np_module.linalg.norm(force_xy))
    if force_norm <= 1e-12:
        return {}
    direction_xy = force_xy / force_norm
    command_xy = np_module.asarray(command[:2], dtype=np_module.float64)
    velocity_xy = np_module.asarray(base_linear_velocity[:2], dtype=np_module.float64)

    desired_along = max(0.0, float(np_module.dot(command_xy, direction_xy)))
    if desired_along <= 1e-9:
        return {}
    actual_along = float(np_module.dot(velocity_xy, direction_xy))
    shortfall = max(0.0, desired_along - actual_along)

    terms: dict[str, float] = {}
    if tracking_weight != 0.0:
        sigma = max(1e-6, float(tracking_sigma))
        terms["force_yielding_velocity_tracking"] = float(tracking_weight) * float(
            np_module.exp(-(shortfall * shortfall) / (sigma * sigma))
        )
    if shortfall_penalty != 0.0:
        terms["force_yielding_velocity_shortfall_penalty"] = float(shortfall_penalty) * shortfall * shortfall
    return terms


def _sample_external_force_direction_value(
    *,
    np_module: Any,
    rng: Any,
    fixed_angle_rad: float | None,
    z_fraction: float,
    direction_mode: str,
    lateral_probability: float,
) -> Any:
    if fixed_angle_rad is not None:
        angle = float(fixed_angle_rad)
        return np_module.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=np_module.float64)

    mode = str(direction_mode)
    if mode not in EXTERNAL_FORCE_DIRECTION_MODES:
        choices = ", ".join(EXTERNAL_FORCE_DIRECTION_MODES)
        raise ValueError(f"unknown external force direction mode {mode!r}; choose one of: {choices}")

    lateral_probability = max(0.0, min(1.0, float(lateral_probability)))
    if mode == "yielding_tail_mixed":
        # Focus on the empirically observed strict-compliance tail while keeping
        # side pushes and a small forward anchor in the training distribution.
        sample = float(rng.uniform())
        weighted_angles = (
            (-math.pi, 0.25),
            (-3.0 * math.pi / 4.0, 0.25),
            (3.0 * math.pi / 4.0, 0.25),
            (math.pi / 2.0, 0.10),
            (-math.pi / 2.0, 0.10),
            (0.0, 0.05),
        )
        cumulative = 0.0
        angle = 0.0
        for candidate_angle, probability in weighted_angles:
            cumulative += probability
            if sample <= cumulative:
                angle = candidate_angle
                break
        return np_module.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=np_module.float64)

    if mode == "lateral_bimodal" or (mode == "lateral_mixed" and float(rng.uniform()) < lateral_probability):
        sign = 1.0 if float(rng.uniform()) < 0.5 else -1.0
        return np_module.asarray([0.0, sign, 0.0], dtype=np_module.float64)

    z_fraction = max(0.0, float(z_fraction))
    if z_fraction <= 0.0:
        angle = float(rng.uniform(-math.pi, math.pi))
        return np_module.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=np_module.float64)

    direction = rng.normal(size=3)
    direction[2] *= z_fraction
    norm = float(np_module.linalg.norm(direction))
    if norm <= 1e-9:
        return np_module.asarray([1.0, 0.0, 0.0], dtype=np_module.float64)
    return direction / norm


def _compliant_target_for_force_body(body_name: str) -> str | None:
    body_name = str(body_name)
    if body_name == "base_link":
        return "base_link"
    for leg in FEET:
        if body_name == f"{leg}_thigh":
            return f"{leg}_hip"
        if body_name == f"{leg}_calf":
            return f"{leg}_calf"
    return None


def _map_applied_forces_to_compliant_targets(
    body_forces_root: dict[str, Any],
    *,
    np_module: Any,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for body_name, force_root in body_forces_root.items():
        target_name = _compliant_target_for_force_body(body_name)
        if target_name is None:
            continue
        force = np_module.asarray(force_root, dtype=np_module.float64)
        if target_name in mapped:
            mapped[target_name] = mapped[target_name] + force
        else:
            mapped[target_name] = force.copy()
    return mapped


@dataclass(frozen=True)
class VelocityRewardConfig:
    """Reward weights for velocity tracking tasks."""

    linear_std: float = 0.06
    yaw_std: float = 0.3
    alive: float = 0.25
    height: float = 0.35
    track_linear_velocity: float = 3.0
    track_angular_velocity: float = 0.4
    track_angular_velocity_moving: float = 0.15
    forward_progress: float = 1.5
    target_forward_velocity_band: float = 0.0
    target_forward_velocity_band_min: float = 0.35
    target_forward_velocity_band_max: float = 0.45
    target_forward_velocity_band_decay: float = 0.10
    velocity_shortfall_penalty: float = -16.0
    velocity_overspeed_penalty: float = 0.0
    lateral_velocity_l2: float = -1.2
    lateral_displacement_l2: float = 0.0
    lateral_return_velocity_l2: float = 0.0
    lateral_return_velocity_gain: float = 0.5
    lateral_return_velocity_clip: float = 0.12
    yaw_rate_l2: float = -0.8
    heading_l2: float = -2.0
    feet_air_time: float = 20.0
    contact_count_penalty: float = -0.06
    stance_stagnation_penalty: float = -0.16
    pose_standing: float = 0.5
    pose_moving: float = 0.15
    flat_orientation_l2: float = -2.0
    body_ang_vel: float = -0.02
    joint_acc_l2: float = -2.5e-8
    joint_pos_limits: float = -5.0
    action_rate_l2: float = -0.02
    foot_slip: float = -0.10
    force_active_velocity_scale: float = 1.0
    force_active_drift_scale: float = 1.0
    force_active_stance_scale: float = 1.0
    force_active_pose_scale: float = 1.0
    recovery_velocity_scale: float = 1.0
    recovery_drift_scale: float = 1.0
    recovery_stance_scale: float = 1.0
    recovery_pose_scale: float = 1.0
    recovery_command_velocity_scale: float = 1.0
    force_active_compliance_velocity_per_n: float = 0.0
    force_active_compliance_velocity_clip: float = 0.0
    recovery_window_s: float = 0.0
    recovery_track_linear_velocity: float = 0.0
    recovery_forward_progress: float = 0.0
    recovery_velocity_shortfall_penalty: float = 0.0
    recovery_lateral_velocity_l2: float = 0.0
    recovery_yaw_rate_l2: float = 0.0
    recovery_stance_stagnation_penalty: float = 0.0
    anti_collapse_height: float = 0.0
    anti_collapse_z_safe: float = 0.26
    anti_collapse_downward_velocity: float = 0.0
    anti_collapse_vz_safe: float = 0.05
    anti_collapse_orientation: float = 0.0
    anti_collapse_recovery_scale: float = 1.0
    reference_dynamics_tracking: float = 0.0
    reference_dynamics_tracking_sigma: float = 0.12
    reference_dynamics_velocity_scale: float = 0.05
    reference_force_tracking: float = 0.0
    reference_force_tracking_sigma_n: float = 12.0
    unsafe_force_penalty: float = 0.0
    unsafe_force_peak_penalty: float = 0.0
    unsafe_force_onset_penalty: float = 0.0
    unsafe_force_onset_window_s: float = 0.35
    external_force_excess_penalty: float = 0.0
    external_force_excess_onset_penalty: float = 0.0
    external_force_excess_onset_window_s: float = 0.35
    external_force_violation_penalty: float = 0.0
    external_force_violation_onset_penalty: float = 0.0
    external_force_peak_delta_penalty: float = 0.0
    external_force_peak_delta_onset_penalty: float = 0.0
    force_active_min_contact_count: float = 0.0
    force_active_contact_count_penalty: float = 0.0
    force_onset_contact_count_penalty: float = 0.0
    force_yielding_velocity_tracking: float = 0.0
    force_yielding_velocity_tracking_sigma: float = 0.25
    force_yielding_velocity_shortfall_penalty: float = 0.0
    force_yielding_velocity_onset_start_s: float = 0.10
    force_yielding_velocity_onset_window_s: float = 0.35
    unsafe_force_limit_n: float = 18.0
    unsafe_force_margin_n: float = 2.0
    keypoint_tracking_comp: float = 0.0
    keypoint_tracking_sigma: float = 0.08
    keypoint_tracking_base_weight: float = 1.0
    keypoint_tracking_hip_weight: float = 1.0
    keypoint_tracking_calf_weight: float = 1.0
    keypoint_tracking_foot_weight: float = 1.0
    root_tracking: float = 0.0
    root_tracking_sigma: float = 0.20
    root_yaw_tracking: float = 0.0
    root_yaw_tracking_sigma: float = 0.35
    nominal_gait_reference_force_scale: float = 1.0
    nominal_gait_reference_recovery_scale: float = 1.0
    expert_action_tracking: float = 0.0
    expert_action_tracking_sigma: float = 0.35
    joint_pos_tracking: float = 0.0
    joint_pos_tracking_sigma: float = 0.35
    joint_vel_tracking: float = 0.0
    joint_vel_tracking_sigma: float = 4.0
    impact_force_l2: float = 0.0
    torque_smoothness_l2: float = 0.0
    residual_action_l2: float = 0.0
    residual_action_rate_l2: float = 0.0
    compliant_target_admittance: float = 0.0
    compliant_target_stiffness: float = 8.0
    compliant_target_damping: float = 3.0
    compliant_target_offset_clip: float = 0.18
    compliant_target_velocity_clip: float = 1.2
    compliant_target_horizontal_only: float = 1.0
    compliant_target_z_offset_clip: float = 0.0
    compliant_target_z_velocity_clip: float = 0.0


VELOCITY_REWARD_PROFILES: dict[str, VelocityRewardConfig] = {
    "default": VelocityRewardConfig(),
    "tracking": VelocityRewardConfig(
        linear_std=0.08,
        track_linear_velocity=2.0,
        forward_progress=2.5,
        velocity_shortfall_penalty=-20.0,
        lateral_velocity_l2=-1.6,
        yaw_rate_l2=-1.0,
        heading_l2=-2.8,
        feet_air_time=14.0,
        stance_stagnation_penalty=-0.20,
        pose_moving=0.10,
    ),
    "stable": VelocityRewardConfig(
        linear_std=0.10,
        track_linear_velocity=2.0,
        forward_progress=0.8,
        velocity_shortfall_penalty=-8.0,
        lateral_velocity_l2=-2.0,
        yaw_rate_l2=-1.2,
        heading_l2=-3.0,
        feet_air_time=8.0,
        contact_count_penalty=-0.03,
        stance_stagnation_penalty=-0.08,
        pose_moving=0.25,
        flat_orientation_l2=-3.0,
    ),
    "straight_stable": VelocityRewardConfig(
        linear_std=0.05,
        yaw_std=0.20,
        track_linear_velocity=4.0,
        track_angular_velocity=0.8,
        track_angular_velocity_moving=0.8,
        forward_progress=0.8,
        velocity_shortfall_penalty=-12.0,
        lateral_velocity_l2=-4.0,
        yaw_rate_l2=-3.0,
        heading_l2=-1.0,
        feet_air_time=12.0,
        contact_count_penalty=-0.05,
        stance_stagnation_penalty=-0.14,
        pose_moving=0.18,
        flat_orientation_l2=-3.0,
        body_ang_vel=-0.04,
        foot_slip=-0.12,
    ),
    "disturbance_recovery": VelocityRewardConfig(
        lateral_velocity_l2=-1.6,
        yaw_rate_l2=-1.0,
        feet_air_time=18.0,
        recovery_window_s=2.0,
        recovery_track_linear_velocity=4.0,
        recovery_forward_progress=2.0,
        recovery_velocity_shortfall_penalty=-24.0,
        recovery_lateral_velocity_l2=-4.0,
        recovery_yaw_rate_l2=-2.0,
        recovery_stance_stagnation_penalty=-0.28,
    ),
    "gentle_disturbance_recovery": VelocityRewardConfig(
        lateral_velocity_l2=-1.6,
        yaw_rate_l2=-1.0,
        feet_air_time=18.0,
        force_active_velocity_scale=0.35,
        force_active_drift_scale=0.15,
        force_active_stance_scale=0.20,
        force_active_pose_scale=0.50,
        recovery_window_s=3.0,
        recovery_track_linear_velocity=8.0,
        recovery_forward_progress=4.0,
        recovery_velocity_shortfall_penalty=-48.0,
        recovery_lateral_velocity_l2=-4.0,
        recovery_yaw_rate_l2=-2.0,
        recovery_stance_stagnation_penalty=-0.55,
    ),
    "gentle_compliant_recovery": VelocityRewardConfig(
        linear_std=0.10,
        yaw_std=0.35,
        track_linear_velocity=3.0,
        track_angular_velocity=0.35,
        track_angular_velocity_moving=0.12,
        forward_progress=1.8,
        velocity_shortfall_penalty=-18.0,
        lateral_velocity_l2=-1.4,
        yaw_rate_l2=-0.8,
        feet_air_time=18.0,
        contact_count_penalty=-0.04,
        stance_stagnation_penalty=-0.16,
        pose_moving=0.12,
        flat_orientation_l2=-2.5,
        body_ang_vel=-0.03,
        foot_slip=-0.10,
        force_active_velocity_scale=0.25,
        force_active_drift_scale=0.12,
        force_active_stance_scale=0.25,
        force_active_pose_scale=0.45,
        force_active_compliance_velocity_per_n=0.008,
        force_active_compliance_velocity_clip=0.22,
        recovery_window_s=6.0,
        recovery_track_linear_velocity=10.0,
        recovery_forward_progress=6.0,
        recovery_velocity_shortfall_penalty=-72.0,
        recovery_lateral_velocity_l2=-5.0,
        recovery_yaw_rate_l2=-2.5,
        recovery_stance_stagnation_penalty=-0.8,
    ),
    "gentle_fullbody_tracking": VelocityRewardConfig(
        linear_std=0.10,
        yaw_std=0.30,
        track_linear_velocity=2.0,
        track_angular_velocity=0.45,
        track_angular_velocity_moving=0.45,
        forward_progress=1.2,
        velocity_shortfall_penalty=-12.0,
        lateral_velocity_l2=-2.4,
        yaw_rate_l2=-1.8,
        heading_l2=-4.0,
        feet_air_time=14.0,
        contact_count_penalty=-0.04,
        stance_stagnation_penalty=-0.12,
        pose_moving=0.10,
        flat_orientation_l2=-3.0,
        body_ang_vel=-0.04,
        foot_slip=-0.14,
        force_active_velocity_scale=0.45,
        force_active_drift_scale=0.45,
        force_active_stance_scale=0.45,
        force_active_pose_scale=0.65,
        recovery_window_s=2.5,
        recovery_track_linear_velocity=6.0,
        recovery_forward_progress=4.0,
        recovery_velocity_shortfall_penalty=-40.0,
        recovery_lateral_velocity_l2=-4.0,
        recovery_yaw_rate_l2=-2.5,
        recovery_stance_stagnation_penalty=-0.45,
        keypoint_tracking_comp=2.0,
        keypoint_tracking_sigma=0.09,
        root_tracking=0.8,
        root_tracking_sigma=0.25,
        root_yaw_tracking=0.7,
        root_yaw_tracking_sigma=0.35,
        joint_pos_tracking=0.9,
        joint_pos_tracking_sigma=0.35,
        joint_vel_tracking=0.35,
        joint_vel_tracking_sigma=4.0,
        impact_force_l2=-2.0e-5,
        torque_smoothness_l2=-2.0e-5,
        compliant_target_admittance=0.012,
        compliant_target_stiffness=8.0,
        compliant_target_damping=3.0,
        compliant_target_offset_clip=0.18,
        compliant_target_velocity_clip=1.0,
    ),
    "gentle_fullbody_tracking_v1": VelocityRewardConfig(
        linear_std=0.16,
        yaw_std=0.30,
        track_linear_velocity=3.4,
        track_angular_velocity=0.45,
        track_angular_velocity_moving=0.45,
        forward_progress=2.0,
        velocity_shortfall_penalty=-20.0,
        lateral_velocity_l2=-2.2,
        yaw_rate_l2=-1.6,
        heading_l2=-3.8,
        feet_air_time=16.0,
        contact_count_penalty=-0.04,
        stance_stagnation_penalty=-0.10,
        pose_moving=0.08,
        flat_orientation_l2=-3.2,
        body_ang_vel=-0.04,
        foot_slip=-0.12,
        force_active_velocity_scale=0.65,
        force_active_drift_scale=0.50,
        force_active_stance_scale=0.45,
        force_active_pose_scale=0.45,
        recovery_window_s=3.5,
        recovery_track_linear_velocity=8.0,
        recovery_forward_progress=5.0,
        recovery_velocity_shortfall_penalty=-52.0,
        recovery_lateral_velocity_l2=-3.5,
        recovery_yaw_rate_l2=-2.0,
        recovery_stance_stagnation_penalty=-0.35,
        keypoint_tracking_comp=1.2,
        keypoint_tracking_sigma=0.18,
        keypoint_tracking_base_weight=1.0,
        keypoint_tracking_hip_weight=0.50,
        keypoint_tracking_calf_weight=0.25,
        keypoint_tracking_foot_weight=0.05,
        root_tracking=1.2,
        root_tracking_sigma=0.55,
        root_yaw_tracking=0.5,
        root_yaw_tracking_sigma=0.35,
        joint_pos_tracking=0.45,
        joint_pos_tracking_sigma=0.45,
        joint_vel_tracking=0.15,
        joint_vel_tracking_sigma=5.0,
        impact_force_l2=-2.0e-5,
        torque_smoothness_l2=-2.0e-5,
        compliant_target_admittance=0.010,
        compliant_target_stiffness=6.0,
        compliant_target_damping=2.5,
        compliant_target_offset_clip=0.16,
        compliant_target_velocity_clip=0.9,
    ),
    "gentle_fullbody_teacher_tracking": VelocityRewardConfig(
        linear_std=0.18,
        yaw_std=0.35,
        track_linear_velocity=3.6,
        track_angular_velocity=0.50,
        track_angular_velocity_moving=0.50,
        forward_progress=2.2,
        velocity_shortfall_penalty=-22.0,
        lateral_velocity_l2=-2.0,
        yaw_rate_l2=-1.4,
        heading_l2=-3.4,
        feet_air_time=16.0,
        contact_count_penalty=-0.035,
        stance_stagnation_penalty=-0.09,
        pose_moving=0.06,
        flat_orientation_l2=-3.0,
        body_ang_vel=-0.035,
        foot_slip=-0.12,
        force_active_velocity_scale=0.75,
        force_active_drift_scale=0.55,
        force_active_stance_scale=0.45,
        force_active_pose_scale=0.40,
        recovery_window_s=3.5,
        recovery_track_linear_velocity=8.5,
        recovery_forward_progress=5.5,
        recovery_velocity_shortfall_penalty=-56.0,
        velocity_overspeed_penalty=-24.0,
        recovery_lateral_velocity_l2=-3.2,
        recovery_yaw_rate_l2=-1.8,
        recovery_stance_stagnation_penalty=-0.30,
        anti_collapse_height=-80.0,
        anti_collapse_z_safe=0.27,
        anti_collapse_downward_velocity=-5.0,
        anti_collapse_vz_safe=0.05,
        anti_collapse_orientation=-5.0,
        keypoint_tracking_comp=1.8,
        keypoint_tracking_sigma=0.20,
        keypoint_tracking_base_weight=1.0,
        keypoint_tracking_hip_weight=0.65,
        keypoint_tracking_calf_weight=0.45,
        keypoint_tracking_foot_weight=0.25,
        root_tracking=1.4,
        root_tracking_sigma=0.60,
        root_yaw_tracking=0.55,
        root_yaw_tracking_sigma=0.40,
        joint_pos_tracking=0.55,
        joint_pos_tracking_sigma=0.50,
        joint_vel_tracking=0.18,
        joint_vel_tracking_sigma=5.5,
        impact_force_l2=-2.0e-5,
        torque_smoothness_l2=-2.0e-5,
        compliant_target_admittance=0.012,
        compliant_target_stiffness=6.0,
        compliant_target_damping=2.5,
        compliant_target_offset_clip=0.18,
        compliant_target_velocity_clip=1.0,
    ),
    "gentle_fullbody_teacher_impedance_tracking": VelocityRewardConfig(
        linear_std=0.18,
        yaw_std=0.35,
        track_linear_velocity=3.6,
        track_angular_velocity=0.50,
        track_angular_velocity_moving=0.50,
        forward_progress=2.2,
        velocity_shortfall_penalty=-22.0,
        lateral_velocity_l2=-2.0,
        yaw_rate_l2=-1.4,
        heading_l2=-3.4,
        feet_air_time=16.0,
        contact_count_penalty=-0.035,
        stance_stagnation_penalty=-0.09,
        pose_moving=0.06,
        flat_orientation_l2=-3.0,
        body_ang_vel=-0.035,
        foot_slip=-0.12,
        force_active_velocity_scale=0.75,
        force_active_drift_scale=0.55,
        force_active_stance_scale=0.45,
        force_active_pose_scale=0.40,
        recovery_window_s=3.5,
        recovery_track_linear_velocity=8.5,
        recovery_forward_progress=5.5,
        recovery_velocity_shortfall_penalty=-56.0,
        velocity_overspeed_penalty=-36.0,
        recovery_lateral_velocity_l2=-3.2,
        recovery_yaw_rate_l2=-1.8,
        recovery_stance_stagnation_penalty=-0.30,
        anti_collapse_height=-120.0,
        anti_collapse_z_safe=0.28,
        anti_collapse_downward_velocity=-8.0,
        anti_collapse_vz_safe=0.05,
        anti_collapse_orientation=-8.0,
        reference_dynamics_tracking=2.0,
        reference_dynamics_tracking_sigma=0.18,
        reference_dynamics_velocity_scale=0.04,
        reference_force_tracking=2.0,
        reference_force_tracking_sigma_n=10.0,
        unsafe_force_penalty=-6.0,
        unsafe_force_limit_n=18.0,
        unsafe_force_margin_n=2.0,
        keypoint_tracking_comp=1.8,
        keypoint_tracking_sigma=0.20,
        keypoint_tracking_base_weight=1.0,
        keypoint_tracking_hip_weight=0.65,
        keypoint_tracking_calf_weight=0.45,
        keypoint_tracking_foot_weight=0.25,
        root_tracking=1.4,
        root_tracking_sigma=0.60,
        root_yaw_tracking=0.55,
        root_yaw_tracking_sigma=0.40,
        joint_pos_tracking=0.55,
        joint_pos_tracking_sigma=0.50,
        joint_vel_tracking=0.18,
        joint_vel_tracking_sigma=5.5,
        impact_force_l2=-2.0e-5,
        torque_smoothness_l2=-2.0e-5,
        compliant_target_admittance=0.014,
        compliant_target_stiffness=7.0,
        compliant_target_damping=3.2,
        compliant_target_offset_clip=0.08,
        compliant_target_velocity_clip=0.35,
        compliant_target_horizontal_only=1.0,
        compliant_target_z_offset_clip=0.0,
        compliant_target_z_velocity_clip=0.0,
    ),
    "gentle_fullbody_teacher_push_recovery": VelocityRewardConfig(
        linear_std=0.18,
        yaw_std=0.35,
        track_linear_velocity=3.2,
        track_angular_velocity=0.45,
        track_angular_velocity_moving=0.45,
        forward_progress=2.0,
        velocity_shortfall_penalty=-20.0,
        lateral_velocity_l2=-2.2,
        yaw_rate_l2=-1.6,
        heading_l2=-3.4,
        feet_air_time=14.0,
        contact_count_penalty=-0.04,
        stance_stagnation_penalty=-0.08,
        pose_moving=0.06,
        flat_orientation_l2=-4.0,
        body_ang_vel=-0.05,
        foot_slip=-0.14,
        force_active_velocity_scale=0.70,
        force_active_drift_scale=0.60,
        force_active_stance_scale=0.65,
        force_active_pose_scale=0.65,
        recovery_window_s=4.5,
        recovery_track_linear_velocity=10.0,
        recovery_forward_progress=6.5,
        recovery_velocity_shortfall_penalty=-70.0,
        velocity_overspeed_penalty=-30.0,
        recovery_lateral_velocity_l2=-4.0,
        recovery_yaw_rate_l2=-2.4,
        recovery_stance_stagnation_penalty=-0.40,
        anti_collapse_height=-190.0,
        anti_collapse_z_safe=0.29,
        anti_collapse_downward_velocity=-14.0,
        anti_collapse_vz_safe=0.04,
        anti_collapse_orientation=-14.0,
        anti_collapse_recovery_scale=1.0,
        reference_dynamics_tracking=1.5,
        reference_dynamics_tracking_sigma=0.20,
        reference_dynamics_velocity_scale=0.04,
        reference_force_tracking=0.0,
        unsafe_force_penalty=0.0,
        unsafe_force_peak_penalty=0.0,
        unsafe_force_onset_penalty=0.0,
        external_force_excess_penalty=0.0,
        external_force_excess_onset_penalty=0.0,
        external_force_violation_penalty=0.0,
        external_force_violation_onset_penalty=0.0,
        external_force_peak_delta_penalty=0.0,
        external_force_peak_delta_onset_penalty=0.0,
        force_active_min_contact_count=3.0,
        force_active_contact_count_penalty=-3.2,
        force_onset_contact_count_penalty=0.0,
        keypoint_tracking_comp=1.5,
        keypoint_tracking_sigma=0.22,
        keypoint_tracking_base_weight=1.0,
        keypoint_tracking_hip_weight=0.60,
        keypoint_tracking_calf_weight=0.40,
        keypoint_tracking_foot_weight=0.20,
        root_tracking=1.2,
        root_tracking_sigma=0.65,
        root_yaw_tracking=0.45,
        root_yaw_tracking_sigma=0.45,
        joint_pos_tracking=0.45,
        joint_pos_tracking_sigma=0.55,
        joint_vel_tracking=0.16,
        joint_vel_tracking_sigma=6.0,
        impact_force_l2=-1.0e-5,
        torque_smoothness_l2=-2.0e-5,
        residual_action_l2=-0.04,
        residual_action_rate_l2=-0.02,
        compliant_target_admittance=0.010,
        compliant_target_stiffness=6.0,
        compliant_target_damping=3.0,
        compliant_target_offset_clip=0.10,
        compliant_target_velocity_clip=0.45,
        compliant_target_horizontal_only=1.0,
        compliant_target_z_offset_clip=0.0,
        compliant_target_z_velocity_clip=0.0,
    ),
    "gentle_fullbody_teacher_tracking_stability": VelocityRewardConfig(
        linear_std=0.16,
        yaw_std=0.28,
        track_linear_velocity=3.8,
        track_angular_velocity=0.70,
        track_angular_velocity_moving=0.70,
        forward_progress=2.4,
        velocity_shortfall_penalty=-24.0,
        lateral_velocity_l2=-4.5,
        yaw_rate_l2=-3.0,
        heading_l2=-6.0,
        feet_air_time=15.0,
        contact_count_penalty=-0.05,
        stance_stagnation_penalty=-0.12,
        pose_moving=0.08,
        flat_orientation_l2=-5.0,
        body_ang_vel=-0.06,
        foot_slip=-0.18,
        force_active_velocity_scale=0.90,
        force_active_drift_scale=1.00,
        force_active_stance_scale=0.85,
        force_active_pose_scale=0.75,
        recovery_window_s=4.0,
        recovery_track_linear_velocity=9.0,
        recovery_forward_progress=5.5,
        recovery_velocity_shortfall_penalty=-60.0,
        velocity_overspeed_penalty=-36.0,
        recovery_lateral_velocity_l2=-8.0,
        recovery_yaw_rate_l2=-5.0,
        recovery_stance_stagnation_penalty=-0.45,
        anti_collapse_height=-140.0,
        anti_collapse_z_safe=0.28,
        anti_collapse_downward_velocity=-10.0,
        anti_collapse_vz_safe=0.05,
        anti_collapse_orientation=-10.0,
        keypoint_tracking_comp=1.2,
        keypoint_tracking_sigma=0.18,
        keypoint_tracking_base_weight=0.80,
        keypoint_tracking_hip_weight=0.55,
        keypoint_tracking_calf_weight=0.35,
        keypoint_tracking_foot_weight=0.18,
        root_tracking=2.2,
        root_tracking_sigma=0.38,
        root_yaw_tracking=1.2,
        root_yaw_tracking_sigma=0.25,
        joint_pos_tracking=0.65,
        joint_pos_tracking_sigma=0.45,
        joint_vel_tracking=0.20,
        joint_vel_tracking_sigma=5.0,
        impact_force_l2=-4.0e-5,
        torque_smoothness_l2=-3.0e-5,
        compliant_target_admittance=0.006,
        compliant_target_stiffness=9.0,
        compliant_target_damping=4.0,
        compliant_target_offset_clip=0.10,
        compliant_target_velocity_clip=0.55,
        compliant_target_horizontal_only=1.0,
        compliant_target_z_offset_clip=0.0,
        compliant_target_z_velocity_clip=0.0,
    ),
    "mjlab": VelocityRewardConfig(
        linear_std=0.5,
        yaw_std=0.5,
        alive=0.0,
        height=0.0,
        track_linear_velocity=1.0,
        track_angular_velocity=1.0,
        track_angular_velocity_moving=1.0,
        forward_progress=0.0,
        velocity_shortfall_penalty=0.0,
        lateral_velocity_l2=0.0,
        yaw_rate_l2=0.0,
        heading_l2=0.0,
        feet_air_time=0.0,
        contact_count_penalty=0.0,
        stance_stagnation_penalty=0.0,
        pose_standing=1.0,
        pose_moving=1.0,
        flat_orientation_l2=-5.0,
        body_ang_vel=-0.05,
        joint_acc_l2=-2.5e-7,
        joint_pos_limits=-10.0,
        action_rate_l2=-0.05,
        foot_slip=-0.25,
    ),
}


VELOCITY_REWARD_PROFILES["gentle_fullbody_teacher_layered_gait_quality"] = replace(
    VELOCITY_REWARD_PROFILES["gentle_fullbody_teacher_impedance_tracking"],
    track_linear_velocity=0.0,
    forward_progress=0.0,
    target_forward_velocity_band=5.0,
    target_forward_velocity_band_min=0.35,
    target_forward_velocity_band_max=0.45,
    target_forward_velocity_band_decay=0.10,
    velocity_shortfall_penalty=-28.0,
    velocity_overspeed_penalty=-48.0,
    keypoint_tracking_comp=0.20,
    keypoint_tracking_sigma=0.24,
    expert_action_tracking=1.20,
    expert_action_tracking_sigma=0.30,
    joint_pos_tracking=0.20,
    joint_vel_tracking=0.04,
    nominal_gait_reference_force_scale=0.25,
    nominal_gait_reference_recovery_scale=0.60,
)


@dataclass(frozen=True)
class StandBalanceEnvConfig:
    task: str = "stand"
    control_dt: float = 0.02
    episode_length_s: float = 8.0
    action_scale: float = 0.25
    action_smoothing: float = 0.8
    reward_scale: float = 0.1
    target_base_z: float = 0.36
    init_base_z: float = 0.36
    fall_base_z: float = 0.25
    bad_orientation_limit_rad: float = 1.2132252231493863
    reset_settle_s: float = 0.5
    include_command: bool = False
    randomize_commands: bool = False
    standing_command_prob: float = 0.05
    command_lin_vel_x_range: tuple[float, float] = (-0.5, 1.0)
    command_lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    command_yaw_rate_range: tuple[float, float] = (-0.5, 0.5)
    target_forward_velocity: float = 0.0
    target_lateral_velocity: float = 0.0
    target_yaw_rate: float = 0.0
    force_yielding_command_mode: str = "scaled"
    force_yielding_command_velocity_per_n: float = 0.0
    force_yielding_command_velocity_clip: float = 0.0
    force_yielding_command_pulse_start_s: float = 0.10
    force_yielding_command_pulse_duration_s: float = 0.20
    force_yielding_command_pulse_recovery_s: float = 0.10
    force_yielding_command_pulse_post_clip: float = 0.0
    force_impedance_mode: str = "off"
    force_impedance_joint_scope: str = "all"
    force_impedance_kp_scale: float = 1.0
    force_impedance_kd_scale: float = 1.0
    force_impedance_delay_s: float = 0.0
    force_impedance_hold_s: float = 0.15
    force_impedance_recovery_s: float = 0.10
    force_impedance_tail_kp_scale: float = 1.0
    force_impedance_tail_kd_scale: float = 1.0
    force_reference_governor_mode: str = "off"
    force_reference_governor_admittance_mps_per_n: float = 0.0
    force_reference_governor_damping: float = 5.0
    force_reference_governor_offset_clip_m: float = 0.0
    force_reference_governor_velocity_clip_mps: float = 0.0
    force_reference_governor_delay_s: float = 0.10
    force_reference_governor_hold_s: float = 0.20
    force_reference_governor_recovery_s: float = 0.10
    force_reference_governor_tail_admittance_scale: float = 0.0
    force_reference_governor_tail_offset_clip_scale: float = 1.0
    force_reference_governor_tail_velocity_clip_scale: float = 1.0
    force_response_router_mode: str = "off"
    force_response_profile: str = "body_yield_foot_guard"
    force_response_foot_kp_scale: float = 1.05
    force_response_foot_kd_scale: float = 1.15
    force_safety_trigger_source: str = "oracle"
    force_safety_detector_linear_acceleration_threshold: float = 1.5
    force_safety_detector_angular_acceleration_threshold: float = 8.0
    force_safety_detector_joint_error_threshold: float = 0.20
    force_safety_detector_joint_velocity_threshold: float = 12.0
    force_safety_detector_contact_loss: bool = True
    force_safety_detector_enable_after_s: float = 0.0
    force_safety_detector_hold_s: float = 0.25
    force_safety_detector_recovery_s: float = 0.12
    force_safety_history_estimator_path: str | None = None
    push_time_s: float | None = None
    push_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    push_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_force_mode: str = "constant"
    external_force_probability: float = 0.0
    external_force_body_names: tuple[str, ...] = ("base_link",)
    external_force_active_body_count: int = 1
    external_force_event_count_range: tuple[int, int] = (1, 1)
    external_force_rest_s_range: tuple[float, float] = (1.0, 2.0)
    external_force_start_s_range: tuple[float, float] = (1.0, 3.0)
    external_force_duration_s_range: tuple[float, float] = (0.1, 0.3)
    external_force_min_n: float = 0.0
    external_force_max_n: float = 0.0
    external_force_curriculum_start_n: float | None = None
    external_force_z_fraction: float = 0.0
    external_force_direction_angle_rad: float | None = None
    external_force_direction_mode: str = "default"
    external_force_lateral_probability: float = 0.85
    external_force_torque_max_nm: float = 0.0
    external_force_spring_stiffness_range: tuple[float, float] = (50.0, 250.0)
    external_force_spring_damping: float = 2.0
    external_force_guiding_probability: float = 0.5
    external_force_transition_s: float = 0.08
    external_force_net_force_limit_n: float = 0.0
    external_force_net_torque_limit_nm: float = 0.0
    external_force_reference_mass: float = 0.1
    external_force_reference_damping: float = 2.0
    external_force_reference_velocity_clip: float = 4.0
    external_force_reference_acceleration_clip: float = 1000.0
    external_force_safe_limit_min_n: float = 0.0
    external_force_safe_limit_max_n: float = 0.0
    external_force_safe_margin_n: float = 10.0
    include_external_force_observation: bool = False
    observation_mode: str = "policy"
    policy_action_mode: str = "full_action"
    onnx_policy_path: str | None = None
    onnx_normalizer_checkpoint: str | None = None
    residual_action_scale: float = 0.0
    velocity_pose_profile: str = "official"
    policy_leg_order: tuple[str, ...] = FEET
    velocity_reward_profile: str = "default"
    velocity_command_frame: str = "world"
    fullbody_reference_mode: str = "static"
    nominal_reference_dataset: str | None = None
    gait_frequency_hz: float = 1.7
    gait_step_length: float = 0.12
    gait_swing_height: float = 0.055
    randomize_gait_parameters: bool = False
    gait_frequency_range: tuple[float, float] = (1.6, 1.8)
    gait_step_length_range: tuple[float, float] = (0.07, 0.09)
    gait_swing_height_range: tuple[float, float] = (0.035, 0.05)
    gait_joint_thigh_amplitude: float = 0.18
    gait_joint_calf_amplitude: float = 0.28
    velocity_reward: VelocityRewardConfig = field(default_factory=VelocityRewardConfig)
    pd: PDConfig = PDConfig(kp=90.0, kd=5.0, torque_limit=45.0)


@dataclass
class ExternalForceSpring:
    body_id: int
    body_name: str
    spring_type: str
    start_s: float
    end_s: float
    direction_root: Any
    anchor_root: Any
    anchor_velocity_root: Any
    anchor_target_root: Any
    torque_root: Any
    stiffness: float
    damping: float
    cap_n: float
    frame_origin_world: Any | None = None
    frame_rotation_world: Any | None = None
    initialized: bool = False


@dataclass
class ExternalForcePulse:
    body_id: int
    body_name: str
    start_s: float
    end_s: float
    force_vector: Any
    torque_vector: Any


class Go2StandBalanceEnv:
    """Small MuJoCo RL environment without a Gym dependency."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        config: StandBalanceEnvConfig | None = None,
    ) -> None:
        try:
            import mujoco
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo dependencies are missing. Install with: pip install -e '.[sim]'"
            ) from exc

        self.mujoco = mujoco
        self.np = np
        self.model_path = Path(model_path)
        self.config = config or StandBalanceEnvConfig()
        if self.config.task not in RL_TASKS:
            raise ValueError(f"unknown RL task {self.config.task!r}; choose one of: {', '.join(RL_TASKS)}")
        if self.config.external_force_mode not in EXTERNAL_FORCE_MODES:
            raise ValueError(
                f"unknown external force mode {self.config.external_force_mode!r}; "
                f"choose one of: {', '.join(EXTERNAL_FORCE_MODES)}"
            )
        if self.config.external_force_direction_mode not in EXTERNAL_FORCE_DIRECTION_MODES:
            raise ValueError(
                f"unknown external force direction mode {self.config.external_force_direction_mode!r}; "
                f"choose one of: {', '.join(EXTERNAL_FORCE_DIRECTION_MODES)}"
            )
        if not 0.0 <= float(self.config.external_force_lateral_probability) <= 1.0:
            raise ValueError("external force lateral probability must be in [0, 1]")
        if self.config.velocity_command_frame not in VELOCITY_COMMAND_FRAMES:
            raise ValueError(
                f"unknown velocity command frame {self.config.velocity_command_frame!r}; "
                f"choose one of: {', '.join(VELOCITY_COMMAND_FRAMES)}"
            )
        if self.config.observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"unknown observation mode {self.config.observation_mode!r}; "
                f"choose one of: {', '.join(OBSERVATION_MODES)}"
            )
        if self.config.policy_action_mode not in POLICY_ACTION_MODES:
            raise ValueError(
                f"unknown policy action mode {self.config.policy_action_mode!r}; "
                f"choose one of: {', '.join(POLICY_ACTION_MODES)}"
            )
        if self.config.policy_action_mode in ONNX_POLICY_ACTION_MODES and not self.config.onnx_policy_path:
            raise ValueError(f"{self.config.policy_action_mode} action mode requires --onnx-policy")
        if self.config.fullbody_reference_mode not in FULLBODY_REFERENCE_MODES:
            raise ValueError(
                f"unknown full-body reference mode {self.config.fullbody_reference_mode!r}; "
                f"choose one of: {', '.join(FULLBODY_REFERENCE_MODES)}"
            )
        if self.config.force_reference_governor_mode not in FORCE_REFERENCE_GOVERNOR_MODES:
            raise ValueError(
                f"unknown force reference governor mode {self.config.force_reference_governor_mode!r}; "
                f"choose one of: {', '.join(FORCE_REFERENCE_GOVERNOR_MODES)}"
            )
        if self.config.force_response_router_mode not in FORCE_RESPONSE_ROUTER_MODES:
            raise ValueError(
                f"unknown force response router mode {self.config.force_response_router_mode!r}; "
                f"choose one of: {', '.join(FORCE_RESPONSE_ROUTER_MODES)}"
            )
        if self.config.force_response_profile not in FORCE_RESPONSE_PROFILES:
            raise ValueError(
                f"unknown force response profile {self.config.force_response_profile!r}; "
                f"choose one of: {', '.join(FORCE_RESPONSE_PROFILES)}"
            )
        if self.config.force_safety_trigger_source not in FORCE_SAFETY_TRIGGER_SOURCES:
            raise ValueError(
                f"unknown force safety trigger source {self.config.force_safety_trigger_source!r}; "
                f"choose one of: {', '.join(FORCE_SAFETY_TRIGGER_SOURCES)}"
            )
        if (
            self.config.force_safety_trigger_source in HISTORY_ESTIMATOR_FORCE_SAFETY_TRIGGER_SOURCES
            and not self.config.force_safety_history_estimator_path
        ):
            raise ValueError(
                f"{self.config.force_safety_trigger_source} requires --force-safety-history-estimator"
            )
        if _uses_rollout_nominal_reference(self.config.fullbody_reference_mode) and not self.config.nominal_reference_dataset:
            raise ValueError(
                "rollout full-body reference mode requires --nominal-reference-dataset"
            )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.controller = TrotGaitController(
            GaitConfig(hip_swing=0.0, thigh_swing=0.0, calf_swing=0.0)
        )
        self.joint_names = _joint_names_for_leg_order(self.config.policy_leg_order)
        self.default_targets = _default_targets(
            self.joint_names,
            self.config.task,
            self.config.velocity_pose_profile,
        )
        self.joint_map = _find_joint_addresses(mujoco, self.model, self.joint_names)
        self.actuator_map = _find_actuators_for_joints(mujoco, self.model, self.joint_map)
        self.joint_action_size = len(self.joint_names)
        self.action_size = _policy_action_size_value(
            self.config.policy_action_mode,
            joint_action_size=self.joint_action_size,
        )
        self.observation_names = self._make_observation_names()
        self.foot_body_ids = {
            foot: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{foot}_foot")
            for foot in FEET
        }
        self.fullbody_tracking_body_ids = self._resolve_fullbody_tracking_body_ids()
        self.nominal_gait_reference = self._load_nominal_gait_reference()
        self._nominal_reference_keypoint_indices = self._make_nominal_reference_index_map("keypoint_names")
        self._nominal_reference_joint_indices = self._make_nominal_reference_index_map("action_names")
        self.external_force_body_ids = (
            self._resolve_external_force_body_ids()
            if self._external_force_config_enabled()
            else ()
        )
        self._rng = np.random.default_rng()
        self._last_action = np.zeros(self.joint_action_size, dtype=np.float64)
        self._last_onnx_action = np.zeros(self.joint_action_size, dtype=np.float64)
        self._last_residual_action = np.zeros(self.joint_action_size, dtype=np.float64)
        self._last_policy_action = np.zeros(self.joint_action_size, dtype=np.float64)
        self._last_residual_action_rate = np.zeros(self.joint_action_size, dtype=np.float64)
        self._last_safety_layer_action = np.zeros(len(SAFETY_LAYER_ACTION_NAMES), dtype=np.float64)
        self._last_safety_layer_action_rate = np.zeros(len(SAFETY_LAYER_ACTION_NAMES), dtype=np.float64)
        self._onnx_session = None
        self._onnx_input_name = ""
        self._onnx_output_name = ""
        self._onnx_normalizer = None
        self._episode_gait_frequency_hz = float(self.config.gait_frequency_hz)
        self._episode_gait_step_length = float(self.config.gait_step_length)
        self._episode_gait_swing_height = float(self.config.gait_swing_height)
        self._command_values = np.asarray(
            [
                self.config.target_forward_velocity,
                self.config.target_lateral_velocity,
                self.config.target_yaw_rate,
            ],
            dtype=np.float64,
        )
        self._previous_joint_vel = np.zeros(self.joint_action_size, dtype=np.float64)
        self._foot_air_time = {foot: 0.0 for foot in FEET}
        self._previous_contacts = {foot: False for foot in FEET}
        self._elapsed_s = 0.0
        self._push_applied = False
        self._external_force_curriculum_progress = 1.0
        self._external_force_body_id = -1
        self._external_force_body_name = ""
        self._external_force_vector = np.zeros(3, dtype=np.float64)
        self._external_force_torque_vector = np.zeros(3, dtype=np.float64)
        self._external_force_applied_by_body_root: dict[str, Any] = {}
        self._external_force_safe_limit_n = 0.0
        self._external_force_episode_peak_excess_cost = 0.0
        self._external_force_start_s = math.inf
        self._external_force_end_s = -math.inf
        self._external_force_applied_this_step = False
        self._external_force_pulses: list[ExternalForcePulse] = []
        self._external_force_springs: list[ExternalForceSpring] = []
        self._last_force_impedance_kp_scale = 1.0
        self._last_force_impedance_kd_scale = 1.0
        self._last_force_response_step = _force_response_nominal_step(tuple(self.joint_names))
        self._last_force_response_joint_kp_scales = np.ones(self.joint_action_size, dtype=np.float64)
        self._last_force_response_joint_kd_scales = np.ones(self.joint_action_size, dtype=np.float64)
        self._force_reference_governor_offset = np.zeros(3, dtype=np.float64)
        self._force_reference_governor_velocity = np.zeros(3, dtype=np.float64)
        self._force_reference_governor_gate = 0.0
        self._force_safety_detector_start_s = math.inf
        self._force_safety_detector_gate = 0.0
        self._force_safety_detector_score = 0.0
        self._force_safety_detector_active = False
        self._force_safety_detector_force_proxy = np.zeros(3, dtype=np.float64)
        self._force_safety_detector_v2_linear_baseline = np.zeros(3, dtype=np.float64)
        self._force_safety_detector_v2_angular_baseline = np.zeros(3, dtype=np.float64)
        self._force_safety_detector_v2_sample_count = 0
        self._force_safety_history_estimator = self._load_force_safety_history_estimator()
        self._force_safety_history_features: list[list[float]] = []
        self._previous_base_linear_velocity = np.zeros(3, dtype=np.float64)
        self._previous_base_angular_velocity = np.zeros(3, dtype=np.float64)
        self._previous_contact_count = 0
        self._fullbody_reference_root_position = np.zeros(3, dtype=np.float64)
        self._fullbody_reference_root_yaw = 0.0
        self._fullbody_keypoint_offsets_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_compliant_offsets_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_compliant_velocities_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_motion_targets_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_motion_target_velocities_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_reference_positions_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_reference_velocities_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_reference_driving_forces_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._fullbody_reference_interaction_forces_root = {
            name: np.zeros(3, dtype=np.float64)
            for name in FULLBODY_TRACKING_KEYPOINTS
        }
        self._control_steps = max(1, round(self.config.control_dt / self.model.opt.timestep))

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    def set_external_force_curriculum_progress(self, progress: float) -> None:
        self._external_force_curriculum_progress = max(0.0, min(1.0, float(progress)))

    def _load_nominal_gait_reference(self) -> Any | None:
        if not self.config.nominal_reference_dataset:
            return None
        from .reference.nominal_gait import NominalGaitReference

        return NominalGaitReference.load(self.config.nominal_reference_dataset)

    def _make_nominal_reference_index_map(self, metadata_key: str) -> dict[str, int]:
        if self.nominal_gait_reference is None:
            return {}
        names = self.nominal_gait_reference.metadata.get(metadata_key) or []
        return {str(name): index for index, name in enumerate(names)}

    def reset(
        self,
        *,
        seed: int | None = None,
        noise: float = 0.0,
        replay_state: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self._rng = self.np.random.default_rng(seed)

        self.mujoco.mj_resetData(self.model, self.data)
        self._clear_external_forces()
        _set_initial_pose(self.data, self.joint_map, self.default_targets)
        if len(self.data.qpos) >= 3:
            self.data.qpos[2] = self.config.init_base_z
        if noise > 0:
            for joint_name, (qpos_adr, _) in self.joint_map.items():
                self.data.qpos[qpos_adr] += float(self._rng.uniform(-noise, noise))

        self.data.qvel[:] = 0.0
        self._last_action = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        self._last_onnx_action = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        self._last_residual_action = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        self._last_policy_action = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        self._last_residual_action_rate = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        self._last_safety_layer_action = self.np.zeros(
            len(SAFETY_LAYER_ACTION_NAMES),
            dtype=self.np.float64,
        )
        self._last_safety_layer_action_rate = self.np.zeros(
            len(SAFETY_LAYER_ACTION_NAMES),
            dtype=self.np.float64,
        )
        self._sample_gait_parameters()
        self._sample_command()
        self._external_force_safe_limit_n = _sample_external_force_safe_limit(
            self._rng,
            min_n=self.config.external_force_safe_limit_min_n,
            max_n=self.config.external_force_safe_limit_max_n,
        )
        self._external_force_episode_peak_excess_cost = 0.0
        self._elapsed_s = 0.0
        self._push_applied = False
        self._external_force_applied_this_step = False
        self._force_reference_governor_offset[:] = 0.0
        self._force_reference_governor_velocity[:] = 0.0
        self._force_reference_governor_gate = 0.0
        self._last_force_impedance_kp_scale = 1.0
        self._last_force_impedance_kd_scale = 1.0
        self._last_force_response_step = _force_response_nominal_step(tuple(self.joint_names))
        self._last_force_response_joint_kp_scales[:] = 1.0
        self._last_force_response_joint_kd_scales[:] = 1.0
        self._force_safety_detector_start_s = math.inf
        self._force_safety_detector_gate = 0.0
        self._force_safety_detector_score = 0.0
        self._force_safety_detector_active = False
        self._force_safety_detector_force_proxy[:] = 0.0
        self._force_safety_detector_v2_linear_baseline[:] = 0.0
        self._force_safety_detector_v2_angular_baseline[:] = 0.0
        self._force_safety_detector_v2_sample_count = 0
        self._force_safety_history_features.clear()
        self.mujoco.mj_forward(self.model, self.data)
        self._settle()
        self._reset_fullbody_tracking_reference()
        self._sample_external_force()
        if replay_state is not None:
            self._apply_replay_state(replay_state)
        self._previous_joint_vel = self._joint_vel_array()
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        self._foot_air_time = {foot: 0.0 for foot in FEET}
        self._previous_contacts = {foot: bool(contacts.get(foot, False)) for foot in FEET}
        self._previous_contact_count = sum(1 for value in contacts.values() if value)
        self._previous_base_linear_velocity = self.np.asarray(self.data.qvel[:3], dtype=self.np.float64).copy()
        self._previous_base_angular_velocity = self.np.asarray(self.data.qvel[3:6], dtype=self.np.float64).copy()
        obs = self.observation()
        return obs, self._info(reward_terms={})

    def _apply_replay_state(self, state: dict[str, Any]) -> None:
        if "qpos" in state:
            qpos = self.np.asarray(state["qpos"], dtype=self.np.float64).reshape(-1)
            if qpos.shape != self.data.qpos.shape:
                raise ValueError(f"replay qpos shape {qpos.shape} does not match {self.data.qpos.shape}")
            self.data.qpos[:] = qpos
        if "qvel" in state:
            qvel = self.np.asarray(state["qvel"], dtype=self.np.float64).reshape(-1)
            if qvel.shape != self.data.qvel.shape:
                raise ValueError(f"replay qvel shape {qvel.shape} does not match {self.data.qvel.shape}")
            self.data.qvel[:] = qvel
        if "ctrl" in state:
            ctrl = self.np.asarray(state["ctrl"], dtype=self.np.float64).reshape(-1)
            if ctrl.shape != self.data.ctrl.shape:
                raise ValueError(f"replay ctrl shape {ctrl.shape} does not match {self.data.ctrl.shape}")
            self.data.ctrl[:] = ctrl
        if "t" in state:
            self._elapsed_s = max(0.0, float(state["t"]))
        if "command" in state:
            command = self.np.asarray(state["command"], dtype=self.np.float64).reshape(-1)
            if command.shape != (3,):
                raise ValueError(f"replay command must have shape (3,), got {command.shape}")
            self._command_values = command.copy()
        self._restore_replay_action_state(state)
        if "external_force_safe_limit_n" in state:
            self._external_force_safe_limit_n = max(0.0, float(state["external_force_safe_limit_n"]))
        if "external_force_schedule" in state:
            self._restore_external_force_schedule(state["external_force_schedule"])
        if "force_reference_governor_offset" in state:
            self._force_reference_governor_offset[:] = self.np.asarray(
                state["force_reference_governor_offset"], dtype=self.np.float64
            ).reshape(3)
        if "force_reference_governor_velocity" in state:
            self._force_reference_governor_velocity[:] = self.np.asarray(
                state["force_reference_governor_velocity"], dtype=self.np.float64
            ).reshape(3)
        if "force_reference_governor_gate" in state:
            self._force_reference_governor_gate = max(0.0, min(1.0, float(state["force_reference_governor_gate"])))
        self.mujoco.mj_forward(self.model, self.data)

    def _restore_replay_action_state(self, state: dict[str, Any]) -> None:
        action_fields = (
            ("last_action", "_last_action", self.joint_action_size),
            ("onnx_action", "_last_onnx_action", self.joint_action_size),
            ("residual_action", "_last_residual_action", self.joint_action_size),
            ("final_action", "_last_policy_action", self.joint_action_size),
            ("safety_layer_action", "_last_safety_layer_action", len(SAFETY_LAYER_ACTION_NAMES)),
        )
        for key, attr, size in action_fields:
            if key not in state:
                continue
            values = self.np.asarray(state[key], dtype=self.np.float64).reshape(-1)
            if values.shape != (size,):
                raise ValueError(f"replay {key} must have shape ({size},), got {values.shape}")
            getattr(self, attr)[:] = values
        if "applied_action" in state:
            values = self.np.asarray(state["applied_action"], dtype=self.np.float64).reshape(-1)
            if values.shape != (self.joint_action_size,):
                raise ValueError(
                    f"replay applied_action must have shape ({self.joint_action_size},), got {values.shape}"
                )
            self._last_action[:] = values

    def _restore_external_force_schedule(self, schedule: dict[str, Any]) -> None:
        self._external_force_pulses = []
        self._external_force_springs = []
        self._external_force_body_id = -1
        self._external_force_body_name = ""
        self._external_force_start_s = math.inf
        self._external_force_end_s = -math.inf

        for item in schedule.get("pulses", []) or []:
            body_name = str(item.get("body_name", ""))
            body_id = int(item.get("body_id", -1))
            if body_id < 0 and body_name:
                body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"replay pulse has unknown body {body_name!r}")
            if not body_name:
                body_name = str(self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id))
            self._external_force_pulses.append(
                ExternalForcePulse(
                    body_id=body_id,
                    body_name=body_name,
                    start_s=float(item.get("start_s", 0.0)),
                    end_s=float(item.get("end_s", 0.0)),
                    force_vector=self.np.asarray(item.get("force_vector", [0.0, 0.0, 0.0]), dtype=self.np.float64),
                    torque_vector=self.np.asarray(item.get("torque_vector", [0.0, 0.0, 0.0]), dtype=self.np.float64),
                )
            )

        for item in schedule.get("springs", []) or []:
            body_name = str(item.get("body_name", ""))
            body_id = int(item.get("body_id", -1))
            if body_id < 0 and body_name:
                body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"replay spring has unknown body {body_name!r}")
            if not body_name:
                body_name = str(self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id))
            frame_origin = item.get("frame_origin_world")
            frame_rotation = item.get("frame_rotation_world")
            self._external_force_springs.append(
                ExternalForceSpring(
                    body_id=body_id,
                    body_name=body_name,
                    spring_type=str(item.get("spring_type", "resistive")),
                    start_s=float(item.get("start_s", 0.0)),
                    end_s=float(item.get("end_s", 0.0)),
                    direction_root=self.np.asarray(item.get("direction_root", [0.0, 0.0, 0.0]), dtype=self.np.float64),
                    anchor_root=self.np.asarray(item.get("anchor_root", [0.0, 0.0, 0.0]), dtype=self.np.float64),
                    anchor_velocity_root=self.np.asarray(
                        item.get("anchor_velocity_root", [0.0, 0.0, 0.0]), dtype=self.np.float64
                    ),
                    anchor_target_root=self.np.asarray(
                        item.get("anchor_target_root", [0.0, 0.0, 0.0]), dtype=self.np.float64
                    ),
                    torque_root=self.np.asarray(item.get("torque_root", [0.0, 0.0, 0.0]), dtype=self.np.float64),
                    stiffness=float(item.get("stiffness", 0.0)),
                    damping=float(item.get("damping", 0.0)),
                    cap_n=float(item.get("cap_n", 0.0)),
                    frame_origin_world=(
                        None if frame_origin is None else self.np.asarray(frame_origin, dtype=self.np.float64)
                    ),
                    frame_rotation_world=(
                        None if frame_rotation is None else self.np.asarray(frame_rotation, dtype=self.np.float64)
                    ),
                    initialized=bool(item.get("initialized", False))
                    and frame_origin is not None
                    and frame_rotation is not None,
                )
            )
        self._refresh_external_force_summary_from_schedule()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action_array = self.np.asarray(action, dtype=self.np.float64)
        if action_array.shape != (self.action_size,):
            raise ValueError(f"action must have shape ({self.action_size},), got {action_array.shape}")
        action_array = self.np.clip(action_array, -1.0, 1.0)
        previous_action = self._last_action.copy()
        previous_residual_action = self._last_residual_action.copy()
        previous_safety_layer_action = self._last_safety_layer_action.copy()
        previous_joint_vel = self._joint_vel_array()
        previous_ctrl = self.data.ctrl.copy()
        self._update_force_safety_detector()
        onnx_action = (
            self._onnx_expert_action()
            if self.config.policy_action_mode in ONNX_POLICY_ACTION_MODES
            else None
        )
        safety_layer_action = self.np.zeros(len(SAFETY_LAYER_ACTION_NAMES), dtype=self.np.float64)
        compose_action = action_array
        if self.config.policy_action_mode == "onnx_safety_layer":
            safety_layer_action = action_array.copy()
            compose_action = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        final_action, base_action, residual_action = _compose_policy_action(
            compose_action,
            mode=self.config.policy_action_mode,
            onnx_action=onnx_action,
            residual_scale=self.config.residual_action_scale,
            np_module=self.np,
        )
        smoothing = max(0.0, min(0.98, float(self.config.action_smoothing)))
        filtered_action = smoothing * previous_action + (1.0 - smoothing) * final_action
        self._last_safety_layer_action = self.np.asarray(safety_layer_action, dtype=self.np.float64).copy()
        self._last_safety_layer_action_rate = self._last_safety_layer_action - previous_safety_layer_action
        self._apply_action(filtered_action)
        self._maybe_apply_push()
        self._last_onnx_action = self.np.asarray(base_action, dtype=self.np.float64).copy()
        self._last_residual_action = self.np.asarray(residual_action, dtype=self.np.float64).copy()
        self._last_policy_action = self.np.asarray(final_action, dtype=self.np.float64).copy()
        self._last_residual_action_rate = self._last_residual_action - previous_residual_action
        self._last_action = filtered_action
        self._clear_external_forces()
        self._maybe_apply_external_force()
        self._update_force_reference_governor()

        for _ in range(self._control_steps):
            self.mujoco.mj_step(self.model, self.data)
        self._elapsed_s += self._control_steps * self.model.opt.timestep

        reward, raw_reward_terms, reward_terms = self._reward(
            previous_action,
            previous_joint_vel,
            previous_ctrl=previous_ctrl,
        )
        self._previous_joint_vel = self._joint_vel_array()
        reward *= self.config.reward_scale
        raw_reward_terms = {
            name: value * self.config.reward_scale
            for name, value in raw_reward_terms.items()
        }
        reward_terms = {
            name: value * self.config.reward_scale
            for name, value in reward_terms.items()
        }
        terminated = self._terminated()
        truncated = self._elapsed_s >= self.config.episode_length_s
        return self.observation(), reward, terminated, truncated, self._info(
            reward_terms=reward_terms,
            reward_raw_terms=raw_reward_terms,
        )

    def sample_action(self, scale: float = 1.0) -> Any:
        scale = max(0.0, min(1.0, scale))
        return self._rng.uniform(-scale, scale, self.action_size)

    def zero_action(self) -> Any:
        return self.np.zeros(self.action_size, dtype=self.np.float64)

    def amp_observation(self) -> Any:
        joint_pos_rel = []
        joint_vel = []
        for joint_name in self.joint_names:
            qpos_adr, qvel_adr = self.joint_map[joint_name]
            joint_pos_rel.append(float(self.data.qpos[qpos_adr]) - self.default_targets[joint_name])
            joint_vel.append(float(self.data.qvel[qvel_adr]))

        return self.np.asarray(
            [
                float(self.data.qpos[2]) if len(self.data.qpos) >= 3 else 0.0,
                *[float(v) for v in self._projected_gravity()],
                *[float(v) for v in self.data.qvel[:3]],
                *[float(v) for v in self.data.qvel[3:6]],
                *joint_pos_rel,
                *joint_vel,
            ],
            dtype=self.np.float32,
        )

    def observation(self) -> Any:
        if self.config.task in VELOCITY_TASKS:
            return self.observation_for_mode(self.config.observation_mode)

        return self._stand_observation()

    def observation_for_mode(
        self,
        mode: str,
        *,
        include_external_force_observation: bool | None = None,
    ) -> Any:
        if mode not in OBSERVATION_MODES:
            raise ValueError(f"unknown observation mode {mode!r}; choose one of: {', '.join(OBSERVATION_MODES)}")
        if self.config.task in VELOCITY_TASKS and mode == "privileged":
            return self._velocity_privileged_observation()
        if self.config.task in VELOCITY_TASKS:
            include_force = (
                self.config.include_external_force_observation
                if include_external_force_observation is None
                else include_external_force_observation
            )
            return self._velocity_policy_observation(include_external_force_observation=include_force)
        return self._stand_observation()

    def _onnx_policy_observation(self) -> Any:
        if self.config.task in VELOCITY_TASKS:
            return self._velocity_policy_observation(include_external_force_observation=False)
        return self.observation_for_mode("policy")

    def _onnx_expert_action(self) -> Any:
        self._ensure_onnx_policy_loaded()
        policy_obs = self.np.asarray(self._onnx_policy_observation(), dtype=self.np.float32).reshape(1, -1)
        if self._onnx_normalizer is not None:
            policy_obs = self._onnx_normalizer(policy_obs)
        action = self._onnx_session.run([self._onnx_output_name], {self._onnx_input_name: policy_obs})[0]
        action = self.np.asarray(action, dtype=self.np.float64).reshape(-1)
        if action.shape != (self.joint_action_size,):
            raise ValueError(f"ONNX action must have shape ({self.joint_action_size},), got {action.shape}")
        return self.np.clip(action, -1.0, 1.0)

    def _ensure_onnx_policy_loaded(self) -> None:
        if self._onnx_session is not None:
            return
        if not self.config.onnx_policy_path:
            raise ValueError(f"{self.config.policy_action_mode} action mode requires --onnx-policy")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is missing. Install with: pip install -e '.[reference]'") from exc
        policy_path = Path(self.config.onnx_policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
        self._onnx_session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self._onnx_input_name = self._onnx_session.get_inputs()[0].name
        self._onnx_output_name = self._onnx_session.get_outputs()[0].name
        if self.config.onnx_normalizer_checkpoint:
            from .onnx_rollout import _load_actor_obs_normalizer

            self._onnx_normalizer = _load_actor_obs_normalizer(Path(self.config.onnx_normalizer_checkpoint))

    def _load_force_safety_history_estimator(self) -> LinearHistoryEventEstimator | None:
        path = self.config.force_safety_history_estimator_path
        if not path:
            return None
        estimator_path = Path(path)
        if not estimator_path.exists():
            raise FileNotFoundError(f"force safety history estimator not found: {estimator_path}")
        return LinearHistoryEventEstimator.load(estimator_path)

    def _force_safety_history_current_feature(self) -> list[float]:
        joint_positions = []
        joint_velocities = []
        for joint_name in self.joint_names:
            qpos_adr, qvel_adr = self.joint_map[joint_name]
            joint_positions.append(float(self.data.qpos[qpos_adr]))
            joint_velocities.append(float(self.data.qvel[qvel_adr]))
        return build_proprioceptive_feature(
            command=[
                float(value)
                for value in self._command(include_force_reference_governor=False)
            ],
            base_angular_velocity=[float(value) for value in self.data.qvel[3:6]],
            projected_gravity=[float(value) for value in self._projected_gravity()],
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            last_action=[float(value) for value in self._last_action],
        )

    def _append_force_safety_history_feature(self) -> None:
        estimator = self._force_safety_history_estimator
        if estimator is None:
            return
        self._force_safety_history_features.append(self._force_safety_history_current_feature())
        if len(self._force_safety_history_features) > estimator.history_steps:
            self._force_safety_history_features = self._force_safety_history_features[-estimator.history_steps :]

    def _force_safety_history_detector_step(
        self,
        *,
        linear_acceleration: Any,
        previous_active: bool,
    ) -> ForceSafetyDetectorStep:
        estimator = self._force_safety_history_estimator
        if estimator is None:
            raise ValueError("deployable_history requires a loaded force safety history estimator")

        self._append_force_safety_history_feature()
        first = self._force_safety_history_features[0]
        padded_history = [first] * max(0, estimator.history_steps - len(self._force_safety_history_features))
        padded_history.extend(self._force_safety_history_features)

        score = float(estimator.score_feature(build_history_feature(padded_history)))
        threshold = estimator.release_threshold if bool(previous_active) else estimator.threshold
        active = bool(score >= threshold)
        force_proxy = tuple(float(value) for value in self.np.asarray(linear_acceleration, dtype=self.np.float64)[:3])
        return ForceSafetyDetectorStep(score=score, active=active, force_proxy=force_proxy)

    def _velocity_policy_observation(self, *, include_external_force_observation: bool) -> Any:
        qvel = self.data.qvel
        projected_gravity = self._projected_gravity()
        joint_pos_rel = []
        joint_vel = []
        for joint_name in self.joint_names:
            qpos_adr, qvel_adr = self.joint_map[joint_name]
            joint_pos_rel.append(float(self.data.qpos[qpos_adr]) - self.default_targets[joint_name])
            joint_vel.append(float(qvel[qvel_adr]))

        return self.np.asarray(
            [
                *[float(v) for v in qvel[3:6]],
                *projected_gravity,
                *self._command().tolist(),
                *joint_pos_rel,
                *joint_vel,
                *self._last_action.tolist(),
                *self._external_force_observation_values(force_include=include_external_force_observation),
            ],
            dtype=self.np.float32,
        )

    def _stand_observation(self) -> Any:
        qvel = self.data.qvel
        projected_gravity = self._projected_gravity()
        joint_pos_rel = []
        joint_vel = []
        for joint_name in self.joint_names:
            qpos_adr, qvel_adr = self.joint_map[joint_name]
            joint_pos_rel.append(float(self.data.qpos[qpos_adr]) - self.default_targets[joint_name])
            joint_vel.append(float(qvel[qvel_adr]))
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        contact_values = [1.0 if contacts[foot] else 0.0 for foot in FEET]
        command_values = self._command().tolist() if self.config.include_command else []
        return self.np.asarray(
            [
                *projected_gravity,
                *[float(v) for v in qvel[:3]],
                *[float(v) for v in qvel[3:6]],
                *command_values,
                *joint_pos_rel,
                *joint_vel,
                *self._last_action.tolist(),
                *contact_values,
            ],
            dtype=self.np.float32,
        )

    def _velocity_privileged_observation(self) -> Any:
        policy_obs = self._velocity_policy_observation(include_external_force_observation=False)
        root_rotation = self._root_rotation_matrix()
        qvel = self.np.asarray(self.data.qvel, dtype=self.np.float64)
        root_position = self.np.asarray(self.data.qpos[:3], dtype=self.np.float64)
        target_root_position, target_root_yaw = self._fullbody_reference_root_target()
        target_keypoints = self._fullbody_target_keypoints_world(target_root_position, target_root_yaw)
        root_error_body = root_rotation.T @ (target_root_position - root_position)
        yaw_error = _wrap_angle(self._base_yaw() - target_root_yaw)
        tracking_linear_velocity = self._velocity_tracking_linear_velocity(qvel[:3])
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        reference_velocity_scale = self._fullbody_privileged_reference_velocity_scale()
        reference_force_scale = self._fullbody_privileged_force_scale()

        keypoint_diff_root = []
        keypoint_vel_root = []
        compliant_offsets = []
        compliant_velocities = []
        reference_velocities = []
        reference_driving_forces = []
        reference_interaction_forces = []
        actual_interaction_forces = []
        interaction_force_errors = []
        for name in FULLBODY_TRACKING_KEYPOINTS:
            body_id = self.fullbody_tracking_body_ids.get(name, -1)
            target = target_keypoints.get(name, root_position)
            if body_id >= 0:
                actual = self.np.asarray(self.data.xpos[body_id], dtype=self.np.float64)
                velocity = self.np.asarray(self.data.cvel[body_id][3:6], dtype=self.np.float64)
            else:
                actual = root_position
                velocity = self.np.zeros(3, dtype=self.np.float64)
            keypoint_diff_root.extend(root_rotation.T @ (target - actual))
            keypoint_vel_root.extend(root_rotation.T @ velocity)
            compliant_offsets.extend(self._fullbody_compliant_offsets_root[name])
            compliant_velocities.extend(self._fullbody_compliant_velocities_root[name])
            reference_velocity = self._fullbody_reference_velocities_root[name]
            driving_force = self._fullbody_reference_driving_forces_root[name]
            reference_force = self._fullbody_reference_interaction_forces_root[name]
            actual_force = self._fullbody_actual_interaction_force_root(name)
            reference_velocities.extend(
                self.np.clip(reference_velocity / reference_velocity_scale, -1.0, 1.0)
            )
            reference_driving_forces.extend(
                self.np.clip(driving_force / reference_force_scale, -1.0, 1.0)
            )
            reference_interaction_forces.extend(
                self.np.clip(reference_force / reference_force_scale, -1.0, 1.0)
            )
            actual_interaction_forces.extend(
                self.np.clip(actual_force / reference_force_scale, -1.0, 1.0)
            )
            interaction_force_errors.extend(
                self.np.clip((actual_force - reference_force) / reference_force_scale, -1.0, 1.0)
            )

        contact_values = [1.0 if contacts.get(foot, False) else 0.0 for foot in FEET]
        contact_forces = []
        for foot in FEET:
            body_id = self.foot_body_ids.get(foot, -1)
            if body_id >= 0:
                force_world = self.np.asarray(self.data.cfrc_ext[body_id][3:6], dtype=self.np.float64)
                force_root = root_rotation.T @ force_world
            else:
                force_root = self.np.zeros(3, dtype=self.np.float64)
            contact_forces.extend(self.np.clip(force_root / 100.0, -5.0, 5.0))

        ctrl_scale = max(1.0, float(self.config.pd.torque_limit))
        ctrl_values = self.np.clip(self.np.asarray(self.data.ctrl, dtype=self.np.float64) / ctrl_scale, -2.0, 2.0)

        return self.np.asarray(
            [
                *policy_obs.tolist(),
                *[float(v) for v in tracking_linear_velocity],
                float(self.data.qpos[2]) if len(self.data.qpos) >= 3 else 0.0,
                *self._external_force_observation_values(force_include=True),
                *[float(v) for v in root_error_body],
                float(yaw_error),
                float(self._external_force_active_scale()),
                float(self._external_force_recovery_scale()),
                *(
                    [
                        _normalize_external_force_safe_limit(
                            self._external_force_safe_limit_n,
                            max_n=self.config.external_force_safe_limit_max_n,
                        )
                    ]
                    if self.config.external_force_safe_limit_max_n > 0.0
                    else []
                ),
                *[float(v) for v in keypoint_diff_root],
                *[float(v) for v in keypoint_vel_root],
                *[float(v) for v in compliant_offsets],
                *[float(v) for v in compliant_velocities],
                *contact_values,
                *[float(v) for v in contact_forces],
                *[float(v) for v in ctrl_values],
                *[float(v) for v in reference_velocities],
                *[float(v) for v in reference_driving_forces],
                *[float(v) for v in reference_interaction_forces],
                *[float(v) for v in actual_interaction_forces],
                *[float(v) for v in interaction_force_errors],
            ],
            dtype=self.np.float32,
        )

    def make_frame(self, *, frame_t: float | None = None) -> dict[str, Any]:
        qpos = [float(v) for v in self.data.qpos]
        qvel = [float(v) for v in self.data.qvel]
        ctrl = [float(v) for v in self.data.ctrl]
        base = BaseState(
            position=qpos[:3] if len(qpos) >= 3 else [0.0, 0.0, 0.0],
            quaternion=qpos[3:7] if len(qpos) >= 7 else [1.0, 0.0, 0.0, 0.0],
            linear_velocity=qvel[:3] if len(qvel) >= 3 else [0.0, 0.0, 0.0],
            angular_velocity=qvel[3:6] if len(qvel) >= 6 else [0.0, 0.0, 0.0],
        )
        return {
            "t": round(self._elapsed_s if frame_t is None else frame_t, 6),
            "qpos": qpos,
            "qvel": qvel,
            "ctrl": ctrl,
            "base": asdict(base),
            "joints": {
                joint_name: float(self.data.qpos[qpos_adr])
                for joint_name, (qpos_adr, _) in self.joint_map.items()
            },
            "contacts": _foot_contacts(self.mujoco, self.model, self.data),
            "gait_phase": {foot: 0.0 for foot in FEET},
        }

    def _external_force_schedule_info(self) -> dict[str, Any]:
        return {
            "pulses": [
                {
                    "body_id": int(pulse.body_id),
                    "body_name": pulse.body_name,
                    "start_s": float(pulse.start_s),
                    "end_s": float(pulse.end_s),
                    "force_vector": [float(v) for v in pulse.force_vector],
                    "torque_vector": [float(v) for v in pulse.torque_vector],
                }
                for pulse in self._external_force_pulses
            ],
            "springs": [
                {
                    "body_id": int(spring.body_id),
                    "body_name": spring.body_name,
                    "spring_type": spring.spring_type,
                    "start_s": float(spring.start_s),
                    "end_s": float(spring.end_s),
                    "direction_root": [float(v) for v in spring.direction_root],
                    "anchor_root": [float(v) for v in spring.anchor_root],
                    "anchor_velocity_root": [float(v) for v in spring.anchor_velocity_root],
                    "anchor_target_root": [float(v) for v in spring.anchor_target_root],
                    "torque_root": [float(v) for v in spring.torque_root],
                    "stiffness": float(spring.stiffness),
                    "damping": float(spring.damping),
                    "cap_n": float(spring.cap_n),
                    "frame_origin_world": (
                        None
                        if spring.frame_origin_world is None
                        else [float(v) for v in spring.frame_origin_world]
                    ),
                    "frame_rotation_world": (
                        None
                        if spring.frame_rotation_world is None
                        else [
                            [float(value) for value in row]
                            for row in spring.frame_rotation_world
                        ]
                    ),
                    "initialized": bool(spring.initialized),
                }
                for spring in self._external_force_springs
            ],
        }

    def _settle(self) -> None:
        settle_steps = max(0, round(self.config.reset_settle_s / self.model.opt.timestep))
        zero_action = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        self._clear_external_forces()
        for _ in range(settle_steps):
            self._apply_action(zero_action)
            self.mujoco.mj_step(self.model, self.data)
        self._clear_external_forces()
        self.mujoco.mj_forward(self.model, self.data)

    def _apply_action(self, action: Any) -> None:
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        if self.config.force_response_router_mode != "off":
            router_step = self._force_response_router_step(contacts)
            self._last_force_response_step = router_step
            self._last_force_response_joint_kp_scales = self.np.asarray(
                router_step.joint_kp_scales,
                dtype=self.np.float64,
            )
            self._last_force_response_joint_kd_scales = self.np.asarray(
                router_step.joint_kd_scales,
                dtype=self.np.float64,
            )
            kp_scales = self._last_force_response_joint_kp_scales
            kd_scales = self._last_force_response_joint_kd_scales
            self._last_force_impedance_kp_scale = (
                float(kp_scales.min()) if float(kp_scales.min()) < 1.0 else float(kp_scales.max())
            )
            self._last_force_impedance_kd_scale = (
                float(kd_scales.max()) if float(kd_scales.max()) > 1.0 else float(kd_scales.min())
            )
        else:
            kp_scale, kd_scale = self._force_impedance_scales()
            self._last_force_impedance_kp_scale = kp_scale
            self._last_force_impedance_kd_scale = kd_scale
            self._last_force_response_step = _force_response_nominal_step(tuple(self.joint_names))
            self._last_force_response_joint_kp_scales[:] = 1.0
            self._last_force_response_joint_kd_scales[:] = 1.0
        for index, joint_name in enumerate(self.joint_names):
            qpos_adr, qvel_adr = self.joint_map[joint_name]
            target = self.default_targets[joint_name] + self.config.action_scale * float(action[index])
            actuator_id = self.actuator_map[joint_name]
            if self.config.force_response_router_mode != "off":
                pd_config = self._force_response_pd_config_for_joint(index)
            else:
                pd_config = self._force_impedance_pd_config_for_joint(
                    joint_name,
                    kp_scale,
                    kd_scale,
                    contacts,
                )
            self.data.ctrl[actuator_id] = compute_pd_torque(
                float(self.data.qpos[qpos_adr]),
                float(self.data.qvel[qvel_adr]),
                target,
                config=pd_config,
            )

    def _force_response_pd_config_for_joint(self, joint_index: int) -> PDConfig:
        kp_scale = float(self._last_force_response_joint_kp_scales[joint_index])
        kd_scale = float(self._last_force_response_joint_kd_scales[joint_index])
        if kp_scale == 1.0 and kd_scale == 1.0:
            return self.config.pd
        return replace(
            self.config.pd,
            kp=self.config.pd.kp * kp_scale,
            kd=self.config.pd.kd * kd_scale,
        )

    def _force_impedance_pd_config_for_joint(
        self,
        joint_name: str,
        kp_scale: float,
        kd_scale: float,
        contacts: dict[str, bool],
    ) -> PDConfig:
        if not _force_impedance_joint_is_selected(
            joint_name,
            self.config.force_impedance_joint_scope,
            contacts,
        ):
            return self.config.pd
        if kp_scale == 1.0 and kd_scale == 1.0:
            return self.config.pd
        return replace(
            self.config.pd,
            kp=self.config.pd.kp * kp_scale,
            kd=self.config.pd.kd * kd_scale,
        )

    def _force_impedance_scales(self) -> tuple[float, float]:
        kp_scale, kd_scale = self._force_impedance_target_scales()
        return _force_impedance_scale_value(
            t_s=float(self._elapsed_s),
            windows=self._force_safety_windows(),
            mode=self.config.force_impedance_mode,
            kp_scale=kp_scale,
            kd_scale=kd_scale,
            delay_s=self.config.force_impedance_delay_s,
            hold_s=self.config.force_impedance_hold_s,
            recovery_s=self.config.force_impedance_recovery_s,
            tail_kp_scale=self.config.force_impedance_tail_kp_scale,
            tail_kd_scale=self.config.force_impedance_tail_kd_scale,
        )

    def _force_impedance_target_scales(self) -> tuple[float, float]:
        kp_scale = self.config.force_impedance_kp_scale
        kd_scale = self.config.force_impedance_kd_scale
        if self.config.policy_action_mode == "onnx_safety_layer":
            kp_scale = max(0.0, min(1.0, kp_scale * (1.0 + 0.25 * float(self._last_safety_layer_action[0]))))
            kd_scale = max(0.0, kd_scale * (1.0 + 0.50 * float(self._last_safety_layer_action[1])))
        return float(kp_scale), float(kd_scale)

    def _force_response_router_step(self, contacts: dict[str, bool]) -> ForceResponseRouterStep:
        base_kp_scale, base_kd_scale = self._force_impedance_target_scales()
        external_force_body = (
            self._external_force_body_name
            if self.config.force_response_router_mode == "semantic_oracle"
            else None
        )
        return _semantic_force_response_router_value(
            t_s=float(self._elapsed_s),
            windows=self._force_safety_windows(),
            mode=self.config.force_response_router_mode,
            profile=self.config.force_response_profile,
            external_force_body=external_force_body,
            contacts=contacts,
            joint_names=tuple(self.joint_names),
            base_joint_scope=self.config.force_impedance_joint_scope,
            base_kp_scale=base_kp_scale,
            base_kd_scale=base_kd_scale,
            foot_kp_scale=self.config.force_response_foot_kp_scale,
            foot_kd_scale=self.config.force_response_foot_kd_scale,
            delay_s=self.config.force_impedance_delay_s,
            hold_s=self.config.force_impedance_hold_s,
            recovery_s=self.config.force_impedance_recovery_s,
            tail_kp_scale=self.config.force_impedance_tail_kp_scale,
            tail_kd_scale=self.config.force_impedance_tail_kd_scale,
        )

    def _update_force_reference_governor(self) -> None:
        force_tracking_frame = self._velocity_tracking_linear_velocity(self._force_safety_force_vector())
        admittance = self.config.force_reference_governor_admittance_mps_per_n
        offset_clip = self.config.force_reference_governor_offset_clip_m
        velocity_clip = self.config.force_reference_governor_velocity_clip_mps
        governor_mode = self.config.force_reference_governor_mode
        if (
            self.config.force_response_router_mode != "off"
            and not self._last_force_response_step.governor_enabled
        ):
            governor_mode = "off"
        if self.config.policy_action_mode == "onnx_safety_layer":
            admittance *= max(0.0, 1.0 + float(self._last_safety_layer_action[2]))
            clip_scale = max(0.2, 1.0 + float(self._last_safety_layer_action[3]))
            offset_clip *= clip_scale
            velocity_clip *= clip_scale
        step = _integrate_force_reference_governor_value(
            np_module=self.np,
            previous_offset=self._force_reference_governor_offset,
            force_tracking_frame=force_tracking_frame,
            t_s=float(self._elapsed_s),
            windows=self._force_safety_windows(),
            mode=governor_mode,
            admittance_mps_per_n=admittance,
            damping=self.config.force_reference_governor_damping,
            offset_clip_m=offset_clip,
            velocity_clip_mps=velocity_clip,
            dt=self.config.control_dt,
            delay_s=self.config.force_reference_governor_delay_s,
            hold_s=self.config.force_reference_governor_hold_s,
            recovery_s=self.config.force_reference_governor_recovery_s,
            tail_admittance_scale=self.config.force_reference_governor_tail_admittance_scale,
            tail_offset_clip_scale=self.config.force_reference_governor_tail_offset_clip_scale,
            tail_velocity_clip_scale=self.config.force_reference_governor_tail_velocity_clip_scale,
        )
        self._force_reference_governor_offset[:] = self.np.asarray(step.offset, dtype=self.np.float64)
        self._force_reference_governor_velocity[:] = self.np.asarray(step.velocity, dtype=self.np.float64)
        self._force_reference_governor_gate = float(step.gate)

    def _update_force_safety_detector(self) -> None:
        base_linear_velocity = self.np.asarray(self.data.qvel[:3], dtype=self.np.float64)
        base_angular_velocity = self.np.asarray(self.data.qvel[3:6], dtype=self.np.float64)
        dt = max(1e-6, float(self.config.control_dt))
        linear_acceleration = (base_linear_velocity - self._previous_base_linear_velocity) / dt
        angular_acceleration = (base_angular_velocity - self._previous_base_angular_velocity) / dt
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        contact_count = sum(1 for value in contacts.values() if value)

        source = str(self.config.force_safety_trigger_source)
        if source in ANY_DEPLOYABLE_FORCE_SAFETY_TRIGGER_SOURCES and not _force_safety_detector_enabled_value(
            t_s=float(self._elapsed_s),
            enable_after_s=self.config.force_safety_detector_enable_after_s,
        ):
            if source in HISTORY_ESTIMATOR_FORCE_SAFETY_TRIGGER_SOURCES:
                self._append_force_safety_history_feature()
            if source in (
                DEPLOYABLE_V2_FORCE_SAFETY_TRIGGER_SOURCES
                | DEPLOYABLE_HYBRID_FORCE_SAFETY_TRIGGER_SOURCES
                | DEPLOYABLE_HISTORY_OR_V2_FORCE_SAFETY_TRIGGER_SOURCES
                | DEPLOYABLE_HISTORY_GATED_V2_FORCE_SAFETY_TRIGGER_SOURCES
            ):
                _detector_step, linear_baseline, angular_baseline = _force_safety_detector_v2_step_value(
                    np_module=self.np,
                    base_linear_acceleration=linear_acceleration,
                    base_angular_acceleration=angular_acceleration,
                    linear_acceleration_baseline=self._force_safety_detector_v2_linear_baseline,
                    angular_acceleration_baseline=self._force_safety_detector_v2_angular_baseline,
                    previous_active=False,
                    warmup=True,
                    linear_acceleration_threshold=self.config.force_safety_detector_linear_acceleration_threshold,
                    angular_acceleration_threshold=self.config.force_safety_detector_angular_acceleration_threshold,
                )
                self._force_safety_detector_v2_linear_baseline[:] = self.np.asarray(
                    linear_baseline,
                    dtype=self.np.float64,
                )
                self._force_safety_detector_v2_angular_baseline[:] = self.np.asarray(
                    angular_baseline,
                    dtype=self.np.float64,
                )
                self._force_safety_detector_v2_sample_count += 1
            self._force_safety_detector_score = 0.0
            self._force_safety_detector_gate = 0.0
            self._force_safety_detector_active = False
            self._force_safety_detector_start_s = math.inf
            self._force_safety_detector_force_proxy[:] = 0.0
            self._previous_base_linear_velocity = base_linear_velocity.copy()
            self._previous_base_angular_velocity = base_angular_velocity.copy()
            self._previous_contact_count = contact_count
            return
        if source in DEPLOYABLE_FORCE_SAFETY_TRIGGER_SOURCES:
            max_joint_error = 0.0
            max_joint_velocity = 0.0
            for joint_name, action_value in zip(self.joint_names, self._last_action):
                qpos_adr, qvel_adr = self.joint_map[joint_name]
                target = float(self.default_targets[joint_name]) + float(self.config.action_scale) * float(action_value)
                max_joint_error = max(max_joint_error, abs(float(self.data.qpos[qpos_adr]) - target))
                max_joint_velocity = max(max_joint_velocity, abs(float(self.data.qvel[qvel_adr])))

            detector_step = _force_safety_detector_step_value(
                np_module=self.np,
                base_linear_acceleration=linear_acceleration,
                base_angular_acceleration=angular_acceleration,
                max_joint_tracking_error=max_joint_error,
                max_joint_velocity=max_joint_velocity,
                contact_count=contact_count,
                previous_contact_count=self._previous_contact_count,
                linear_acceleration_threshold=self.config.force_safety_detector_linear_acceleration_threshold,
                angular_acceleration_threshold=self.config.force_safety_detector_angular_acceleration_threshold,
                joint_tracking_error_threshold=self.config.force_safety_detector_joint_error_threshold,
                joint_velocity_threshold=self.config.force_safety_detector_joint_velocity_threshold,
                contact_loss_enabled=self.config.force_safety_detector_contact_loss,
            )
            previous_gate = self._force_safety_detector_gate
            if detector_step.active and previous_gate <= 0.0:
                self._force_safety_detector_start_s = float(self._elapsed_s)
            if detector_step.active:
                self._force_safety_detector_force_proxy[:] = self.np.asarray(
                    detector_step.force_proxy,
                    dtype=self.np.float64,
                )
            self._force_safety_detector_score = float(detector_step.score)
        elif source in DEPLOYABLE_V2_FORCE_SAFETY_TRIGGER_SOURCES:
            detector_step, linear_baseline, angular_baseline = _force_safety_detector_v2_step_value(
                np_module=self.np,
                base_linear_acceleration=linear_acceleration,
                base_angular_acceleration=angular_acceleration,
                linear_acceleration_baseline=self._force_safety_detector_v2_linear_baseline,
                angular_acceleration_baseline=self._force_safety_detector_v2_angular_baseline,
                previous_active=bool(self._force_safety_detector_gate > 0.0),
                warmup=bool(
                    self._force_safety_detector_v2_sample_count
                    < FORCE_SAFETY_DETECTOR_V2_WARMUP_STEPS
                ),
                linear_acceleration_threshold=self.config.force_safety_detector_linear_acceleration_threshold,
                angular_acceleration_threshold=self.config.force_safety_detector_angular_acceleration_threshold,
            )
            self._force_safety_detector_v2_linear_baseline[:] = self.np.asarray(
                linear_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_angular_baseline[:] = self.np.asarray(
                angular_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_sample_count += 1
            previous_gate = self._force_safety_detector_gate
            if detector_step.active and previous_gate <= 0.0:
                self._force_safety_detector_start_s = float(self._elapsed_s)
            if detector_step.active:
                self._force_safety_detector_force_proxy[:] = self.np.asarray(
                    detector_step.force_proxy,
                    dtype=self.np.float64,
                )
            self._force_safety_detector_score = float(detector_step.score)
        elif source in DEPLOYABLE_HISTORY_FORCE_SAFETY_TRIGGER_SOURCES:
            previous_gate = self._force_safety_detector_gate
            detector_step = self._force_safety_history_detector_step(
                linear_acceleration=linear_acceleration,
                previous_active=bool(previous_gate > 0.0),
            )
            if detector_step.active and previous_gate <= 0.0:
                self._force_safety_detector_start_s = float(self._elapsed_s)
            if detector_step.active:
                self._force_safety_detector_force_proxy[:] = self.np.asarray(
                    detector_step.force_proxy,
                    dtype=self.np.float64,
                )
            self._force_safety_detector_score = float(detector_step.score)
        elif source in DEPLOYABLE_HYBRID_FORCE_SAFETY_TRIGGER_SOURCES:
            previous_gate = self._force_safety_detector_gate
            acceleration_step, linear_baseline, angular_baseline = _force_safety_detector_v2_step_value(
                np_module=self.np,
                base_linear_acceleration=linear_acceleration,
                base_angular_acceleration=angular_acceleration,
                linear_acceleration_baseline=self._force_safety_detector_v2_linear_baseline,
                angular_acceleration_baseline=self._force_safety_detector_v2_angular_baseline,
                previous_active=bool(previous_gate > 0.0),
                warmup=bool(
                    self._force_safety_detector_v2_sample_count
                    < FORCE_SAFETY_DETECTOR_V2_WARMUP_STEPS
                ),
                linear_acceleration_threshold=self.config.force_safety_detector_linear_acceleration_threshold,
                angular_acceleration_threshold=self.config.force_safety_detector_angular_acceleration_threshold,
            )
            self._force_safety_detector_v2_linear_baseline[:] = self.np.asarray(
                linear_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_angular_baseline[:] = self.np.asarray(
                angular_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_sample_count += 1
            history_step = self._force_safety_history_detector_step(
                linear_acceleration=linear_acceleration,
                previous_active=bool(previous_gate > 0.0),
            )
            estimator = self._force_safety_history_estimator
            history_threshold = (
                1.0
                if estimator is None
                else (
                    estimator.release_threshold
                    if bool(previous_gate > 0.0)
                    else estimator.threshold
                )
            )
            detector_step = _hybrid_force_safety_detector_step_value(
                history_step=history_step,
                acceleration_step=acceleration_step,
                history_threshold=history_threshold,
            )
            if detector_step.active and previous_gate <= 0.0:
                self._force_safety_detector_start_s = float(self._elapsed_s)
            if detector_step.active:
                self._force_safety_detector_force_proxy[:] = self.np.asarray(
                    detector_step.force_proxy,
                    dtype=self.np.float64,
                )
            self._force_safety_detector_score = float(detector_step.score)
        elif source in DEPLOYABLE_HISTORY_OR_V2_FORCE_SAFETY_TRIGGER_SOURCES:
            previous_gate = self._force_safety_detector_gate
            acceleration_step, linear_baseline, angular_baseline = _force_safety_detector_v2_step_value(
                np_module=self.np,
                base_linear_acceleration=linear_acceleration,
                base_angular_acceleration=angular_acceleration,
                linear_acceleration_baseline=self._force_safety_detector_v2_linear_baseline,
                angular_acceleration_baseline=self._force_safety_detector_v2_angular_baseline,
                previous_active=bool(previous_gate > 0.0),
                warmup=bool(
                    self._force_safety_detector_v2_sample_count
                    < FORCE_SAFETY_DETECTOR_V2_WARMUP_STEPS
                ),
                linear_acceleration_threshold=self.config.force_safety_detector_linear_acceleration_threshold,
                angular_acceleration_threshold=self.config.force_safety_detector_angular_acceleration_threshold,
            )
            self._force_safety_detector_v2_linear_baseline[:] = self.np.asarray(
                linear_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_angular_baseline[:] = self.np.asarray(
                angular_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_sample_count += 1
            history_step = self._force_safety_history_detector_step(
                linear_acceleration=linear_acceleration,
                previous_active=bool(previous_gate > 0.0),
            )
            estimator = self._force_safety_history_estimator
            history_threshold = (
                1.0
                if estimator is None
                else (
                    estimator.release_threshold
                    if bool(previous_gate > 0.0)
                    else estimator.threshold
                )
            )
            detector_step = _history_or_v2_force_safety_detector_step_value(
                history_step=history_step,
                acceleration_step=acceleration_step,
                history_threshold=history_threshold,
            )
            if detector_step.active and previous_gate <= 0.0:
                self._force_safety_detector_start_s = float(self._elapsed_s)
            if detector_step.active:
                self._force_safety_detector_force_proxy[:] = self.np.asarray(
                    detector_step.force_proxy,
                    dtype=self.np.float64,
                )
            self._force_safety_detector_score = float(detector_step.score)
        elif source in DEPLOYABLE_HISTORY_GATED_V2_FORCE_SAFETY_TRIGGER_SOURCES:
            previous_gate = self._force_safety_detector_gate
            acceleration_step, linear_baseline, angular_baseline = _force_safety_detector_v2_step_value(
                np_module=self.np,
                base_linear_acceleration=linear_acceleration,
                base_angular_acceleration=angular_acceleration,
                linear_acceleration_baseline=self._force_safety_detector_v2_linear_baseline,
                angular_acceleration_baseline=self._force_safety_detector_v2_angular_baseline,
                previous_active=bool(previous_gate > 0.0),
                warmup=bool(
                    self._force_safety_detector_v2_sample_count
                    < FORCE_SAFETY_DETECTOR_V2_WARMUP_STEPS
                ),
                linear_acceleration_threshold=self.config.force_safety_detector_linear_acceleration_threshold,
                angular_acceleration_threshold=self.config.force_safety_detector_angular_acceleration_threshold,
            )
            self._force_safety_detector_v2_linear_baseline[:] = self.np.asarray(
                linear_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_angular_baseline[:] = self.np.asarray(
                angular_baseline,
                dtype=self.np.float64,
            )
            self._force_safety_detector_v2_sample_count += 1
            history_step = self._force_safety_history_detector_step(
                linear_acceleration=linear_acceleration,
                previous_active=bool(previous_gate > 0.0),
            )
            estimator = self._force_safety_history_estimator
            history_threshold = (
                1.0
                if estimator is None
                else (
                    estimator.release_threshold
                    if bool(previous_gate > 0.0)
                    else estimator.threshold
                )
            )
            detector_step = _history_gated_v2_force_safety_detector_step_value(
                history_step=history_step,
                acceleration_step=acceleration_step,
                history_threshold=history_threshold,
            )
            if detector_step.active and previous_gate <= 0.0:
                self._force_safety_detector_start_s = float(self._elapsed_s)
            if detector_step.active:
                self._force_safety_detector_force_proxy[:] = self.np.asarray(
                    detector_step.force_proxy,
                    dtype=self.np.float64,
                )
            self._force_safety_detector_score = float(detector_step.score)
        else:
            self._force_safety_detector_score = 0.0
            if self._force_safety_detector_gate <= 0.0:
                self._force_safety_detector_force_proxy[:] = 0.0

        self._force_safety_detector_gate = _force_safety_detector_gate_value(
            t_s=float(self._elapsed_s),
            start_s=self._force_safety_detector_start_s,
            hold_s=self.config.force_safety_detector_hold_s,
            recovery_s=self.config.force_safety_detector_recovery_s,
        )
        self._force_safety_detector_active = bool(
            self._force_safety_detector_gate > 0.0 or self._force_safety_detector_score >= 1.0
        )
        if self._force_safety_detector_gate <= 0.0 and self._force_safety_detector_score < 1.0:
            self._force_safety_detector_start_s = math.inf
            self._force_safety_detector_force_proxy[:] = 0.0

        self._previous_base_linear_velocity = base_linear_velocity.copy()
        self._previous_base_angular_velocity = base_angular_velocity.copy()
        self._previous_contact_count = contact_count

    def _force_safety_windows(self) -> list[tuple[float, float]]:
        return _force_safety_windows_value(
            source=self.config.force_safety_trigger_source,
            oracle_windows=self._external_force_windows(),
            detector_start_s=self._force_safety_detector_start_s,
            detector_hold_s=self.config.force_safety_detector_hold_s,
        )

    def _force_safety_force_vector(self) -> Any:
        source = str(self.config.force_safety_trigger_source)
        oracle_norm = float(self.np.linalg.norm(self._external_force_vector))
        if source in ORACLE_FORCE_SAFETY_TRIGGER_SOURCES and oracle_norm > 1e-9:
            return self._external_force_vector
        if source in ANY_DEPLOYABLE_FORCE_SAFETY_TRIGGER_SOURCES and self._force_safety_detector_gate > 0.0:
            return self._force_safety_detector_force_proxy
        return self.np.zeros(3, dtype=self.np.float64)

    def _external_force_windows(self) -> list[tuple[float, float]]:
        windows = [(pulse.start_s, pulse.end_s) for pulse in self._external_force_pulses]
        windows.extend((spring.start_s, spring.end_s) for spring in self._external_force_springs)
        if (
            not windows
            and self._external_force_applied_this_step
            and math.isfinite(float(self._external_force_start_s))
            and math.isfinite(float(self._external_force_end_s))
        ):
            windows.append((float(self._external_force_start_s), float(self._external_force_end_s)))
        return windows

    def _current_force_safety_window_elapsed_s(self) -> float | None:
        t_s = float(self._elapsed_s)
        for start_s, end_s in self._force_safety_windows():
            start_s = float(start_s)
            end_s = float(end_s)
            if start_s <= t_s <= end_s:
                return t_s - start_s
        return None

    def _current_external_force_window(self) -> tuple[float, float] | None:
        t_s = float(self._elapsed_s)
        for start_s, end_s in self._external_force_windows():
            start_s = float(start_s)
            end_s = float(end_s)
            if start_s <= t_s <= end_s:
                return start_s, end_s
        return None

    def _current_external_force_window_elapsed_s(self) -> float | None:
        window = self._current_external_force_window()
        if window is None:
            return None
        start_s, _ = window
        return max(0.0, float(self._elapsed_s) - start_s)

    def _maybe_apply_push(self) -> None:
        if self.config.push_time_s is None or self._push_applied:
            return
        if self._elapsed_s + 1e-12 < self.config.push_time_s:
            return

        for index, delta in enumerate(self.config.push_linear_velocity):
            if index < len(self.data.qvel):
                self.data.qvel[index] += float(delta)
        for index, delta in enumerate(self.config.push_angular_velocity, start=3):
            if index < len(self.data.qvel):
                self.data.qvel[index] += float(delta)
        self._push_applied = True

    def _sample_external_force(self) -> None:
        self._external_force_body_id = -1
        self._external_force_body_name = ""
        self._external_force_vector[:] = 0.0
        self._external_force_torque_vector[:] = 0.0
        self._external_force_start_s = math.inf
        self._external_force_end_s = -math.inf
        self._external_force_pulses = []
        self._external_force_springs = []

        probability = max(0.0, min(1.0, float(self.config.external_force_probability)))
        if probability <= 0.0 or not self.external_force_body_ids:
            return
        if float(self._rng.random()) >= probability:
            return

        current_max_n = self._current_external_force_max_n()
        if current_max_n <= 0.0:
            return

        windows = self._sample_external_force_windows()
        if not windows:
            return

        if self.config.external_force_mode == "spring":
            for start_s, end_s in windows:
                self._sample_external_springs(start_s=start_s, end_s=end_s, current_max_n=current_max_n)
            self._refresh_external_force_summary_from_schedule()
            return

        for start_s, end_s in windows:
            self._sample_external_pulse(start_s=start_s, end_s=end_s, current_max_n=current_max_n)
        self._refresh_external_force_summary_from_schedule()

    def _sample_external_force_windows(self) -> list[tuple[float, float]]:
        count_min, count_max = _sorted_range(self.config.external_force_event_count_range)
        count_min = max(1, int(round(count_min)))
        count_max = max(count_min, int(round(count_max)))
        event_count = (
            int(self._rng.integers(count_min, count_max + 1))
            if count_max > count_min
            else count_min
        )

        first_window = self._sample_external_force_window()
        if first_window is None:
            return []

        windows = [first_window]
        if event_count <= 1:
            return windows

        rest_min, rest_max = _sorted_range(self.config.external_force_rest_s_range)
        rest_min = max(0.0, rest_min)
        rest_max = max(rest_min, rest_max)

        while len(windows) < event_count:
            previous_end = windows[-1][1]
            rest_s = float(self._rng.uniform(rest_min, rest_max)) if rest_max > rest_min else rest_min
            start_s = previous_end + rest_s
            if start_s >= self.config.episode_length_s:
                break

            duration_s = self._sample_external_force_duration()
            end_s = min(self.config.episode_length_s, start_s + duration_s)
            if end_s <= start_s:
                break
            windows.append((start_s, end_s))
        return windows

    def _sample_external_force_window(self) -> tuple[float, float] | None:
        start_min, start_max = _sorted_range(self.config.external_force_start_s_range)
        duration_min, duration_max = _sorted_range(self.config.external_force_duration_s_range)
        duration_min = max(0.0, duration_min)
        duration_max = max(0.0, duration_max)
        if duration_max <= 0.0:
            return None

        latest_start = max(start_min, min(start_max, self.config.episode_length_s - duration_min))
        start_s = float(self._rng.uniform(start_min, latest_start)) if latest_start > start_min else start_min
        duration_s = self._sample_external_force_duration()
        end_s = min(self.config.episode_length_s, start_s + duration_s)
        if end_s <= start_s:
            return None
        return start_s, end_s

    def _sample_external_force_duration(self) -> float:
        duration_min, duration_max = _sorted_range(self.config.external_force_duration_s_range)
        duration_min = max(0.0, duration_min)
        duration_max = max(duration_min, duration_max)
        return float(self._rng.uniform(duration_min, duration_max)) if duration_max > duration_min else duration_min

    def _sample_external_pulse(self, *, start_s: float, end_s: float, current_max_n: float) -> None:
        min_n = max(0.0, min(float(self.config.external_force_min_n), current_max_n))
        magnitude = float(self._rng.uniform(min_n, current_max_n)) if current_max_n > min_n else current_max_n
        if magnitude <= 0.0:
            return

        body_id, body_name = self.external_force_body_ids[
            int(self._rng.integers(0, len(self.external_force_body_ids)))
        ]
        self._external_force_pulses.append(
            ExternalForcePulse(
                body_id=body_id,
                body_name=body_name,
                start_s=start_s,
                end_s=end_s,
                force_vector=magnitude * self._sample_external_force_direction(),
                torque_vector=self._sample_external_torque_vector(),
            )
        )

    def _sample_external_springs(self, *, start_s: float, end_s: float, current_max_n: float) -> None:
        active_count = max(1, int(self.config.external_force_active_body_count))
        active_count = min(active_count, len(self.external_force_body_ids))
        selected_indices = self._rng.choice(
            len(self.external_force_body_ids),
            size=active_count,
            replace=False,
        )
        shared_direction_root = self._sample_external_force_direction()
        min_n = max(0.0, min(float(self.config.external_force_min_n), current_max_n))
        stiffness_min, stiffness_max = _sorted_range(self.config.external_force_spring_stiffness_range)
        stiffness_min = max(1e-6, stiffness_min)
        stiffness_max = max(stiffness_min, stiffness_max)
        guiding_probability = max(0.0, min(1.0, float(self.config.external_force_guiding_probability)))

        for index in selected_indices:
            body_id, body_name = self.external_force_body_ids[int(index)]
            cap_n = float(self._rng.uniform(min_n, current_max_n)) if current_max_n > min_n else current_max_n
            if cap_n <= 0.0:
                continue
            stiffness = (
                float(self._rng.uniform(stiffness_min, stiffness_max))
                if stiffness_max > stiffness_min
                else stiffness_min
            )
            spring_type = "guiding" if float(self._rng.random()) < guiding_probability else "resistive"
            compression = cap_n / max(stiffness, 1e-6)
            torque_root = self._sample_external_torque_vector()

            self._external_force_springs.append(
                ExternalForceSpring(
                    body_id=body_id,
                    body_name=body_name,
                    spring_type=spring_type,
                    start_s=start_s,
                    end_s=end_s,
                    direction_root=shared_direction_root.copy(),
                    anchor_root=self.np.zeros(3, dtype=self.np.float64),
                    anchor_velocity_root=self.np.zeros(3, dtype=self.np.float64),
                    anchor_target_root=compression * shared_direction_root.copy(),
                    torque_root=torque_root,
                    stiffness=stiffness,
                    damping=max(0.0, float(self.config.external_force_spring_damping)),
                    cap_n=cap_n,
                )
            )

        if not self._external_force_springs:
            return

        first = self._external_force_springs[0]
        self._external_force_body_id = first.body_id
        self._external_force_body_name = ",".join(spring.body_name for spring in self._external_force_springs)
        self._external_force_start_s = start_s
        self._external_force_end_s = end_s

    def _refresh_external_force_summary_from_schedule(self) -> None:
        starts: list[float] = []
        ends: list[float] = []
        body_names: list[str] = []
        body_id = -1

        for pulse in self._external_force_pulses:
            starts.append(pulse.start_s)
            ends.append(pulse.end_s)
            body_names.append(pulse.body_name)
            if body_id < 0:
                body_id = pulse.body_id
        for spring in self._external_force_springs:
            starts.append(spring.start_s)
            ends.append(spring.end_s)
            body_names.append(spring.body_name)
            if body_id < 0:
                body_id = spring.body_id

        if not starts:
            return

        self._external_force_body_id = body_id
        self._external_force_body_name = ",".join(dict.fromkeys(body_names))
        self._external_force_start_s = min(starts)
        self._external_force_end_s = max(ends)

    def _current_external_force_max_n(self) -> float:
        final_max = max(0.0, float(self.config.external_force_max_n))
        start_max = self.config.external_force_curriculum_start_n
        if start_max is None:
            return final_max
        start_max = max(0.0, float(start_max))
        progress = self._external_force_curriculum_progress
        return max(0.0, start_max + progress * (final_max - start_max))

    def _current_external_torque_max_nm(self) -> float:
        final_max = max(0.0, float(self.config.external_force_torque_max_nm))
        if final_max <= 0.0:
            return 0.0
        if self.config.external_force_curriculum_start_n is None:
            return final_max
        return final_max * self._external_force_curriculum_progress

    def _sample_external_force_direction(self) -> Any:
        return _sample_external_force_direction_value(
            np_module=self.np,
            rng=self._rng,
            fixed_angle_rad=self.config.external_force_direction_angle_rad,
            z_fraction=self.config.external_force_z_fraction,
            direction_mode=self.config.external_force_direction_mode,
            lateral_probability=self.config.external_force_lateral_probability,
        )

    def _sample_external_torque_vector(self) -> Any:
        max_nm = self._current_external_torque_max_nm()
        if max_nm <= 0.0:
            return self.np.zeros(3, dtype=self.np.float64)
        magnitude = float(self._rng.uniform(0.0, max_nm))
        direction = self._rng.normal(size=3)
        norm = float(self.np.linalg.norm(direction))
        if norm <= 1e-9:
            return self.np.zeros(3, dtype=self.np.float64)
        return magnitude * direction / norm

    def _clear_external_forces(self) -> None:
        self.data.xfrc_applied[:] = 0.0
        self._external_force_applied_this_step = False
        self._external_force_applied_by_body_root = {}

    def _maybe_apply_external_force(self) -> None:
        if self.config.external_force_mode == "spring":
            self._maybe_apply_external_spring_forces()
            return

        total_force = self.np.zeros(3, dtype=self.np.float64)
        total_torque = self.np.zeros(3, dtype=self.np.float64)
        active_body_names: list[str] = []
        for pulse in self._external_force_pulses:
            if not self._external_force_pulse_is_currently_active(pulse):
                continue
            scale = self._external_force_window_scale(pulse.start_s, pulse.end_s)
            force = pulse.force_vector * scale
            torque = pulse.torque_vector * scale
            self.data.xfrc_applied[pulse.body_id, :3] += force
            self.data.xfrc_applied[pulse.body_id, 3:] += torque
            force_root = self._root_rotation_matrix().T @ force
            previous = self._external_force_applied_by_body_root.get(pulse.body_name)
            self._external_force_applied_by_body_root[pulse.body_name] = (
                force_root if previous is None else previous + force_root
            )
            total_force += force
            total_torque += torque
            active_body_names.append(pulse.body_name)

        force_norm = float(self.np.linalg.norm(total_force))
        torque_norm = float(self.np.linalg.norm(total_torque))
        if force_norm <= 1e-9 and torque_norm <= 1e-9:
            return
        self._external_force_vector[:] = total_force
        self._external_force_torque_vector[:] = total_torque
        self._external_force_body_name = ",".join(dict.fromkeys(active_body_names))
        self._external_force_applied_this_step = True

    def _maybe_apply_external_spring_forces(self) -> None:
        active_forces = []
        active_torques = []
        for spring in self._external_force_springs:
            if not self._external_force_spring_is_currently_active(spring):
                continue
            force_root = self._compute_external_spring_force_root(spring)
            force_world = spring.frame_rotation_world @ force_root
            torque_world = spring.frame_rotation_world @ (
                spring.torque_root * self._external_force_window_scale(spring.start_s, spring.end_s)
            )
            active_forces.append((spring.body_id, force_world))
            active_torques.append((spring.body_id, torque_world))

        if not active_forces and not active_torques:
            return

        active_forces = self._limit_external_spring_net_wrench(active_forces)
        total_force = self.np.zeros(3, dtype=self.np.float64)
        total_torque = self.np.zeros(3, dtype=self.np.float64)
        active_body_names: list[str] = []
        for body_id, force_world in active_forces:
            self.data.xfrc_applied[body_id, :3] += force_world
            total_force += force_world
            body_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                body_name = str(body_name)
                active_body_names.append(body_name)
                force_root = self._root_rotation_matrix().T @ force_world
                previous = self._external_force_applied_by_body_root.get(body_name)
                self._external_force_applied_by_body_root[body_name] = (
                    force_root if previous is None else previous + force_root
                )
        for body_id, torque_world in active_torques:
            self.data.xfrc_applied[body_id, 3:] += torque_world
            total_torque += torque_world

        force_norm = float(self.np.linalg.norm(total_force))
        torque_norm = float(self.np.linalg.norm(total_torque))
        if force_norm <= 1e-9 and torque_norm <= 1e-9:
            return
        self._external_force_vector[:] = total_force
        self._external_force_torque_vector[:] = total_torque
        if active_body_names:
            self._external_force_body_name = ",".join(dict.fromkeys(active_body_names))
        self._external_force_applied_this_step = True

    def _compute_external_spring_force_root(self, spring: ExternalForceSpring) -> Any:
        if not spring.initialized:
            self._initialize_external_spring(spring)
        if spring.spring_type == "guiding":
            self._integrate_external_spring_anchor(spring)

        force_root = _compute_external_spring_force_root_value(
            np=self.np,
            spring_type=spring.spring_type,
            anchor_root=spring.anchor_root,
            anchor_velocity_root=spring.anchor_velocity_root,
            body_position_root=self._body_position_external_spring_frame(spring),
            body_velocity_root=self._body_velocity_external_spring_frame(spring),
            direction_root=spring.direction_root,
            stiffness=spring.stiffness,
            damping=spring.damping,
            cap_n=spring.cap_n,
        )
        return force_root * self._external_force_window_scale(spring.start_s, spring.end_s)

    def _initialize_external_spring(self, spring: ExternalForceSpring) -> None:
        spring.frame_origin_world = self.np.asarray(self.data.qpos[:3], dtype=self.np.float64).copy()
        spring.frame_rotation_world = self._root_rotation_matrix().copy()
        body_position = self._body_position_external_spring_frame(spring)
        compression = spring.cap_n / max(spring.stiffness, 1e-6)
        if spring.spring_type == "guiding":
            spring.anchor_root = body_position.copy()
            spring.anchor_target_root = body_position + spring.direction_root * compression
        else:
            spring.anchor_root = body_position + spring.direction_root * compression
            spring.anchor_target_root = spring.anchor_root.copy()
        spring.anchor_velocity_root = self.np.zeros(3, dtype=self.np.float64)
        spring.initialized = True

    def _integrate_external_spring_anchor(self, spring: ExternalForceSpring) -> None:
        mass = max(1e-6, float(self.config.external_force_reference_mass))
        drive_stiffness = max(1e-6, spring.stiffness)
        drive_damping = 2.0 * math.sqrt(mass * drive_stiffness)
        damping = max(0.0, float(self.config.external_force_reference_damping))
        drive_force = drive_stiffness * (spring.anchor_target_root - spring.anchor_root)
        damping_force = -(drive_damping + damping) * spring.anchor_velocity_root
        acceleration = (drive_force + damping_force) / mass
        acceleration = _clip_vector_norm(
            acceleration,
            max(0.0, float(self.config.external_force_reference_acceleration_clip)),
            self.np,
        )
        dt = max(1e-6, float(self.config.control_dt))
        spring.anchor_velocity_root = spring.anchor_velocity_root + acceleration * dt
        spring.anchor_velocity_root = _clip_vector_norm(
            spring.anchor_velocity_root,
            max(0.0, float(self.config.external_force_reference_velocity_clip)),
            self.np,
        )
        spring.anchor_root = spring.anchor_root + spring.anchor_velocity_root * dt

    def _limit_external_spring_net_wrench(self, forces: list[tuple[int, Any]]) -> list[tuple[int, Any]]:
        scale = 1.0
        net_force_limit = max(0.0, float(self.config.external_force_net_force_limit_n))
        if net_force_limit > 0.0:
            total_force = self.np.sum([force for _, force in forces], axis=0)
            total_force_norm = float(self.np.linalg.norm(total_force))
            if total_force_norm > net_force_limit:
                scale = min(scale, net_force_limit / total_force_norm)

        net_torque_limit = max(0.0, float(self.config.external_force_net_torque_limit_nm))
        if net_torque_limit > 0.0:
            root_position = self.np.asarray(self.data.qpos[:3], dtype=self.np.float64)
            net_torque = self.np.zeros(3, dtype=self.np.float64)
            for body_id, force in forces:
                arm = self.np.asarray(self.data.xpos[body_id], dtype=self.np.float64) - root_position
                net_torque += self.np.cross(arm, force)
            net_torque_norm = float(self.np.linalg.norm(net_torque))
            if net_torque_norm > net_torque_limit:
                scale = min(scale, net_torque_limit / net_torque_norm)

        if scale >= 1.0:
            return forces
        return [(body_id, force * scale) for body_id, force in forces]

    def _external_force_is_currently_active(self) -> bool:
        if self._external_force_body_id < 0:
            return False
        return self._external_force_start_s <= self._elapsed_s <= self._external_force_end_s

    def _external_force_spring_is_currently_active(self, spring: ExternalForceSpring) -> bool:
        return spring.start_s <= self._elapsed_s <= spring.end_s

    def _external_force_pulse_is_currently_active(self, pulse: ExternalForcePulse) -> bool:
        return pulse.start_s <= self._elapsed_s <= pulse.end_s

    def _recent_external_force_end_s(self) -> float | None:
        ends = [
            pulse.end_s
            for pulse in self._external_force_pulses
            if pulse.end_s <= self._elapsed_s
        ]
        ends.extend(
            spring.end_s
            for spring in self._external_force_springs
            if spring.end_s <= self._elapsed_s
        )
        if not ends:
            return None
        return max(ends)

    def _external_force_window_scale(self, start_s: float, end_s: float) -> float:
        return _external_force_window_scale_value(
            elapsed_s=self._elapsed_s,
            start_s=start_s,
            end_s=end_s,
            transition_s=self.config.external_force_transition_s,
        )

    def _current_external_force_reward_elapsed_s(self) -> float:
        current_window_elapsed = self._current_external_force_window_elapsed_s()
        if current_window_elapsed is not None:
            return float(current_window_elapsed)
        return float(self._elapsed_s - self._external_force_start_s)

    def _reward(
        self,
        previous_action: Any,
        previous_joint_vel: Any,
        previous_ctrl: Any | None = None,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        projected_gravity = self._projected_gravity()
        base_lin_vel = self.np.asarray(self.data.qvel[:3], dtype=self.np.float64)
        base_ang_vel = self.np.asarray(self.data.qvel[3:6], dtype=self.np.float64)
        base_z = float(self.data.qpos[2]) if len(self.data.qpos) >= 3 else 0.0
        height_error = base_z - self.config.target_base_z
        upright = max(0.0, min(1.0, -float(projected_gravity[2])))
        joint_pos_penalty = 0.0
        joint_vel_penalty = 0.0
        for joint_name in self.joint_names:
            qpos_adr, qvel_adr = self.joint_map[joint_name]
            joint_pos_penalty += (float(self.data.qpos[qpos_adr]) - self.default_targets[joint_name]) ** 2
            joint_vel_penalty += float(self.data.qvel[qvel_adr]) ** 2

        action_rate = self.np.asarray(self._last_action - previous_action, dtype=self.np.float64)
        height_reward = float(self.np.exp(-40.0 * height_error * height_error))
        torque_penalty = float(self.np.dot(self.data.ctrl, self.data.ctrl))

        if self.config.task in VELOCITY_TASKS:
            terms = self._mjlab_velocity_reward(
                projected_gravity=projected_gravity,
                base_lin_vel=base_lin_vel,
                base_ang_vel=base_ang_vel,
                base_z=base_z,
                action_rate=action_rate,
                previous_joint_vel=previous_joint_vel,
                previous_ctrl=previous_ctrl,
            )
        else:
            terms = {
                "alive": 0.25,
                "upright": 1.25 * upright,
                "height": 0.75 * height_reward,
                "lin_vel_penalty": -0.20 * float(self.np.dot(base_lin_vel, base_lin_vel)),
                "ang_vel_penalty": -0.08 * float(self.np.dot(base_ang_vel, base_ang_vel)),
                "joint_pos_penalty": -0.05 * joint_pos_penalty,
                "joint_vel_penalty": -0.001 * joint_vel_penalty,
                "action_rate_penalty": -0.02 * float(self.np.dot(action_rate, action_rate)),
                "torque_penalty": -0.0002 * torque_penalty,
            }
        if self._terminated():
            terms["termination_penalty"] = -2.0
        return float(sum(terms.values())), terms, self._group_reward_terms(terms)

    def _group_reward_terms(self, terms: dict[str, float]) -> dict[str, float]:
        if self.config.task in VELOCITY_TASKS:
            legacy_terms = {
                "alive": terms.get("alive", 0.0),
                "height": terms.get("height", 0.0),
                "velocity_tracking": terms.get("target_forward_velocity_band", 0.0)
                + terms.get("track_linear_velocity", 0.0)
                + terms.get("track_angular_velocity", 0.0)
                + terms.get("forward_progress", 0.0),
                "velocity_shortfall_penalty": terms.get("velocity_shortfall_penalty", 0.0),
                "velocity_overspeed_penalty": terms.get("velocity_overspeed_penalty", 0.0),
                "drift_penalty": terms.get("lateral_velocity_l2", 0.0)
                + terms.get("lateral_displacement_l2", 0.0)
                + terms.get("lateral_return_velocity_l2", 0.0)
                + terms.get("yaw_rate_l2", 0.0)
                + terms.get("heading_l2", 0.0),
                "feet_air_time": terms.get("feet_air_time", 0.0),
                "contact_count_penalty": terms.get("contact_count_penalty", 0.0)
                + terms.get("force_active_contact_count_penalty", 0.0)
                + terms.get("force_onset_contact_count_penalty", 0.0),
                "stance_stagnation_penalty": terms.get("stance_stagnation_penalty", 0.0),
                "posture": terms.get("pose", 0.0),
                "orientation": terms.get("flat_orientation_l2", 0.0)
                + terms.get("body_ang_vel", 0.0),
                "joint_acc_penalty": terms.get("joint_acc_l2", 0.0),
                "joint_limit_penalty": terms.get("joint_pos_limits", 0.0),
                "action_rate_penalty": terms.get("action_rate_l2", 0.0),
                "residual_penalty": terms.get("residual_action_l2", 0.0)
                + terms.get("residual_action_rate_l2", 0.0),
                "foot_slip_penalty": terms.get("foot_slip", 0.0),
                "compliant_tracking": terms.get("keypoint_tracking_comp", 0.0)
                + terms.get("reference_dynamics_tracking", 0.0)
                + terms.get("reference_force_tracking", 0.0)
                + terms.get("root_tracking", 0.0)
                + terms.get("root_yaw_tracking", 0.0)
                + terms.get("expert_action_tracking", 0.0)
                + terms.get("joint_pos_tracking", 0.0)
                + terms.get("joint_vel_tracking", 0.0),
                "interaction_stability": terms.get("impact_force_l2", 0.0)
                + terms.get("torque_smoothness_l2", 0.0)
                + terms.get("unsafe_force_penalty", 0.0)
                + terms.get("unsafe_force_peak_penalty", 0.0)
                + terms.get("unsafe_force_onset_penalty", 0.0)
                + terms.get("external_force_excess_penalty", 0.0)
                + terms.get("external_force_excess_onset_penalty", 0.0)
                + terms.get("external_force_violation_penalty", 0.0)
                + terms.get("external_force_violation_onset_penalty", 0.0)
                + terms.get("external_force_peak_delta_penalty", 0.0)
                + terms.get("external_force_peak_delta_onset_penalty", 0.0),
                "anti_collapse": terms.get("anti_collapse_height", 0.0)
                + terms.get("anti_collapse_downward_velocity", 0.0)
                + terms.get("anti_collapse_orientation", 0.0),
                "recovery": terms.get("recovery_track_linear_velocity", 0.0)
                + terms.get("recovery_forward_progress", 0.0)
                + terms.get("recovery_velocity_shortfall_penalty", 0.0)
                + terms.get("recovery_lateral_velocity_l2", 0.0)
                + terms.get("recovery_yaw_rate_l2", 0.0)
                + terms.get("recovery_stance_stagnation_penalty", 0.0),
                "energy_penalty": 0.0,
                "fall_penalty": terms.get("termination_penalty", 0.0),
            }
            return {**legacy_terms, **_layered_velocity_reward_terms(terms)}

        return {
            "alive": terms.get("alive", 0.0),
            "height": terms.get("height", 0.0),
            "orientation": terms.get("upright", 0.0),
            "velocity_penalty": terms.get("lin_vel_penalty", 0.0)
            + terms.get("ang_vel_penalty", 0.0),
            "joint_position_penalty": terms.get("joint_pos_penalty", 0.0),
            "joint_velocity_penalty": terms.get("joint_vel_penalty", 0.0),
            "energy_penalty": terms.get("torque_penalty", 0.0),
            "action_rate_penalty": terms.get("action_rate_penalty", 0.0),
            "foot_slip_penalty": 0.0,
            "fall_penalty": terms.get("termination_penalty", 0.0),
        }

    def _mjlab_velocity_reward(
        self,
        *,
        projected_gravity: Any,
        base_lin_vel: Any,
        base_ang_vel: Any,
        base_z: float,
        action_rate: Any,
        previous_joint_vel: Any,
        previous_ctrl: Any | None = None,
    ) -> dict[str, float]:
        reward_config = self.config.velocity_reward
        command = self._command()
        linear_std = reward_config.linear_std
        yaw_std = reward_config.yaw_std
        tracking_lin_vel = self._velocity_tracking_linear_velocity(base_lin_vel)
        force_active_scale = self._external_force_active_scale()
        compliant_command = self._compliant_velocity_command(command, force_active_scale)
        xy_error = float(self.np.sum(self.np.square(compliant_command[:2] - tracking_lin_vel[:2])))
        z_error = float(tracking_lin_vel[2] ** 2)
        lateral_error = float((compliant_command[1] - tracking_lin_vel[1]) ** 2)
        target_root_position, _ = self._fullbody_reference_root_target()
        lateral_displacement = _lateral_displacement_error(
            self.data.qpos[:2],
            target_root_position[:2],
            command[:2],
        )
        lateral_displacement_error = lateral_displacement * lateral_displacement
        lateral_return_velocity_error = _lateral_return_velocity_error(
            tracking_lin_vel[:2],
            command[:2],
            lateral_displacement,
            gain=reward_config.lateral_return_velocity_gain,
            clip=reward_config.lateral_return_velocity_clip,
        )
        yaw_error = float((command[2] - base_ang_vel[2]) ** 2)
        roll_pitch_rate_error = float(self.np.sum(self.np.square(base_ang_vel[:2])))
        flat_orientation_l2 = float(self.np.sum(self.np.square(projected_gravity[:2])))
        command_speed = float(self.np.linalg.norm(command[:2]))
        actual_along_command = 0.0
        if command_speed > 1e-6:
            actual_along_command = float(self.np.dot(tracking_lin_vel[:2], command[:2]) / command_speed)
        if command_speed > 0.05 and reward_config.target_forward_velocity_band != 0.0:
            velocity_shortfall, velocity_overspeed = _target_band_velocity_errors(
                actual_along_command,
                lower=reward_config.target_forward_velocity_band_min,
                upper=reward_config.target_forward_velocity_band_max,
            )
        else:
            velocity_shortfall = max(0.0, command_speed - actual_along_command) if command_speed > 0.05 else 0.0
            velocity_overspeed = max(0.0, actual_along_command - command_speed) if command_speed > 0.05 else 0.0
        forward_progress = 0.0
        if command_speed > 0.05:
            forward_progress = max(0.0, min(actual_along_command / command_speed, 1.0))
        joint_vel = self._joint_vel_array()
        joint_acc = (joint_vel - self.np.asarray(previous_joint_vel, dtype=self.np.float64)) / max(
            self.config.control_dt,
            1e-6,
        )
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        moving_command = command_speed + abs(float(command[2])) > 0.05
        angular_tracking_weight = (
            reward_config.track_angular_velocity_moving
            if command_speed > 0.05 and abs(float(command[2])) < 0.05
            else reward_config.track_angular_velocity
        )
        contact_count = sum(1 for foot in FEET if contacts.get(foot, False))
        heading_error = self._heading_error(command_speed, float(command[2]))
        force_elapsed_s = self._current_external_force_reward_elapsed_s()

        recovery_scale = self._external_force_recovery_scale()
        force_velocity_scale = _blend_force_recovery_reward_scale(
            reward_config.force_active_velocity_scale,
            reward_config.recovery_velocity_scale,
            force_active_scale,
            recovery_scale,
        )
        force_drift_scale = _blend_force_recovery_reward_scale(
            reward_config.force_active_drift_scale,
            reward_config.recovery_drift_scale,
            force_active_scale,
            recovery_scale,
        )
        force_stance_scale = _blend_force_recovery_reward_scale(
            reward_config.force_active_stance_scale,
            reward_config.recovery_stance_scale,
            force_active_scale,
            recovery_scale,
        )
        force_pose_scale = _blend_force_recovery_reward_scale(
            reward_config.force_active_pose_scale,
            reward_config.recovery_pose_scale,
            force_active_scale,
            recovery_scale,
        )
        terms = {
            "alive": reward_config.alive,
            "height": reward_config.height * float(self.np.exp(-40.0 * (base_z - self.config.target_base_z) ** 2)),
            "track_linear_velocity": force_velocity_scale * reward_config.track_linear_velocity * float(
                self.np.exp(-(xy_error + z_error) / (linear_std * linear_std))
            ),
            "track_angular_velocity": force_drift_scale * angular_tracking_weight * float(
                self.np.exp(-(yaw_error + 0.05 * roll_pitch_rate_error) / (yaw_std * yaw_std))
            ),
            "target_forward_velocity_band": (
                force_velocity_scale
                * reward_config.target_forward_velocity_band
                * (
                    _target_band_velocity_score(
                        actual_along_command,
                        lower=reward_config.target_forward_velocity_band_min,
                        upper=reward_config.target_forward_velocity_band_max,
                        decay_width=reward_config.target_forward_velocity_band_decay,
                    )
                    if command_speed > 0.05
                    else 0.0
                )
            ),
            "forward_progress": force_velocity_scale * reward_config.forward_progress * forward_progress,
            "velocity_shortfall_penalty": (
                force_velocity_scale * reward_config.velocity_shortfall_penalty * velocity_shortfall
            ),
            "velocity_overspeed_penalty": (
                force_velocity_scale * reward_config.velocity_overspeed_penalty * velocity_overspeed
            ),
            "lateral_velocity_l2": force_drift_scale * reward_config.lateral_velocity_l2 * lateral_error,
            "lateral_displacement_l2": reward_config.lateral_displacement_l2 * lateral_displacement_error,
            "lateral_return_velocity_l2": (
                force_drift_scale
                * reward_config.lateral_return_velocity_l2
                * lateral_return_velocity_error
                * lateral_return_velocity_error
            ),
            "yaw_rate_l2": force_drift_scale * reward_config.yaw_rate_l2 * yaw_error,
            "heading_l2": force_drift_scale * reward_config.heading_l2 * heading_error,
            "feet_air_time": self._feet_air_time_reward(command_speed, contacts),
            "contact_count_penalty": (
                reward_config.contact_count_penalty * max(0, 2 - contact_count)
                if moving_command
                else 0.0
            ),
            "stance_stagnation_penalty": (
                force_stance_scale * reward_config.stance_stagnation_penalty
                if moving_command and contact_count == len(FEET)
                else 0.0
            ),
            "pose": force_pose_scale * self._posture_reward_scale(command_speed) * self._variable_posture_reward(),
            "flat_orientation_l2": reward_config.flat_orientation_l2 * flat_orientation_l2,
            "body_ang_vel": reward_config.body_ang_vel * roll_pitch_rate_error,
            "joint_acc_l2": reward_config.joint_acc_l2 * float(self.np.dot(joint_acc, joint_acc)),
            "joint_pos_limits": reward_config.joint_pos_limits * self._joint_limit_cost(),
            "action_rate_l2": reward_config.action_rate_l2 * float(self.np.dot(action_rate, action_rate)),
            "residual_action_l2": 0.0,
            "residual_action_rate_l2": 0.0,
            "foot_slip": reward_config.foot_slip * self._foot_slip_cost(),
        }
        if (
            reward_config.force_active_min_contact_count > 0.0
            and (
                reward_config.force_active_contact_count_penalty != 0.0
                or reward_config.force_onset_contact_count_penalty != 0.0
            )
        ):
            terms.update(
                _force_active_support_reward_terms_value(
                    contact_count=contact_count,
                    active=bool(self._external_force_applied_this_step),
                    elapsed_since_force_start_s=force_elapsed_s,
                    min_contact_count=reward_config.force_active_min_contact_count,
                    active_penalty=reward_config.force_active_contact_count_penalty,
                    onset_penalty=reward_config.force_onset_contact_count_penalty,
                    onset_window_s=reward_config.external_force_excess_onset_window_s,
                )
            )
        if (
            reward_config.external_force_excess_penalty != 0.0
            or reward_config.external_force_excess_onset_penalty != 0.0
        ):
            if self.config.external_force_safe_limit_max_n > 0.0:
                safe_limit_n = self._external_force_safe_limit_n
                margin_n = self.config.external_force_safe_margin_n
            else:
                safe_limit_n = reward_config.unsafe_force_limit_n
                margin_n = reward_config.unsafe_force_margin_n
            terms.update(
                _external_force_excess_reward_terms_value(
                    force_n=float(self.np.linalg.norm(self._external_force_vector)),
                    safe_limit_n=float(safe_limit_n),
                    margin_n=float(margin_n),
                    active=bool(self._external_force_applied_this_step),
                    elapsed_since_force_start_s=force_elapsed_s,
                    excess_penalty=reward_config.external_force_excess_penalty,
                    onset_penalty=reward_config.external_force_excess_onset_penalty,
                    onset_window_s=reward_config.external_force_excess_onset_window_s,
                )
            )
        if (
            reward_config.external_force_violation_penalty != 0.0
            or reward_config.external_force_violation_onset_penalty != 0.0
            or reward_config.external_force_peak_delta_penalty != 0.0
            or reward_config.external_force_peak_delta_onset_penalty != 0.0
        ):
            if self.config.external_force_safe_limit_max_n > 0.0:
                safe_limit_n = self._external_force_safe_limit_n
                margin_n = self.config.external_force_safe_margin_n
            else:
                safe_limit_n = reward_config.unsafe_force_limit_n
                margin_n = reward_config.unsafe_force_margin_n
            tail_terms, updated_peak_cost = _external_force_tail_reward_terms_value(
                force_n=float(self.np.linalg.norm(self._external_force_vector)),
                safe_limit_n=float(safe_limit_n),
                margin_n=float(margin_n),
                active=bool(self._external_force_applied_this_step),
                elapsed_since_force_start_s=force_elapsed_s,
                previous_peak_cost=self._external_force_episode_peak_excess_cost,
                violation_penalty=reward_config.external_force_violation_penalty,
                violation_onset_penalty=reward_config.external_force_violation_onset_penalty,
                peak_delta_penalty=reward_config.external_force_peak_delta_penalty,
                peak_delta_onset_penalty=reward_config.external_force_peak_delta_onset_penalty,
                onset_window_s=reward_config.external_force_excess_onset_window_s,
            )
            self._external_force_episode_peak_excess_cost = updated_peak_cost
            terms.update(tail_terms)
        if (
            reward_config.force_yielding_velocity_tracking != 0.0
            or reward_config.force_yielding_velocity_shortfall_penalty != 0.0
        ):
            terms.update(
                _force_yielding_velocity_reward_terms_value(
                    np_module=self.np,
                    base_linear_velocity=tracking_lin_vel,
                    command=command,
                    force_vector=self._external_force_vector,
                    active=bool(self._external_force_applied_this_step),
                    elapsed_since_force_start_s=force_elapsed_s,
                    tracking_weight=reward_config.force_yielding_velocity_tracking,
                    tracking_sigma=reward_config.force_yielding_velocity_tracking_sigma,
                    shortfall_penalty=reward_config.force_yielding_velocity_shortfall_penalty,
                    onset_start_s=reward_config.force_yielding_velocity_onset_start_s,
                    onset_window_s=reward_config.force_yielding_velocity_onset_window_s,
                )
            )
        if self.config.policy_action_mode == "onnx_residual":
            if reward_config.residual_action_l2 != 0.0:
                terms["residual_action_l2"] = reward_config.residual_action_l2 * float(
                    self.np.dot(self._last_residual_action, self._last_residual_action)
                )
            if reward_config.residual_action_rate_l2 != 0.0:
                terms["residual_action_rate_l2"] = reward_config.residual_action_rate_l2 * float(
                    self.np.dot(self._last_residual_action_rate, self._last_residual_action_rate)
                )
        anti_collapse_scale = max(
            force_active_scale,
            recovery_scale * max(0.0, float(reward_config.anti_collapse_recovery_scale)),
        )
        if anti_collapse_scale > 0.0:
            z_safe = float(reward_config.anti_collapse_z_safe)
            height_deficit = max(0.0, z_safe - float(base_z))
            downward_velocity = max(
                0.0,
                -float(base_lin_vel[2]) - max(0.0, float(reward_config.anti_collapse_vz_safe)),
            )
            if reward_config.anti_collapse_height != 0.0:
                terms["anti_collapse_height"] = (
                    anti_collapse_scale
                    * reward_config.anti_collapse_height
                    * height_deficit
                    * height_deficit
                )
            if reward_config.anti_collapse_downward_velocity != 0.0:
                terms["anti_collapse_downward_velocity"] = (
                    anti_collapse_scale
                    * reward_config.anti_collapse_downward_velocity
                    * downward_velocity
                    * downward_velocity
                )
            if reward_config.anti_collapse_orientation != 0.0:
                terms["anti_collapse_orientation"] = (
                    anti_collapse_scale
                    * reward_config.anti_collapse_orientation
                    * flat_orientation_l2
                )
        if recovery_scale > 0.0 and moving_command:
            terms.update({
                "recovery_track_linear_velocity": (
                    reward_config.recovery_track_linear_velocity
                    * recovery_scale
                    * float(self.np.exp(-(xy_error + z_error) / (linear_std * linear_std)))
                ),
                "recovery_forward_progress": (
                    reward_config.recovery_forward_progress
                    * recovery_scale
                    * forward_progress
                ),
                "recovery_velocity_shortfall_penalty": (
                    reward_config.recovery_velocity_shortfall_penalty
                    * recovery_scale
                    * velocity_shortfall
                ),
                "recovery_lateral_velocity_l2": (
                    reward_config.recovery_lateral_velocity_l2
                    * recovery_scale
                    * lateral_error
                ),
                "recovery_yaw_rate_l2": (
                    reward_config.recovery_yaw_rate_l2
                    * recovery_scale
                    * yaw_error
                ),
                "recovery_stance_stagnation_penalty": (
                    reward_config.recovery_stance_stagnation_penalty
                    * recovery_scale
                    if contact_count == len(FEET)
                    else 0.0
                ),
            })
        terms.update(
            self._fullbody_tracking_reward_terms(
                base_ang_vel=base_ang_vel,
                previous_ctrl=previous_ctrl,
            )
        )
        return terms

    def _fullbody_tracking_reward_terms(
        self,
        *,
        base_ang_vel: Any,
        previous_ctrl: Any | None,
    ) -> dict[str, float]:
        reward_config = self.config.velocity_reward
        enabled = any(
            abs(float(value)) > 0.0
            for value in (
                reward_config.keypoint_tracking_comp,
                reward_config.reference_dynamics_tracking,
                reward_config.reference_force_tracking,
                reward_config.unsafe_force_penalty,
                reward_config.unsafe_force_peak_penalty,
                reward_config.unsafe_force_onset_penalty,
                reward_config.root_tracking,
                reward_config.root_yaw_tracking,
                reward_config.expert_action_tracking,
                reward_config.joint_pos_tracking,
                reward_config.joint_vel_tracking,
                reward_config.impact_force_l2,
                reward_config.torque_smoothness_l2,
            )
        )
        if not enabled:
            return {}

        self._update_fullbody_compliant_targets()
        target_root_position, target_root_yaw = self._fullbody_reference_root_target()
        target_keypoints = self._fullbody_target_keypoints_world(target_root_position, target_root_yaw)
        root_rotation = self._root_rotation_matrix()

        reference_dynamics_error = 0.0
        reference_dynamics_weight_sum = 0.0
        reference_force_error = 0.0
        reference_force_count = 0
        unsafe_force_cost = 0.0
        unsafe_force_count = 0
        unsafe_force_peak_cost = 0.0
        unsafe_force_onset_peak_cost = 0.0
        unsafe_force_enabled = any(
            value != 0.0
            for value in (
                reward_config.unsafe_force_penalty,
                reward_config.unsafe_force_peak_penalty,
                reward_config.unsafe_force_onset_penalty,
            )
        )
        force_elapsed_s = self._current_external_force_reward_elapsed_s()
        force_onset_active = (
            self._external_force_applied_this_step
            and force_elapsed_s >= 0.0
            and force_elapsed_s <= max(0.0, float(reward_config.unsafe_force_onset_window_s))
        )

        weighted_keypoint_error = 0.0
        keypoint_weight_sum = 0.0
        for name, target in target_keypoints.items():
            body_id = self.fullbody_tracking_body_ids.get(name, -1)
            if body_id < 0:
                continue
            weight = self._fullbody_keypoint_tracking_weight(name)
            if weight <= 0.0:
                continue
            actual = self.np.asarray(self.data.xpos[body_id], dtype=self.np.float64)
            weighted_keypoint_error += weight * float(self.np.sum(self.np.square(actual - target)))
            keypoint_weight_sum += weight

            if reward_config.reference_dynamics_tracking != 0.0:
                actual_velocity_root = root_rotation.T @ self.np.asarray(
                    self.data.cvel[body_id][3:6],
                    dtype=self.np.float64,
                )
                reference_velocity_root = self._fullbody_reference_velocities_root[name]
                position_error = float(self.np.sum(self.np.square(actual - target)))
                velocity_error = float(
                    self.np.sum(self.np.square(actual_velocity_root - reference_velocity_root))
                )
                reference_dynamics_error += weight * (
                    position_error
                    + max(0.0, float(reward_config.reference_dynamics_velocity_scale)) * velocity_error
                )
                reference_dynamics_weight_sum += weight

            reference_force_root = self._fullbody_reference_interaction_forces_root[name]
            reference_force_norm = float(self.np.linalg.norm(reference_force_root))
            if reference_force_norm > 1e-9:
                actual_force_root = self._fullbody_actual_interaction_force_root(name)
                if reward_config.reference_force_tracking != 0.0:
                    force_delta = actual_force_root - reference_force_root
                    reference_force_error += float(self.np.sum(self.np.square(force_delta)))
                    reference_force_count += 1

                if unsafe_force_enabled:
                    force_norm = float(self.np.linalg.norm(actual_force_root))
                    if self.config.external_force_safe_limit_max_n > 0.0:
                        limit_n = self._external_force_safe_limit_n
                        margin_n = self.config.external_force_safe_margin_n
                    else:
                        limit_n = reward_config.unsafe_force_limit_n
                        margin_n = reward_config.unsafe_force_margin_n
                    excess_cost = _force_excess_cost(
                        force_n=force_norm,
                        safe_limit_n=limit_n,
                        margin_n=margin_n,
                    )
                    unsafe_force_cost += excess_cost
                    unsafe_force_peak_cost = max(unsafe_force_peak_cost, excess_cost)
                    if force_onset_active:
                        unsafe_force_onset_peak_cost = max(unsafe_force_onset_peak_cost, excess_cost)
                    unsafe_force_count += 1

        terms: dict[str, float] = {}
        nominal_scale = self._nominal_gait_reference_scale()
        if reference_dynamics_weight_sum > 0.0 and reward_config.reference_dynamics_tracking != 0.0:
            sigma = max(1e-6, float(reward_config.reference_dynamics_tracking_sigma))
            mean_error = reference_dynamics_error / reference_dynamics_weight_sum
            terms["reference_dynamics_tracking"] = reward_config.reference_dynamics_tracking * float(
                self.np.exp(-mean_error / (sigma * sigma))
            )

        if reference_force_count > 0 and reward_config.reference_force_tracking != 0.0:
            sigma_n = max(1e-6, float(reward_config.reference_force_tracking_sigma_n))
            mean_force_error = reference_force_error / reference_force_count
            terms["reference_force_tracking"] = reward_config.reference_force_tracking * float(
                self.np.exp(-mean_force_error / (sigma_n * sigma_n))
            )

        if unsafe_force_count > 0 and reward_config.unsafe_force_penalty != 0.0:
            terms["unsafe_force_penalty"] = (
                reward_config.unsafe_force_penalty * unsafe_force_cost / unsafe_force_count
            )
        if unsafe_force_count > 0 and reward_config.unsafe_force_peak_penalty != 0.0:
            terms["unsafe_force_peak_penalty"] = (
                reward_config.unsafe_force_peak_penalty * unsafe_force_peak_cost
            )
        if force_onset_active and unsafe_force_count > 0 and reward_config.unsafe_force_onset_penalty != 0.0:
            terms["unsafe_force_onset_penalty"] = (
                reward_config.unsafe_force_onset_penalty * unsafe_force_onset_peak_cost
            )

        if keypoint_weight_sum > 0.0 and reward_config.keypoint_tracking_comp != 0.0:
            sigma = max(1e-6, float(reward_config.keypoint_tracking_sigma))
            mean_keypoint_error = weighted_keypoint_error / keypoint_weight_sum
            terms["keypoint_tracking_comp"] = nominal_scale * reward_config.keypoint_tracking_comp * float(
                self.np.exp(-mean_keypoint_error / (sigma * sigma))
            )

        if reward_config.root_tracking != 0.0:
            sigma = max(1e-6, float(reward_config.root_tracking_sigma))
            root_error = float(self.np.sum(self.np.square(self.data.qpos[:3] - target_root_position)))
            terms["root_tracking"] = nominal_scale * reward_config.root_tracking * float(
                self.np.exp(-root_error / (sigma * sigma))
            )

        if reward_config.root_yaw_tracking != 0.0:
            sigma = max(1e-6, float(reward_config.root_yaw_tracking_sigma))
            yaw_error = _wrap_angle(self._base_yaw() - target_root_yaw)
            roll_pitch_rate_error = float(self.np.sum(self.np.square(base_ang_vel[:2])))
            terms["root_yaw_tracking"] = nominal_scale * reward_config.root_yaw_tracking * float(
                self.np.exp(-(yaw_error * yaw_error + 0.05 * roll_pitch_rate_error) / (sigma * sigma))
            )

        if reward_config.expert_action_tracking != 0.0:
            action_ref = self._fullbody_rollout_action_reference()
            if action_ref is not None:
                terms["expert_action_tracking"] = nominal_scale * _expert_action_tracking_score(
                    self._last_action,
                    action_ref,
                    weight=reward_config.expert_action_tracking,
                    sigma=reward_config.expert_action_tracking_sigma,
                    np_module=self.np,
                )

        joint_pos_errors = []
        joint_vel_errors = []
        for joint_name in self.joint_names:
            qpos_adr, qvel_adr = self.joint_map[joint_name]
            q_ref, dq_ref = self._fullbody_reference_joint_state(joint_name)
            joint_pos_errors.append((float(self.data.qpos[qpos_adr]) - q_ref) ** 2)
            joint_vel_errors.append((float(self.data.qvel[qvel_adr]) - dq_ref) ** 2)

        if joint_pos_errors and reward_config.joint_pos_tracking != 0.0:
            sigma = max(1e-6, float(reward_config.joint_pos_tracking_sigma))
            terms["joint_pos_tracking"] = nominal_scale * reward_config.joint_pos_tracking * float(
                self.np.exp(-self.np.mean(joint_pos_errors) / (sigma * sigma))
            )
        if joint_vel_errors and reward_config.joint_vel_tracking != 0.0:
            sigma = max(1e-6, float(reward_config.joint_vel_tracking_sigma))
            terms["joint_vel_tracking"] = nominal_scale * reward_config.joint_vel_tracking * float(
                self.np.exp(-self.np.mean(joint_vel_errors) / (sigma * sigma))
            )

        if reward_config.impact_force_l2 != 0.0:
            impact = 0.0
            for body_id in self.foot_body_ids.values():
                if body_id >= 0:
                    impact += float(self.np.dot(self.data.cfrc_ext[body_id], self.data.cfrc_ext[body_id]))
            terms["impact_force_l2"] = reward_config.impact_force_l2 * impact

        if reward_config.torque_smoothness_l2 != 0.0 and previous_ctrl is not None:
            ctrl_delta = self.np.asarray(self.data.ctrl - previous_ctrl, dtype=self.np.float64)
            terms["torque_smoothness_l2"] = reward_config.torque_smoothness_l2 * float(
                self.np.dot(ctrl_delta, ctrl_delta)
            )

        return terms

    def _nominal_gait_reference_scale(self) -> float:
        reward_config = self.config.velocity_reward
        active_scale = self._external_force_active_scale()
        if active_scale > 0.0:
            return _blend_force_active_reward_scale(
                reward_config.nominal_gait_reference_force_scale,
                active_scale,
            )

        recovery_scale = self._external_force_recovery_scale()
        if recovery_scale <= 0.0:
            return 1.0
        recovery_phase = self._external_force_recovery_phase()
        configured_scale = (
            max(0.0, float(reward_config.nominal_gait_reference_recovery_scale))
            + (1.0 - max(0.0, float(reward_config.nominal_gait_reference_recovery_scale)))
            * recovery_phase
        )
        return _blend_force_active_reward_scale(configured_scale, recovery_scale)

    def _fullbody_actual_interaction_force_root(self, name: str) -> Any:
        applied_by_target = _map_applied_forces_to_compliant_targets(
            self._external_force_applied_by_body_root,
            np_module=self.np,
        )
        applied_root = applied_by_target.get(name)
        if applied_root is not None:
            return self.np.asarray(applied_root, dtype=self.np.float64)

        body_id = self.fullbody_tracking_body_ids.get(name, -1)
        if body_id >= 0:
            measured_world = self.np.asarray(self.data.cfrc_ext[body_id][3:6], dtype=self.np.float64)
            measured_root = self._root_rotation_matrix().T @ measured_world
        else:
            measured_root = self.np.zeros(3, dtype=self.np.float64)

        reference_root = self._fullbody_reference_interaction_forces_root.get(
            name,
            self.np.zeros(3, dtype=self.np.float64),
        )
        if float(self.np.linalg.norm(measured_root)) <= 1e-6 and float(self.np.linalg.norm(reference_root)) > 1e-9:
            # MuJoCo contact-force arrays do not always expose xfrc_applied as a
            # measured contact wrench. Fall back to the applied interaction force
            # so reference-force tracking still audits the intended force model.
            return reference_root
        return measured_root

    def _fullbody_keypoint_tracking_weight(self, name: str) -> float:
        reward_config = self.config.velocity_reward
        if name == "base_link":
            return max(0.0, float(reward_config.keypoint_tracking_base_weight))
        if name.endswith("_hip"):
            return max(0.0, float(reward_config.keypoint_tracking_hip_weight))
        if name.endswith("_calf"):
            return max(0.0, float(reward_config.keypoint_tracking_calf_weight))
        if name.endswith("_foot"):
            return max(0.0, float(reward_config.keypoint_tracking_foot_weight))
        return 1.0

    def _reset_fullbody_tracking_reference(self) -> None:
        self._fullbody_reference_root_position = self.np.asarray(self.data.qpos[:3], dtype=self.np.float64).copy()
        self._fullbody_reference_root_yaw = self._base_yaw()
        root_rotation = self._root_rotation_matrix()
        root_position = self.np.asarray(self.data.qpos[:3], dtype=self.np.float64)
        for name, body_id in self.fullbody_tracking_body_ids.items():
            if body_id < 0:
                continue
            body_position = self.np.asarray(self.data.xpos[body_id], dtype=self.np.float64)
            self._fullbody_keypoint_offsets_root[name] = root_rotation.T @ (body_position - root_position)
        for name in FULLBODY_TRACKING_KEYPOINTS:
            motion_target = self._fullbody_motion_target_root(name)
            self._fullbody_motion_targets_root[name] = motion_target
            self._fullbody_motion_target_velocities_root[name] = self.np.zeros(3, dtype=self.np.float64)
            self._fullbody_reference_positions_root[name] = motion_target.copy()
            self._fullbody_reference_velocities_root[name] = self.np.zeros(3, dtype=self.np.float64)
            self._fullbody_compliant_offsets_root[name] = self.np.zeros(3, dtype=self.np.float64)
            self._fullbody_compliant_velocities_root[name] = self.np.zeros(3, dtype=self.np.float64)
            self._fullbody_reference_driving_forces_root[name] = self.np.zeros(3, dtype=self.np.float64)
            self._fullbody_reference_interaction_forces_root[name] = self.np.zeros(3, dtype=self.np.float64)

    def _fullbody_reference_root_target(self) -> tuple[Any, float]:
        command = self._command(include_force_reference_governor=False)
        target_root_position = self._fullbody_reference_root_position.copy()
        if self.config.velocity_command_frame == "body":
            yaw = self._fullbody_reference_root_yaw
            c = math.cos(yaw)
            s = math.sin(yaw)
            target_root_position[:2] += self.np.asarray(
                [
                    c * command[0] - s * command[1],
                    s * command[0] + c * command[1],
                ],
                dtype=self.np.float64,
            ) * self._elapsed_s
        else:
            target_root_position[:2] += self.np.asarray(command[:2], dtype=self.np.float64) * self._elapsed_s
        target_root_position[2] = self._fullbody_nominal_base_height()
        target_root_yaw = self._fullbody_reference_root_yaw + float(command[2]) * self._elapsed_s
        governor_offset = self.np.asarray(self._force_reference_governor_offset, dtype=self.np.float64)
        if self.config.velocity_command_frame == "body":
            c = math.cos(target_root_yaw)
            s = math.sin(target_root_yaw)
            target_root_position[:2] += self.np.asarray(
                [
                    c * governor_offset[0] - s * governor_offset[1],
                    s * governor_offset[0] + c * governor_offset[1],
                ],
                dtype=self.np.float64,
            )
        else:
            target_root_position[:2] += governor_offset[:2]
        return target_root_position, target_root_yaw

    def _fullbody_target_keypoints_world(self, target_root_position: Any, target_root_yaw: float) -> dict[str, Any]:
        rotation = _yaw_to_matrix(target_root_yaw, self.np)
        targets = {}
        for name in self._fullbody_keypoint_offsets_root:
            targets[name] = target_root_position + rotation @ self._fullbody_reference_positions_root[name]
        return targets

    def _fullbody_motion_target_root(self, name: str) -> Any:
        rollout_target = self._fullbody_rollout_keypoint_target_root(name)
        if rollout_target is not None:
            return rollout_target
        return self._fullbody_keypoint_offsets_root[name] + self._fullbody_phase_keypoint_offset(name)

    def _fullbody_phase_keypoint_offset(self, name: str) -> Any:
        if not _uses_phase_trot_nominal_reference(self.config.fullbody_reference_mode):
            return self.np.zeros(3, dtype=self.np.float64)
        leg = _leg_prefix(name)
        if leg is None:
            return self.np.zeros(3, dtype=self.np.float64)

        phase = self._leg_gait_phase(leg)
        angle = 2.0 * math.pi * phase
        command_speed = min(1.0, abs(float(self._command()[0])) / 0.4) if abs(float(self._command()[0])) > 1e-6 else 0.0
        step_length = self._gait_step_length() * command_speed
        swing_height = self._gait_swing_height() * command_speed
        delta = self.np.zeros(3, dtype=self.np.float64)
        if name.endswith("_foot"):
            delta[0] = 0.5 * step_length * math.cos(angle)
            delta[2] = swing_height * max(0.0, math.sin(angle))
        elif name.endswith("_calf"):
            delta[0] = 0.25 * step_length * math.cos(angle)
            delta[2] = 0.35 * swing_height * max(0.0, math.sin(angle))
        return delta

    def _fullbody_reference_joint_state(self, joint_name: str) -> tuple[float, float]:
        rollout_state = self._fullbody_rollout_joint_state(joint_name)
        if rollout_state is not None:
            return rollout_state
        q_ref = self.default_targets[joint_name]
        if not _uses_phase_trot_nominal_reference(self.config.fullbody_reference_mode):
            return q_ref, 0.0

        leg = _leg_prefix(joint_name)
        if leg is None:
            return q_ref, 0.0
        phase = self._leg_gait_phase(leg)
        omega = 2.0 * math.pi * max(0.0, self._gait_frequency_hz())
        angle = 2.0 * math.pi * phase
        command_speed = min(1.0, abs(float(self._command()[0])) / 0.4) if abs(float(self._command()[0])) > 1e-6 else 0.0
        if "_thigh_" in joint_name:
            amplitude = float(self.config.gait_joint_thigh_amplitude) * command_speed
            return q_ref + amplitude * math.sin(angle), amplitude * omega * math.cos(angle)
        if "_calf_" in joint_name:
            amplitude = float(self.config.gait_joint_calf_amplitude) * command_speed
            swing = max(0.0, math.sin(angle))
            dq_ref = amplitude * omega * math.cos(angle) if math.sin(angle) > 0.0 else 0.0
            return q_ref - amplitude * swing, -dq_ref
        return q_ref, 0.0

    def _fullbody_rollout_reference_sample(self) -> Any | None:
        return self._fullbody_rollout_reference_sample_at(self._elapsed_s)

    def _fullbody_rollout_reference_sample_at(self, elapsed_s: float) -> Any | None:
        if self.nominal_gait_reference is None:
            return None
        if not _uses_rollout_nominal_reference(self.config.fullbody_reference_mode):
            return None
        phase = (max(0.0, float(elapsed_s)) * max(0.0, self._gait_frequency_hz())) % 1.0
        return self.nominal_gait_reference.sample(phase)

    def _fullbody_rollout_keypoint_target_root(self, name: str) -> Any | None:
        sample = self._fullbody_rollout_reference_sample()
        if sample is None:
            return None
        index = self._nominal_reference_keypoint_indices.get(name)
        if index is None or index >= int(sample.keypoint_pos_ref.shape[0]):
            return None
        return self.np.asarray(sample.keypoint_pos_ref[index], dtype=self.np.float64)

    def _fullbody_rollout_joint_state(self, joint_name: str) -> tuple[float, float] | None:
        sample = self._fullbody_rollout_reference_sample()
        if sample is None:
            return None
        index = self._nominal_reference_joint_indices.get(joint_name)
        if index is None or index >= int(sample.q_ref.shape[0]):
            return None
        return float(sample.q_ref[index]), float(sample.dq_ref[index])

    def _fullbody_rollout_action_reference(self) -> Any | None:
        sample = self._fullbody_rollout_reference_sample_at(self._elapsed_s - self.config.control_dt)
        if sample is None:
            return None
        action_ref = self.np.zeros(self.joint_action_size, dtype=self.np.float64)
        if self._nominal_reference_joint_indices:
            for action_index, joint_name in enumerate(self.joint_names):
                reference_index = self._nominal_reference_joint_indices.get(joint_name)
                if reference_index is None or reference_index >= int(sample.action_ref.shape[0]):
                    return None
                action_ref[action_index] = float(sample.action_ref[reference_index])
            return action_ref
        if int(sample.action_ref.shape[0]) < self.joint_action_size:
            return None
        return self.np.asarray(sample.action_ref[: self.joint_action_size], dtype=self.np.float64)

    def _fullbody_nominal_base_height(self) -> float:
        sample = self._fullbody_rollout_reference_sample()
        if sample is not None:
            return float(sample.base_height_ref)
        return self.config.target_base_z

    def _leg_gait_phase(self, leg: str) -> float:
        diagonal_offset = 0.0 if leg in {"FR", "RL"} else 0.5
        frequency = max(0.0, self._gait_frequency_hz())
        return (self._elapsed_s * frequency + diagonal_offset) % 1.0

    def _update_fullbody_compliant_targets(self) -> None:
        reward_config = self.config.velocity_reward
        admittance = max(0.0, float(reward_config.compliant_target_admittance))
        stiffness = max(0.0, float(reward_config.compliant_target_stiffness))
        damping = max(0.0, float(reward_config.compliant_target_damping))
        if admittance <= 0.0 and stiffness <= 0.0 and damping <= 0.0:
            return

        applied_by_target = _map_applied_forces_to_compliant_targets(
            self._external_force_applied_by_body_root,
            np_module=self.np,
        )
        dt = max(1e-6, float(self.config.control_dt))
        offset_clip = max(0.0, float(reward_config.compliant_target_offset_clip))
        velocity_clip = max(0.0, float(reward_config.compliant_target_velocity_clip))
        horizontal_only = float(reward_config.compliant_target_horizontal_only) > 0.0
        z_offset_clip = max(0.0, float(reward_config.compliant_target_z_offset_clip))
        z_velocity_clip = max(0.0, float(reward_config.compliant_target_z_velocity_clip))
        z_offset_limit = z_offset_clip if horizontal_only or z_offset_clip > 0.0 else None
        z_velocity_limit = z_velocity_clip if horizontal_only or z_velocity_clip > 0.0 else None

        for name in FULLBODY_TRACKING_KEYPOINTS:
            motion_target = self._fullbody_motion_target_root(name)
            previous_motion_target = self._fullbody_motion_targets_root[name]
            motion_velocity = (motion_target - previous_motion_target) / dt
            reference_position = self._fullbody_reference_positions_root[name]
            reference_velocity = self._fullbody_reference_velocities_root[name]
            interaction_force = applied_by_target.get(
                name,
                self.np.zeros(3, dtype=self.np.float64),
            )

            step = _integrate_bounded_compliant_reference(
                motion_target=motion_target,
                previous_motion_target=previous_motion_target,
                reference_position=reference_position,
                reference_velocity=reference_velocity,
                interaction_force=interaction_force,
                dt=dt,
                admittance=admittance,
                stiffness=stiffness,
                damping=damping,
                offset_clip=offset_clip,
                velocity_clip=velocity_clip,
                driving_force_limit=self._external_force_safe_limit_n,
                horizontal_only=horizontal_only,
                z_offset_clip=z_offset_limit,
                z_velocity_clip=z_velocity_limit,
            )
            reference_position = self.np.asarray(step.position, dtype=self.np.float64)
            reference_velocity = self.np.asarray(step.velocity, dtype=self.np.float64)
            motion_velocity = self.np.asarray(step.motion_velocity, dtype=self.np.float64)
            offset = self.np.asarray(step.offset, dtype=self.np.float64)
            compliant_velocity = self.np.asarray(step.compliant_velocity, dtype=self.np.float64)
            driving_force = self.np.asarray(step.driving_force, dtype=self.np.float64)
            interaction_force = self.np.asarray(step.interaction_force, dtype=self.np.float64)

            self._fullbody_motion_targets_root[name] = motion_target
            self._fullbody_motion_target_velocities_root[name] = motion_velocity
            self._fullbody_reference_positions_root[name] = reference_position
            self._fullbody_reference_velocities_root[name] = reference_velocity
            self._fullbody_compliant_offsets_root[name] = offset
            self._fullbody_compliant_velocities_root[name] = compliant_velocity
            self._fullbody_reference_driving_forces_root[name] = driving_force
            self._fullbody_reference_interaction_forces_root[name] = interaction_force

    def _active_fullbody_compliant_keypoints(self) -> set[str]:
        return set(
            _map_applied_forces_to_compliant_targets(
                self._external_force_applied_by_body_root,
                np_module=self.np,
            )
        )

    def _compliant_velocity_command(self, command: Any, force_active_scale: float) -> Any:
        reward_config = self.config.velocity_reward
        per_n = max(0.0, float(reward_config.force_active_compliance_velocity_per_n))
        clip_mps = max(0.0, float(reward_config.force_active_compliance_velocity_clip))
        if per_n <= 0.0 or clip_mps <= 0.0 or force_active_scale <= 0.0:
            return command
        if not self._external_force_applied_this_step:
            return command

        compliant_command = self.np.asarray(command, dtype=self.np.float64).copy()
        force_tracking_frame = self._velocity_tracking_linear_velocity(self._external_force_vector)
        offset_xy = force_active_scale * per_n * self.np.asarray(force_tracking_frame[:2], dtype=self.np.float64)
        offset_xy = _clip_vector_norm(offset_xy, clip_mps, self.np)
        compliant_command[:2] += offset_xy
        return compliant_command

    def _external_force_recovery_scale(self) -> float:
        window_s = max(0.0, float(self.config.velocity_reward.recovery_window_s))
        if window_s <= 0.0 or self._external_force_active_scale() > 0.0:
            return 0.0
        recent_end_s = self._recent_external_force_end_s()
        if recent_end_s is None:
            return 0.0
        time_since_end = self._elapsed_s - recent_end_s
        if time_since_end < 0.0 or time_since_end > window_s:
            return 0.0
        return 1.0

    def _external_force_recovery_phase(self) -> float:
        window_s = max(0.0, float(self.config.velocity_reward.recovery_window_s))
        if window_s <= 0.0 or self._external_force_active_scale() > 0.0:
            return 0.0
        recent_end_s = self._recent_external_force_end_s()
        if recent_end_s is None:
            return 0.0
        time_since_end = self._elapsed_s - recent_end_s
        if time_since_end < 0.0 or time_since_end > window_s:
            return 0.0
        return min(time_since_end / window_s, 1.0)

    def _external_force_active_scale(self) -> float:
        if self.config.external_force_mode == "spring":
            scales = [
                self._external_force_window_scale(spring.start_s, spring.end_s)
                for spring in self._external_force_springs
                if self._external_force_spring_is_currently_active(spring)
            ]
            return max(scales) if scales else 0.0
        scales = [
            self._external_force_window_scale(pulse.start_s, pulse.end_s)
            for pulse in self._external_force_pulses
            if self._external_force_pulse_is_currently_active(pulse)
        ]
        return max(scales) if scales else 0.0

    def _posture_reward_scale(self, command_speed: float) -> float:
        reward_config = self.config.velocity_reward
        return reward_config.pose_standing if command_speed < 0.05 else reward_config.pose_moving

    def _heading_error(self, command_speed: float, command_yaw_rate: float) -> float:
        if self.config.velocity_command_frame == "body":
            return 0.0
        if command_speed <= 0.05 or abs(command_yaw_rate) > 0.05:
            return 0.0
        yaw = self._base_yaw()
        return yaw * yaw

    def _feet_air_time_reward(self, command_speed: float, contacts: dict[str, bool]) -> float:
        if command_speed <= 0.05:
            self._foot_air_time = {foot: 0.0 for foot in FEET}
            self._previous_contacts = {foot: bool(contacts.get(foot, False)) for foot in FEET}
            return 0.0

        reward = 0.0
        for foot in FEET:
            contact = bool(contacts.get(foot, False))
            previous_contact = self._previous_contacts.get(foot, False)
            if contact:
                if not previous_contact:
                    air_time = self._foot_air_time.get(foot, 0.0)
                    reward += min(max(air_time - 0.06, 0.0), 0.20)
                self._foot_air_time[foot] = 0.0
            else:
                self._foot_air_time[foot] = self._foot_air_time.get(foot, 0.0) + self.config.control_dt
            self._previous_contacts[foot] = contact
        return self.config.velocity_reward.feet_air_time * reward

    def _variable_posture_reward(self) -> float:
        command = self._command()
        total_speed = float(self.np.linalg.norm(command[:2]) + abs(command[2]))
        if total_speed < 0.1:
            hip_std, thigh_std, calf_std = 0.05, 0.05, 0.1
        else:
            hip_std, thigh_std, calf_std = 0.3, 0.3, 0.6

        normalized_error = []
        for joint_name in self.joint_names:
            qpos_adr, _ = self.joint_map[joint_name]
            if "_hip_" in joint_name:
                std = hip_std
            elif "_thigh_" in joint_name:
                std = thigh_std
            else:
                std = calf_std
            error = float(self.data.qpos[qpos_adr]) - self.default_targets[joint_name]
            normalized_error.append((error * error) / (std * std))
        return float(self.np.exp(-self.np.mean(normalized_error)))

    def _joint_limit_cost(self) -> float:
        cost = 0.0
        for joint_name in self.joint_names:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0 or not bool(self.model.jnt_limited[joint_id]):
                continue
            qpos_adr, _ = self.joint_map[joint_name]
            lo, hi = self.model.jnt_range[joint_id]
            value = float(self.data.qpos[qpos_adr])
            if value < lo:
                cost += float((lo - value) ** 2)
            elif value > hi:
                cost += float((value - hi) ** 2)
        return cost

    def _foot_slip_cost(self) -> float:
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        cost = 0.0
        for foot, body_id in self.foot_body_ids.items():
            if body_id < 0 or not contacts.get(foot, False):
                continue
            linear_velocity = self.data.cvel[body_id][3:6]
            cost += float(linear_velocity[0] ** 2 + linear_velocity[1] ** 2)
        return cost

    def _terminated(self) -> bool:
        base_z = float(self.data.qpos[2]) if len(self.data.qpos) >= 3 else 0.0
        if base_z < self.config.fall_base_z:
            return True
        projected_gravity = self._projected_gravity()
        upright = -projected_gravity[2]
        if bool(upright < math.cos(self.config.bad_orientation_limit_rad)):
            return True
        return self._nonfoot_ground_contact()

    def _info(
        self,
        *,
        reward_terms: dict[str, float],
        reward_raw_terms: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        contacts = _foot_contacts(self.mujoco, self.model, self.data)
        qvel = self.data.qvel
        base_linear_velocity = self.np.asarray(qvel[:3], dtype=self.np.float64)
        tracking_linear_velocity = self._velocity_tracking_linear_velocity(base_linear_velocity)
        reference_dynamics = self._fullbody_reference_dynamics_info()
        joint_positions = [
            float(self.data.qpos[self.joint_map[joint_name][0]])
            for joint_name in self.joint_names
        ]
        joint_velocities = [
            float(self.data.qvel[self.joint_map[joint_name][1]])
            for joint_name in self.joint_names
        ]
        joint_torques = [
            float(self.data.ctrl[self.actuator_map[joint_name]])
            for joint_name in self.joint_names
        ]
        joint_target_positions = [
            float(self.default_targets[joint_name]) + float(self.config.action_scale) * float(action_value)
            for joint_name, action_value in zip(self.joint_names, self._last_action)
        ]
        applied_force_n = (
            float(self.np.linalg.norm(self._external_force_vector))
            if self._external_force_applied_this_step
            else 0.0
        )
        safe_limit_n = float(self._external_force_safe_limit_n)
        safe_margin_n = float(self.config.external_force_safe_margin_n)
        allowed_force_n = safe_limit_n + safe_margin_n if safe_limit_n > 0.0 else 0.0
        force_excess_n = max(0.0, applied_force_n - allowed_force_n) if safe_limit_n > 0.0 else 0.0
        current_force_window = self._current_external_force_window()
        current_force_elapsed_s = self._current_external_force_window_elapsed_s()
        current_safety_elapsed_s = self._current_force_safety_window_elapsed_s()
        return {
            "t": self._elapsed_s,
            "qpos": [float(v) for v in self.data.qpos],
            "qvel": [float(v) for v in self.data.qvel],
            "ctrl": [float(v) for v in self.data.ctrl],
            "base_z": float(self.data.qpos[2]) if len(self.data.qpos) >= 3 else 0.0,
            "base_position": [float(v) for v in self.data.qpos[:3]],
            "base_yaw": self._base_yaw() if len(self.data.qpos) >= 7 else 0.0,
            "base_linear_velocity": [float(v) for v in base_linear_velocity],
            "base_linear_velocity_tracking_frame": [float(v) for v in tracking_linear_velocity],
            "base_angular_velocity": [float(v) for v in qvel[3:6]],
            "task": self.config.task,
            "command": [float(v) for v in self._command()],
            "velocity_command_frame": self.config.velocity_command_frame,
            "external_force_observation": [float(v) for v in self._external_force_observation_values()],
            "push_applied": self._push_applied,
            "external_force_active": self._external_force_applied_this_step,
            "external_force_mode": self.config.external_force_mode,
            "external_force_body": self._external_force_body_name,
            "external_force_event_count": max(
                len(self._external_force_pulses),
                len({(spring.start_s, spring.end_s) for spring in self._external_force_springs}),
            ),
            "external_force_schedule": self._external_force_schedule_info(),
            "external_force_spring_types": [
                spring.spring_type for spring in self._external_force_springs
            ],
            "external_force_vector": [float(v) for v in self._external_force_vector],
            "external_force_magnitude": applied_force_n,
            "external_force_torque_vector": [float(v) for v in self._external_force_torque_vector],
            "external_force_torque_magnitude": float(self.np.linalg.norm(self._external_force_torque_vector))
            if self._external_force_applied_this_step
            else 0.0,
            "force_impedance_kp_scale": float(self._last_force_impedance_kp_scale),
            "force_impedance_kd_scale": float(self._last_force_impedance_kd_scale),
            "force_response_router_mode": self.config.force_response_router_mode,
            "force_response_profile": self.config.force_response_profile,
            "force_response_class": self._last_force_response_step.response_class,
            "force_response_impacted_body": self._last_force_response_step.impacted_body,
            "force_response_impacted_leg": self._last_force_response_step.impacted_leg,
            "force_response_impacted_leg_is_stance": self._last_force_response_step.impacted_leg_is_stance,
            "force_response_governor_enabled": bool(self._last_force_response_step.governor_enabled),
            "force_response_joint_kp_scales": [
                float(value) for value in self._last_force_response_joint_kp_scales
            ],
            "force_response_joint_kd_scales": [
                float(value) for value in self._last_force_response_joint_kd_scales
            ],
            "force_yielding_command_offset": [float(v) for v in self._force_yielding_command_offset()],
            "force_reference_governor_offset": [
                float(v) for v in self._force_reference_governor_offset
            ],
            "force_reference_governor_velocity": [
                float(v) for v in self._force_reference_governor_velocity
            ],
            "force_reference_governor_gate": float(self._force_reference_governor_gate),
            "force_safety_trigger_source": self.config.force_safety_trigger_source,
            "force_safety_detector_active": bool(self._force_safety_detector_active),
            "force_safety_detector_gate": float(self._force_safety_detector_gate),
            "force_safety_detector_score": float(self._force_safety_detector_score),
            "force_safety_detector_force_proxy": [
                float(v) for v in self._force_safety_detector_force_proxy
            ],
            "force_safety_current_window_elapsed_s": current_safety_elapsed_s,
            "external_force_start_s": self._external_force_start_s
            if math.isfinite(self._external_force_start_s)
            else None,
            "external_force_end_s": self._external_force_end_s
            if math.isfinite(self._external_force_end_s)
            else None,
            "external_force_current_window_start_s": (
                current_force_window[0] if current_force_window is not None else None
            ),
            "external_force_current_window_end_s": (
                current_force_window[1] if current_force_window is not None else None
            ),
            "external_force_current_window_elapsed_s": current_force_elapsed_s,
            "external_force_curriculum_progress": self._external_force_curriculum_progress,
            "external_force_safe_limit_n": float(self._external_force_safe_limit_n),
            "external_force_safe_margin_n": float(self.config.external_force_safe_margin_n),
            "external_force_allowed_n": allowed_force_n,
            "external_force_excess_n": force_excess_n,
            "external_force_episode_peak_excess_cost": float(self._external_force_episode_peak_excess_cost),
            "external_force_step_compliant": (
                _force_step_is_compliant(
                    force_n=applied_force_n,
                    safe_limit_n=safe_limit_n,
                    margin_n=safe_margin_n,
                )
                if self._external_force_applied_this_step and safe_limit_n > 0.0
                else None
            ),
            "fullbody_reference_dynamics": reference_dynamics,
            "projected_gravity": [float(v) for v in self._projected_gravity()],
            "contacts": contacts,
            "last_action": [float(v) for v in self._last_action],
            "joint_target_positions": joint_target_positions,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "policy_action_mode": self.config.policy_action_mode,
            "residual_action_scale": float(self.config.residual_action_scale),
            "onnx_action": [float(v) for v in self._last_onnx_action],
            "residual_action": [float(v) for v in self._last_residual_action],
            "safety_layer_action": [float(v) for v in self._last_safety_layer_action],
            "final_action": [float(v) for v in self._last_policy_action],
            "applied_action": [float(v) for v in self._last_action],
            "amp_observation": [float(v) for v in self.amp_observation()],
            "reward_terms": reward_terms,
            "reward_raw_terms": reward_raw_terms or reward_terms,
            "observation_names": self.observation_names,
            "action_names": self.joint_names,
            "policy_action_names": (
                list(SAFETY_LAYER_ACTION_NAMES)
                if self.config.policy_action_mode == "onnx_safety_layer"
                else list(self.joint_names)
            ),
        }

    def _fullbody_reference_dynamics_info(self) -> dict[str, float]:
        return {
            "max_compliant_offset": _max_vector_norm(self._fullbody_compliant_offsets_root, self.np),
            "max_compliant_velocity": _max_vector_norm(self._fullbody_compliant_velocities_root, self.np),
            "max_driving_force": _max_vector_norm(self._fullbody_reference_driving_forces_root, self.np),
            "max_interaction_force": _max_vector_norm(self._fullbody_reference_interaction_forces_root, self.np),
        }

    def _make_observation_names(self) -> tuple[str, ...]:
        if self.config.task in VELOCITY_TASKS:
            if self.config.observation_mode == "privileged":
                return self._velocity_privileged_observation_names()
            return self._velocity_policy_observation_names(
                include_external_force_observation=self.config.include_external_force_observation
            )

        command_names = (
            (
                "command_forward_velocity",
                "command_lateral_velocity",
                "command_yaw_rate",
            )
            if self.config.include_command
            else ()
        )
        return (
            "projected_gravity_x",
            "projected_gravity_y",
            "projected_gravity_z",
            "base_lin_vel_x",
            "base_lin_vel_y",
            "base_lin_vel_z",
            "base_ang_vel_x",
            "base_ang_vel_y",
            "base_ang_vel_z",
            *command_names,
            *[f"{name}_pos_rel" for name in self.joint_names],
            *[f"{name}_vel" for name in self.joint_names],
            *[f"{name}_last_action" for name in self.joint_names],
            "contact_FR",
            "contact_FL",
            "contact_RR",
            "contact_RL",
        )

    def _velocity_policy_observation_names(self, *, include_external_force_observation: bool) -> tuple[str, ...]:
        return (
            "base_ang_vel_x",
            "base_ang_vel_y",
            "base_ang_vel_z",
            "projected_gravity_x",
            "projected_gravity_y",
            "projected_gravity_z",
            "command_forward_velocity",
            "command_lateral_velocity",
            "command_yaw_rate",
            *[f"{name}_pos_rel" for name in self.joint_names],
            *[f"{name}_vel" for name in self.joint_names],
            *[f"{name}_last_action" for name in self.joint_names],
            *self._external_force_observation_names(force_include=include_external_force_observation),
        )

    def _velocity_privileged_observation_names(self) -> tuple[str, ...]:
        return (
            *self._velocity_policy_observation_names(include_external_force_observation=False),
            "priv_base_lin_vel_x",
            "priv_base_lin_vel_y",
            "priv_base_lin_vel_z",
            "priv_base_z",
            *[f"priv_{name}" for name in self._external_force_observation_names(force_include=True)],
            "priv_target_root_error_x",
            "priv_target_root_error_y",
            "priv_target_root_error_z",
            "priv_target_root_yaw_error",
            "priv_external_force_active_scale",
            "priv_external_force_recovery_scale",
            *(
                ("priv_external_force_safe_limit",)
                if self.config.external_force_safe_limit_max_n > 0.0
                else ()
            ),
            *[
                f"priv_{name}_target_diff_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[
                f"priv_{name}_vel_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[
                f"priv_{name}_compliant_offset_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[
                f"priv_{name}_compliant_vel_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[f"priv_{foot}_contact" for foot in FEET],
            *[
                f"priv_{foot}_contact_force_{axis}"
                for foot in FEET
                for axis in ("x", "y", "z")
            ],
            *[f"priv_{name}_ctrl" for name in self.joint_names],
            *[
                f"priv_{name}_reference_vel_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[
                f"priv_{name}_reference_driving_force_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[
                f"priv_{name}_reference_interaction_force_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[
                f"priv_{name}_actual_interaction_force_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
            *[
                f"priv_{name}_interaction_force_error_{axis}"
                for name in FULLBODY_TRACKING_KEYPOINTS
                for axis in ("x", "y", "z")
            ],
        )

    def _fullbody_privileged_reference_velocity_scale(self) -> float:
        reward_config = self.config.velocity_reward
        gait_scale = (
            2.0
            * math.pi
            * max(0.0, self._gait_frequency_hz())
            * max(0.0, self._gait_step_length() + self._gait_swing_height())
        )
        return max(
            1.0,
            gait_scale,
            float(reward_config.compliant_target_velocity_clip),
        )

    def _fullbody_privileged_force_scale(self) -> float:
        reward_config = self.config.velocity_reward
        return max(
            1.0,
            float(self.config.external_force_max_n),
            float(self.config.external_force_net_force_limit_n),
            float(reward_config.reference_force_tracking_sigma_n),
            float(reward_config.unsafe_force_limit_n),
        )

    def _external_force_observation_values(self, *, force_include: bool | None = None) -> list[float]:
        include = self.config.include_external_force_observation if force_include is None else force_include
        if not include:
            return []

        force_scale = max(
            1.0,
            float(self.config.external_force_max_n),
            float(self.config.external_force_net_force_limit_n),
        )
        torque_scale = max(
            1.0,
            float(self.config.external_force_torque_max_nm),
            float(self.config.external_force_net_torque_limit_nm),
        )
        if self._external_force_applied_this_step:
            force = self._velocity_tracking_linear_velocity(self._external_force_vector) / force_scale
            torque = self._velocity_tracking_linear_velocity(self._external_force_torque_vector) / torque_scale
        else:
            force = self.np.zeros(3, dtype=self.np.float64)
            torque = self.np.zeros(3, dtype=self.np.float64)

        force = self.np.clip(force, -1.0, 1.0)
        torque = self.np.clip(torque, -1.0, 1.0)
        return [
            float(self._external_force_active_scale()),
            float(self._external_force_recovery_scale()),
            float(self._external_force_recovery_phase()),
            *[float(v) for v in force],
            *[float(v) for v in torque],
        ]

    def _external_force_observation_names(self, *, force_include: bool | None = None) -> tuple[str, ...]:
        include = self.config.include_external_force_observation if force_include is None else force_include
        if not include:
            return ()
        return (
            "external_force_active_scale",
            "external_force_recovery_scale",
            "external_force_recovery_phase",
            "external_force_x",
            "external_force_y",
            "external_force_z",
            "external_torque_x",
            "external_torque_y",
            "external_torque_z",
        )

    def _sample_command(self) -> None:
        if self.config.randomize_commands:
            if float(self._rng.random()) < self.config.standing_command_prob:
                self._command_values = self.np.zeros(3, dtype=self.np.float64)
                return
            self._command_values = self.np.asarray(
                [
                    self._rng.uniform(*self.config.command_lin_vel_x_range),
                    self._rng.uniform(*self.config.command_lin_vel_y_range),
                    self._rng.uniform(*self.config.command_yaw_rate_range),
                ],
                dtype=self.np.float64,
            )
            return

        self._command_values = self.np.asarray(
            [
                self.config.target_forward_velocity,
                self.config.target_lateral_velocity,
                self.config.target_yaw_rate,
            ],
            dtype=self.np.float64,
        )

    def _sample_gait_parameters(self) -> None:
        if not self.config.randomize_gait_parameters:
            self._episode_gait_frequency_hz = float(self.config.gait_frequency_hz)
            self._episode_gait_step_length = float(self.config.gait_step_length)
            self._episode_gait_swing_height = float(self.config.gait_swing_height)
            return
        self._episode_gait_frequency_hz = float(self._rng.uniform(*self.config.gait_frequency_range))
        self._episode_gait_step_length = float(self._rng.uniform(*self.config.gait_step_length_range))
        self._episode_gait_swing_height = float(self._rng.uniform(*self.config.gait_swing_height_range))

    def _gait_frequency_hz(self) -> float:
        return float(self._episode_gait_frequency_hz)

    def _gait_step_length(self) -> float:
        return float(self._episode_gait_step_length)

    def _gait_swing_height(self) -> float:
        return float(self._episode_gait_swing_height)

    def _joint_vel_array(self) -> Any:
        return self.np.asarray(
            [float(self.data.qvel[self.joint_map[joint_name][1]]) for joint_name in self.joint_names],
            dtype=self.np.float64,
        )

    def _nonfoot_ground_contact(self) -> bool:
        foot_geoms = {"FR", "FL", "RR", "RL"}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_names = (
                self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "",
                self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "",
            )
            lower = " ".join(geom_names).lower()
            touches_ground = "floor" in lower or "ground" in lower
            if not touches_ground:
                continue
            other_geoms = {name for name in geom_names if name and name.lower() not in {"floor", "ground"}}
            if any(name not in foot_geoms for name in other_geoms):
                return True
        return False

    def _projected_gravity(self) -> Any:
        quat = self.np.asarray(self.data.qpos[3:7], dtype=self.np.float64)
        if self.np.linalg.norm(quat) == 0:
            quat = self.np.asarray([1.0, 0.0, 0.0, 0.0], dtype=self.np.float64)
        quat = quat / self.np.linalg.norm(quat)
        rotation = _quat_wxyz_to_matrix(quat, self.np)
        world_gravity = self.np.asarray([0.0, 0.0, -1.0], dtype=self.np.float64)
        return rotation.T @ world_gravity

    def _body_position_root(self, body_id: int) -> Any:
        root_position = self.np.asarray(self.data.qpos[:3], dtype=self.np.float64)
        body_position = self.np.asarray(self.data.xpos[body_id], dtype=self.np.float64)
        return self._root_rotation_matrix().T @ (body_position - root_position)

    def _body_linear_velocity_root(self, body_id: int) -> Any:
        body_velocity = self.np.asarray(self.data.cvel[body_id][3:6], dtype=self.np.float64)
        return self._root_rotation_matrix().T @ body_velocity

    def _body_position_external_spring_frame(self, spring: ExternalForceSpring) -> Any:
        body_position = self.np.asarray(self.data.xpos[spring.body_id], dtype=self.np.float64)
        return spring.frame_rotation_world.T @ (body_position - spring.frame_origin_world)

    def _body_velocity_external_spring_frame(self, spring: ExternalForceSpring) -> Any:
        body_velocity = self.np.asarray(self.data.cvel[spring.body_id][3:6], dtype=self.np.float64)
        return spring.frame_rotation_world.T @ body_velocity

    def _root_vector_to_world(self, vector: Any) -> Any:
        return self._root_rotation_matrix() @ self.np.asarray(vector, dtype=self.np.float64)

    def _root_rotation_matrix(self) -> Any:
        quat = self.np.asarray(self.data.qpos[3:7], dtype=self.np.float64)
        if self.np.linalg.norm(quat) == 0:
            quat = self.np.asarray([1.0, 0.0, 0.0, 0.0], dtype=self.np.float64)
        quat = quat / self.np.linalg.norm(quat)
        return _quat_wxyz_to_matrix(quat, self.np)

    def _velocity_tracking_linear_velocity(self, world_velocity: Any) -> Any:
        velocity = self.np.asarray(world_velocity, dtype=self.np.float64)
        if self.config.velocity_command_frame == "body":
            return self._world_vector_to_yaw_frame(velocity)
        return velocity

    def _world_vector_to_yaw_frame(self, vector: Any) -> Any:
        vector = self.np.asarray(vector, dtype=self.np.float64)
        yaw = self._base_yaw()
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return self.np.asarray(
            [
                cos_yaw * vector[0] + sin_yaw * vector[1],
                -sin_yaw * vector[0] + cos_yaw * vector[1],
                vector[2],
            ],
            dtype=self.np.float64,
        )

    def _base_yaw(self) -> float:
        quat = self.np.asarray(self.data.qpos[3:7], dtype=self.np.float64)
        if self.np.linalg.norm(quat) == 0:
            return 0.0
        quat = quat / self.np.linalg.norm(quat)
        rotation = _quat_wxyz_to_matrix(quat, self.np)
        return float(math.atan2(rotation[1, 0], rotation[0, 0]))

    def _nominal_command(self) -> Any:
        command = self.np.asarray(self._command_values, dtype=self.np.float64)
        multiplier = _recovery_command_multiplier(
            self.config.velocity_reward.recovery_command_velocity_scale,
            self._external_force_recovery_scale(),
            self._external_force_recovery_phase(),
        )
        if multiplier >= 1.0:
            return command
        scaled = command.copy()
        scaled[:2] *= multiplier
        return scaled

    def _force_yielding_command_offset(self) -> Any:
        nominal_command = self._nominal_command()
        yielded_command = self._force_yielding_command(nominal_command)
        return yielded_command - nominal_command

    def _force_yielding_command(self, command: Any) -> Any:
        force_tracking_frame = self._velocity_tracking_linear_velocity(self._external_force_vector)
        return _force_yielding_command_value(
            np_module=self.np,
            command=command,
            force_tracking_frame=force_tracking_frame,
            active=bool(self._external_force_applied_this_step),
            mode=self.config.force_yielding_command_mode,
            velocity_per_n=self.config.force_yielding_command_velocity_per_n,
            velocity_clip=self.config.force_yielding_command_velocity_clip,
            elapsed_since_force_start_s=self._current_external_force_window_elapsed_s(),
            pulse_start_s=self.config.force_yielding_command_pulse_start_s,
            pulse_duration_s=self.config.force_yielding_command_pulse_duration_s,
            pulse_recovery_s=self.config.force_yielding_command_pulse_recovery_s,
            pulse_post_clip=self.config.force_yielding_command_pulse_post_clip,
        )

    def _command(self, *, include_force_reference_governor: bool = True) -> Any:
        command = self._force_yielding_command(self._nominal_command())
        if not include_force_reference_governor:
            return command
        governor_velocity = self.np.asarray(self._force_reference_governor_velocity, dtype=self.np.float64)
        if float(self.np.linalg.norm(governor_velocity[:2])) <= 1e-12:
            return command
        adjusted = command.copy()
        adjusted[:2] += governor_velocity[:2]
        return adjusted

    def _resolve_fullbody_tracking_body_ids(self) -> dict[str, int]:
        body_ids = {}
        for body_name in FULLBODY_TRACKING_KEYPOINTS:
            body_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            body_ids[body_name] = int(body_id)
        return body_ids

    def _resolve_external_force_body_ids(self) -> tuple[tuple[int, str], ...]:
        body_ids = []
        for body_name in self.config.external_force_body_names:
            if not body_name:
                continue
            body_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if body_id < 0:
                raise ValueError(f"unknown external force body {body_name!r}")
            body_ids.append((int(body_id), body_name))
        return tuple(body_ids)

    def _external_force_config_enabled(self) -> bool:
        return (
            float(self.config.external_force_probability) > 0.0
            and float(self.config.external_force_max_n) > 0.0
        )


def _quat_wxyz_to_matrix(quat: Any, np: Any) -> Any:
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _yaw_to_matrix(yaw: float, np: Any) -> Any:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _sorted_range(values: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(values[0]), float(values[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def _smoothstep01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _external_force_window_scale_value(
    *,
    elapsed_s: float,
    start_s: float,
    end_s: float,
    transition_s: float,
) -> float:
    elapsed_s = float(elapsed_s)
    start_s = float(start_s)
    end_s = float(end_s)
    if elapsed_s < start_s or elapsed_s > end_s:
        return 0.0

    transition_s = max(0.0, float(transition_s))
    if transition_s <= 0.0:
        return 1.0
    window_s = max(0.0, end_s - start_s)
    if window_s <= 1e-9:
        return 1.0
    transition_s = min(transition_s, 0.5 * window_s)

    ramp_in = _smoothstep01((elapsed_s - start_s) / transition_s)
    ramp_out = _smoothstep01((end_s - elapsed_s) / transition_s)
    return max(0.0, min(1.0, ramp_in, ramp_out))


def _compute_external_spring_force_root_value(
    *,
    np: Any,
    spring_type: str,
    anchor_root: Any,
    anchor_velocity_root: Any,
    body_position_root: Any,
    body_velocity_root: Any,
    direction_root: Any,
    stiffness: float,
    damping: float,
    cap_n: float,
) -> Any:
    anchor_root = np.asarray(anchor_root, dtype=np.float64)
    anchor_velocity_root = np.asarray(anchor_velocity_root, dtype=np.float64)
    body_position_root = np.asarray(body_position_root, dtype=np.float64)
    body_velocity_root = np.asarray(body_velocity_root, dtype=np.float64)
    relative_position = anchor_root - body_position_root
    relative_velocity = anchor_velocity_root - body_velocity_root
    stiffness = max(0.0, float(stiffness))
    damping = max(0.0, float(damping))

    if spring_type == "resistive":
        direction = np.asarray(direction_root, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        direction = direction / direction_norm
        compression = float(np.dot(relative_position, direction))
        if compression <= 0.0:
            return np.zeros(3, dtype=np.float64)
        compression_velocity = float(np.dot(relative_velocity, direction))
        damping_force = max(0.0, damping * compression_velocity)
        force_magnitude = stiffness * compression + damping_force
        force_root = force_magnitude * direction
    else:
        force_root = stiffness * relative_position + damping * relative_velocity

    return _clip_vector_norm(force_root, cap_n, np)


def _tuple3(values: Any) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _vec_add(
    lhs: tuple[float, float, float],
    rhs: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _vec_sub(
    lhs: tuple[float, float, float],
    rhs: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _vec_scale(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    scale = float(scale)
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _clip_tuple_norm(vector: tuple[float, float, float], max_norm: float) -> tuple[float, float, float]:
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        return vector
    norm = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if norm <= max_norm or norm <= 1e-12:
        return vector
    scale = max_norm / norm
    return _vec_scale(vector, scale)


def _clip_vector_norm(vector: Any, max_norm: float, np: Any) -> Any:
    vector = np.asarray(vector, dtype=np.float64)
    max_norm = float(max_norm)
    if max_norm <= 0.0:
        return vector
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)


def _leg_prefix(name: str) -> str | None:
    for leg in FEET:
        if name.startswith(f"{leg}_"):
            return leg
    return None


def _uses_phase_trot_nominal_reference(fullbody_reference_mode: str) -> bool:
    return fullbody_reference_mode in {"phase_trot", "bounded_compliant"}


def _uses_rollout_nominal_reference(fullbody_reference_mode: str) -> bool:
    return fullbody_reference_mode in {"rollout", "rollout_compliant"}


def _blend_force_active_reward_scale(configured_scale: float, active_scale: float) -> float:
    configured_scale = max(0.0, float(configured_scale))
    active_scale = max(0.0, min(1.0, float(active_scale)))
    return 1.0 + active_scale * (configured_scale - 1.0)


def _blend_force_recovery_reward_scale(
    active_configured_scale: float,
    recovery_configured_scale: float,
    active_scale: float,
    recovery_scale: float,
) -> float:
    active_scale = max(0.0, min(1.0, float(active_scale)))
    recovery_scale = max(0.0, min(1.0, float(recovery_scale)))
    if active_scale > 0.0:
        return _blend_force_active_reward_scale(active_configured_scale, active_scale)
    return _blend_force_active_reward_scale(recovery_configured_scale, recovery_scale)


def _target_band_velocity_score(
    velocity: float,
    *,
    lower: float,
    upper: float,
    decay_width: float,
) -> float:
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        lower, upper = upper, lower
    velocity = float(velocity)
    if lower <= velocity <= upper:
        return 1.0
    decay_width = max(1e-9, float(decay_width))
    error = lower - velocity if velocity < lower else velocity - upper
    return max(0.0, 1.0 - error / decay_width)


def _target_band_velocity_errors(
    velocity: float,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        lower, upper = upper, lower
    velocity = float(velocity)
    return max(0.0, lower - velocity), max(0.0, velocity - upper)


def _expert_action_tracking_score(
    action: Any,
    expert_action: Any,
    *,
    weight: float,
    sigma: float,
    np_module: Any,
) -> float:
    action_array = np_module.asarray(action, dtype=np_module.float64).reshape(-1)
    expert_array = np_module.asarray(expert_action, dtype=np_module.float64).reshape(-1)
    if action_array.shape != expert_array.shape or action_array.size == 0:
        return 0.0
    sigma = max(1e-6, float(sigma))
    mean_error = float(np_module.mean(np_module.square(action_array - expert_array)))
    return float(weight) * float(np_module.exp(-mean_error / (sigma * sigma)))


def _compose_policy_action(
    policy_action: Any,
    *,
    mode: str,
    onnx_action: Any | None,
    residual_scale: float,
    np_module: Any,
) -> tuple[Any, Any, Any]:
    residual_action = np_module.clip(
        np_module.asarray(policy_action, dtype=np_module.float64).reshape(-1),
        -1.0,
        1.0,
    )
    if mode == "full_action":
        base_action = np_module.zeros_like(residual_action)
        return residual_action, base_action, residual_action
    if mode not in ONNX_POLICY_ACTION_MODES:
        raise ValueError(f"unknown policy action mode {mode!r}")
    if onnx_action is None:
        raise ValueError(f"{mode} action mode requires an ONNX action")
    base_action = np_module.clip(
        np_module.asarray(onnx_action, dtype=np_module.float64).reshape(-1),
        -1.0,
        1.0,
    )
    if mode == "onnx_safety_layer":
        return base_action, base_action, np_module.zeros_like(base_action)
    if base_action.shape != residual_action.shape:
        raise ValueError(
            f"ONNX action shape {base_action.shape} does not match residual action shape {residual_action.shape}"
        )
    final_action = np_module.clip(
        base_action + max(0.0, float(residual_scale)) * residual_action,
        -1.0,
        1.0,
    )
    return final_action, base_action, residual_action


def _layered_velocity_reward_terms(terms: dict[str, float]) -> dict[str, float]:
    return {
        "layer1_command_anchor": terms.get("target_forward_velocity_band", 0.0)
        + terms.get("track_linear_velocity", 0.0)
        + terms.get("track_angular_velocity", 0.0)
        + terms.get("forward_progress", 0.0)
        + terms.get("velocity_shortfall_penalty", 0.0)
        + terms.get("velocity_overspeed_penalty", 0.0)
        + terms.get("lateral_velocity_l2", 0.0)
        + terms.get("lateral_displacement_l2", 0.0)
        + terms.get("lateral_return_velocity_l2", 0.0)
        + terms.get("yaw_rate_l2", 0.0)
        + terms.get("heading_l2", 0.0),
        "layer2_nominal_gait_reference": terms.get("keypoint_tracking_comp", 0.0)
        + terms.get("root_tracking", 0.0)
        + terms.get("root_yaw_tracking", 0.0)
        + terms.get("expert_action_tracking", 0.0)
        + terms.get("joint_pos_tracking", 0.0)
        + terms.get("joint_vel_tracking", 0.0),
        "layer3_support_safety": terms.get("alive", 0.0)
        + terms.get("height", 0.0)
        + terms.get("flat_orientation_l2", 0.0)
        + terms.get("body_ang_vel", 0.0)
        + terms.get("joint_pos_limits", 0.0)
        + terms.get("termination_penalty", 0.0)
        + terms.get("force_active_contact_count_penalty", 0.0)
        + terms.get("force_onset_contact_count_penalty", 0.0),
        "layer4_active_force_anti_collapse": terms.get("anti_collapse_height", 0.0)
        + terms.get("anti_collapse_downward_velocity", 0.0)
        + terms.get("anti_collapse_orientation", 0.0)
        + terms.get("external_force_violation_penalty", 0.0)
        + terms.get("external_force_violation_onset_penalty", 0.0)
        + terms.get("external_force_peak_delta_penalty", 0.0)
        + terms.get("external_force_peak_delta_onset_penalty", 0.0),
        "layer5_post_force_recovery": terms.get("recovery_track_linear_velocity", 0.0)
        + terms.get("recovery_forward_progress", 0.0)
        + terms.get("recovery_velocity_shortfall_penalty", 0.0)
        + terms.get("recovery_lateral_velocity_l2", 0.0)
        + terms.get("recovery_yaw_rate_l2", 0.0)
        + terms.get("recovery_stance_stagnation_penalty", 0.0),
        "layer6_motion_quality_smoothness": terms.get("feet_air_time", 0.0)
        + terms.get("contact_count_penalty", 0.0)
        + terms.get("stance_stagnation_penalty", 0.0)
        + terms.get("pose", 0.0)
        + terms.get("joint_acc_l2", 0.0)
        + terms.get("action_rate_l2", 0.0)
        + terms.get("residual_action_l2", 0.0)
        + terms.get("residual_action_rate_l2", 0.0)
        + terms.get("foot_slip", 0.0)
        + terms.get("impact_force_l2", 0.0)
        + terms.get("torque_smoothness_l2", 0.0),
    }


def _recovery_command_multiplier(
    min_scale: float,
    recovery_scale: float,
    recovery_phase: float,
) -> float:
    min_scale = max(0.0, min(1.0, float(min_scale)))
    recovery_scale = max(0.0, min(1.0, float(recovery_scale)))
    recovery_phase = max(0.0, min(1.0, float(recovery_phase)))
    if recovery_scale <= 0.0:
        return 1.0
    target = min_scale + (1.0 - min_scale) * recovery_phase
    return 1.0 + recovery_scale * (target - 1.0)


def _lateral_displacement_error(
    actual_xy: Any,
    target_xy: Any,
    command_xy: Any,
) -> float:
    actual_x, actual_y = float(actual_xy[0]), float(actual_xy[1])
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    command_x, command_y = float(command_xy[0]), float(command_xy[1])
    dx = actual_x - target_x
    dy = actual_y - target_y
    command_norm = math.sqrt(command_x * command_x + command_y * command_y)
    if command_norm <= 1e-9:
        return math.sqrt(dx * dx + dy * dy)
    unit_x = command_x / command_norm
    unit_y = command_y / command_norm
    return -unit_y * dx + unit_x * dy


def _lateral_return_velocity_error(
    velocity_xy: Any,
    command_xy: Any,
    lateral_displacement: float,
    *,
    gain: float,
    clip: float,
) -> float:
    velocity_x, velocity_y = float(velocity_xy[0]), float(velocity_xy[1])
    command_x, command_y = float(command_xy[0]), float(command_xy[1])
    command_norm = math.sqrt(command_x * command_x + command_y * command_y)
    if command_norm <= 1e-9:
        actual_lateral_velocity = velocity_y
    else:
        unit_x = command_x / command_norm
        unit_y = command_y / command_norm
        actual_lateral_velocity = -unit_y * velocity_x + unit_x * velocity_y
    target_lateral_velocity = -float(gain) * float(lateral_displacement)
    velocity_clip = max(0.0, float(clip))
    if velocity_clip > 0.0:
        target_lateral_velocity = max(
            -velocity_clip,
            min(velocity_clip, target_lateral_velocity),
        )
    return actual_lateral_velocity - target_lateral_velocity


def _max_vector_norm(values: dict[str, Any], np: Any) -> float:
    if not values:
        return 0.0
    return max(float(np.linalg.norm(value)) for value in values.values())


def _default_targets(
    joint_names: tuple[str, ...],
    task: str,
    velocity_pose_profile: str = "official",
) -> dict[str, float]:
    if task in VELOCITY_TASKS:
        if velocity_pose_profile == "mjlab":
            return _mjlab_go2_reference_targets(joint_names)
        if velocity_pose_profile != "official":
            raise ValueError(
                f"unknown velocity pose profile {velocity_pose_profile!r}; "
                "choose one of: official, mjlab"
            )
        return _mjlab_go2_velocity_targets(joint_names)
    return _symmetric_stand_targets(joint_names)


def _symmetric_stand_targets(joint_names: tuple[str, ...]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for joint_name in joint_names:
        if "_hip_" in joint_name:
            targets[joint_name] = 0.0
        elif "_thigh_" in joint_name:
            targets[joint_name] = 0.608813
        elif "_calf_" in joint_name:
            targets[joint_name] = -1.21763
        else:
            targets[joint_name] = 0.0
    return targets


def _mjlab_go2_velocity_targets(joint_names: tuple[str, ...]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for joint_name in joint_names:
        if "_hip_" in joint_name:
            targets[joint_name] = 0.1 if joint_name.startswith(("FL_", "RL_")) else -0.1
        elif "_thigh_" in joint_name:
            targets[joint_name] = 1.0 if joint_name.startswith(("RL_", "RR_")) else 0.8
        elif "_calf_" in joint_name:
            targets[joint_name] = -1.5
        else:
            targets[joint_name] = 0.0
    return targets


def _mjlab_go2_reference_targets(joint_names: tuple[str, ...]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for joint_name in joint_names:
        if "_hip_" in joint_name:
            targets[joint_name] = 0.1 if joint_name.startswith(("FR_", "RR_")) else -0.1
        elif "_thigh_" in joint_name:
            targets[joint_name] = 0.9
        elif "_calf_" in joint_name:
            targets[joint_name] = -1.8
        else:
            targets[joint_name] = 0.0
    return targets


def _joint_names_for_leg_order(leg_order: tuple[str, ...]) -> tuple[str, ...]:
    unknown_legs = sorted(set(leg_order) - set(FEET))
    if unknown_legs:
        choices = ", ".join(FEET)
        unknown = ", ".join(unknown_legs)
        raise ValueError(f"unknown leg(s) in policy leg order: {unknown}; choose from: {choices}")
    if sorted(leg_order) != sorted(FEET):
        raise ValueError(
            "policy leg order must contain each leg exactly once: "
            f"{', '.join(FEET)}"
        )
    return tuple(
        f"{leg}_{joint}_joint"
        for leg in leg_order
        for joint in ("hip", "thigh", "calf")
    )


def make_velocity_reward_config(
    profile: str = "default",
    overrides: dict[str, float] | None = None,
) -> VelocityRewardConfig:
    if profile not in VELOCITY_REWARD_PROFILES:
        choices = ", ".join(sorted(VELOCITY_REWARD_PROFILES))
        raise ValueError(f"unknown velocity reward profile {profile!r}; choose one of: {choices}")

    config = VELOCITY_REWARD_PROFILES[profile]
    if not overrides:
        return config

    valid_names = {field.name for field in fields(VelocityRewardConfig)}
    unknown_names = sorted(set(overrides) - valid_names)
    if unknown_names:
        choices = ", ".join(sorted(valid_names))
        unknown = ", ".join(unknown_names)
        raise ValueError(f"unknown velocity reward override(s): {unknown}; choose from: {choices}")
    return replace(config, **{name: float(value) for name, value in overrides.items()})


def make_task_config(
    *,
    task: str,
    control_dt: float = 0.02,
    episode_length_s: float = 8.0,
    target_forward_velocity: float | None = None,
    target_lateral_velocity: float = 0.0,
    target_yaw_rate: float = 0.0,
    force_yielding_command_mode: str | None = None,
    force_yielding_command_velocity_per_n: float | None = None,
    force_yielding_command_velocity_clip: float | None = None,
    force_yielding_command_pulse_start_s: float | None = None,
    force_yielding_command_pulse_duration_s: float | None = None,
    force_yielding_command_pulse_recovery_s: float | None = None,
    force_yielding_command_pulse_post_clip: float | None = None,
    force_impedance_mode: str | None = None,
    force_impedance_joint_scope: str | None = None,
    force_impedance_kp_scale: float | None = None,
    force_impedance_kd_scale: float | None = None,
    force_impedance_delay_s: float | None = None,
    force_impedance_hold_s: float | None = None,
    force_impedance_recovery_s: float | None = None,
    force_impedance_tail_kp_scale: float | None = None,
    force_impedance_tail_kd_scale: float | None = None,
    force_reference_governor_mode: str | None = None,
    force_reference_governor_admittance_mps_per_n: float | None = None,
    force_reference_governor_damping: float | None = None,
    force_reference_governor_offset_clip_m: float | None = None,
    force_reference_governor_velocity_clip_mps: float | None = None,
    force_reference_governor_delay_s: float | None = None,
    force_reference_governor_hold_s: float | None = None,
    force_reference_governor_recovery_s: float | None = None,
    force_reference_governor_tail_admittance_scale: float | None = None,
    force_reference_governor_tail_offset_clip_scale: float | None = None,
    force_reference_governor_tail_velocity_clip_scale: float | None = None,
    force_response_router_mode: str | None = None,
    force_response_profile: str | None = None,
    force_response_foot_kp_scale: float | None = None,
    force_response_foot_kd_scale: float | None = None,
    force_safety_trigger_source: str | None = None,
    force_safety_detector_linear_acceleration_threshold: float | None = None,
    force_safety_detector_angular_acceleration_threshold: float | None = None,
    force_safety_detector_joint_error_threshold: float | None = None,
    force_safety_detector_joint_velocity_threshold: float | None = None,
    force_safety_detector_contact_loss: bool | None = None,
    force_safety_detector_enable_after_s: float | None = None,
    force_safety_detector_hold_s: float | None = None,
    force_safety_detector_recovery_s: float | None = None,
    force_safety_history_estimator_path: str | None = None,
    action_scale: float | None = None,
    action_smoothing: float | None = None,
    reward_scale: float | None = None,
    reset_settle_s: float | None = None,
    include_command: bool | None = None,
    randomize_commands: bool | None = None,
    standing_command_prob: float | None = None,
    command_lin_vel_x_range: tuple[float, float] | None = None,
    command_lin_vel_y_range: tuple[float, float] | None = None,
    command_yaw_rate_range: tuple[float, float] | None = None,
    push_time_s: float | None = None,
    push_linear_velocity: tuple[float, float, float] | None = None,
    push_angular_velocity: tuple[float, float, float] | None = None,
    external_force_mode: str | None = None,
    external_force_probability: float | None = None,
    external_force_body_names: tuple[str, ...] | None = None,
    external_force_active_body_count: int | None = None,
    external_force_event_count_range: tuple[int, int] | None = None,
    external_force_rest_s_range: tuple[float, float] | None = None,
    external_force_start_s_range: tuple[float, float] | None = None,
    external_force_duration_s_range: tuple[float, float] | None = None,
    external_force_min_n: float | None = None,
    external_force_max_n: float | None = None,
    external_force_curriculum_start_n: float | None = None,
    external_force_z_fraction: float | None = None,
    external_force_direction_angle_rad: float | None = None,
    external_force_direction_mode: str | None = None,
    external_force_lateral_probability: float | None = None,
    external_force_torque_max_nm: float | None = None,
    external_force_spring_stiffness_range: tuple[float, float] | None = None,
    external_force_spring_damping: float | None = None,
    external_force_guiding_probability: float | None = None,
    external_force_transition_s: float | None = None,
    external_force_net_force_limit_n: float | None = None,
    external_force_net_torque_limit_nm: float | None = None,
    external_force_reference_mass: float | None = None,
    external_force_reference_damping: float | None = None,
    external_force_reference_velocity_clip: float | None = None,
    external_force_reference_acceleration_clip: float | None = None,
    external_force_safe_limit_min_n: float | None = None,
    external_force_safe_limit_max_n: float | None = None,
    external_force_safe_margin_n: float | None = None,
    include_external_force_observation: bool = False,
    observation_mode: str | None = None,
    policy_action_mode: str | None = None,
    onnx_policy_path: str | None = None,
    onnx_normalizer_checkpoint: str | None = None,
    residual_action_scale: float | None = None,
    velocity_pose_profile: str = "official",
    policy_leg_order: tuple[str, ...] | None = None,
    velocity_reward_profile: str = "default",
    velocity_command_frame: str | None = None,
    fullbody_reference_mode: str | None = None,
    nominal_reference_dataset: str | None = None,
    gait_frequency_hz: float | None = None,
    gait_step_length: float | None = None,
    gait_swing_height: float | None = None,
    randomize_gait_parameters: bool | None = None,
    gait_frequency_range: tuple[float, float] | None = None,
    gait_step_length_range: tuple[float, float] | None = None,
    gait_swing_height_range: tuple[float, float] | None = None,
    gait_joint_thigh_amplitude: float | None = None,
    gait_joint_calf_amplitude: float | None = None,
    velocity_reward_overrides: dict[str, float] | None = None,
) -> StandBalanceEnvConfig:
    if task not in RL_TASKS:
        raise ValueError(f"unknown RL task {task!r}; choose one of: {', '.join(RL_TASKS)}")

    defaults = StandBalanceEnvConfig()
    is_velocity_task = task in VELOCITY_TASKS

    if target_forward_velocity is None:
        target_forward_velocity = DEFAULT_SLOW_WALK_VELOCITY if task == "slow_walk" else 0.0

    if task == "push_recovery":
        if push_time_s is None:
            push_time_s = 0.5
        if push_linear_velocity is None:
            push_linear_velocity = (0.35, -0.08, 0.0)
        if push_angular_velocity is None:
            push_angular_velocity = (0.0, 0.65, 0.0)

    if include_command is None:
        include_command = is_velocity_task
    if randomize_commands is None:
        randomize_commands = task == "velocity_flat"
    if action_scale is None:
        action_scale = 0.25 if is_velocity_task else 0.18
    if action_smoothing is None:
        action_smoothing = 0.5 if is_velocity_task else 0.8
    if reward_scale is None:
        reward_scale = 0.02 if is_velocity_task else 0.1
    if reset_settle_s is None:
        reset_settle_s = 0.5
    pd = PDConfig(kp=55.0, kd=2.0, torque_limit=45.0) if is_velocity_task else StandBalanceEnvConfig().pd
    velocity_reward = make_velocity_reward_config(
        velocity_reward_profile,
        velocity_reward_overrides,
    )
    safe_limit_min_n = (
        defaults.external_force_safe_limit_min_n
        if external_force_safe_limit_min_n is None
        else float(external_force_safe_limit_min_n)
    )
    safe_limit_max_n = (
        defaults.external_force_safe_limit_max_n
        if external_force_safe_limit_max_n is None
        else float(external_force_safe_limit_max_n)
    )
    if safe_limit_max_n > 0.0 and (safe_limit_min_n <= 0.0 or safe_limit_max_n < safe_limit_min_n):
        raise ValueError("external force safe limit requires 0 < min_n <= max_n")
    direction_mode = (
        defaults.external_force_direction_mode
        if external_force_direction_mode is None
        else str(external_force_direction_mode)
    )
    if direction_mode not in EXTERNAL_FORCE_DIRECTION_MODES:
        choices = ", ".join(EXTERNAL_FORCE_DIRECTION_MODES)
        raise ValueError(f"unknown external force direction mode {direction_mode!r}; choose one of: {choices}")
    lateral_probability = (
        defaults.external_force_lateral_probability
        if external_force_lateral_probability is None
        else float(external_force_lateral_probability)
    )
    if not 0.0 <= lateral_probability <= 1.0:
        raise ValueError("external force lateral probability must be in [0, 1]")
    yielding_command_mode = (
        defaults.force_yielding_command_mode
        if force_yielding_command_mode is None
        else str(force_yielding_command_mode)
    )
    if yielding_command_mode not in FORCE_YIELDING_COMMAND_MODES:
        choices = ", ".join(FORCE_YIELDING_COMMAND_MODES)
        raise ValueError(f"unknown force yielding command mode {yielding_command_mode!r}; choose one of: {choices}")
    yielding_pulse_start_s = (
        defaults.force_yielding_command_pulse_start_s
        if force_yielding_command_pulse_start_s is None
        else float(force_yielding_command_pulse_start_s)
    )
    yielding_pulse_duration_s = (
        defaults.force_yielding_command_pulse_duration_s
        if force_yielding_command_pulse_duration_s is None
        else float(force_yielding_command_pulse_duration_s)
    )
    yielding_pulse_recovery_s = (
        defaults.force_yielding_command_pulse_recovery_s
        if force_yielding_command_pulse_recovery_s is None
        else float(force_yielding_command_pulse_recovery_s)
    )
    yielding_pulse_post_clip = (
        defaults.force_yielding_command_pulse_post_clip
        if force_yielding_command_pulse_post_clip is None
        else float(force_yielding_command_pulse_post_clip)
    )
    if (
        yielding_pulse_start_s < 0.0
        or yielding_pulse_duration_s < 0.0
        or yielding_pulse_recovery_s < 0.0
        or yielding_pulse_post_clip < 0.0
    ):
        raise ValueError("force yielding command pulse parameters must be non-negative")
    impedance_mode = (
        defaults.force_impedance_mode
        if force_impedance_mode is None
        else str(force_impedance_mode)
    )
    if impedance_mode not in FORCE_IMPEDANCE_MODES:
        choices = ", ".join(FORCE_IMPEDANCE_MODES)
        raise ValueError(f"unknown force impedance mode {impedance_mode!r}; choose one of: {choices}")
    impedance_joint_scope = (
        defaults.force_impedance_joint_scope
        if force_impedance_joint_scope is None
        else str(force_impedance_joint_scope)
    )
    if impedance_joint_scope not in FORCE_IMPEDANCE_JOINT_SCOPES:
        choices = ", ".join(FORCE_IMPEDANCE_JOINT_SCOPES)
        raise ValueError(
            f"unknown force impedance joint scope {impedance_joint_scope!r}; choose one of: {choices}"
        )
    impedance_kp_scale = (
        defaults.force_impedance_kp_scale
        if force_impedance_kp_scale is None
        else float(force_impedance_kp_scale)
    )
    impedance_kd_scale = (
        defaults.force_impedance_kd_scale
        if force_impedance_kd_scale is None
        else float(force_impedance_kd_scale)
    )
    impedance_delay_s = (
        defaults.force_impedance_delay_s
        if force_impedance_delay_s is None
        else float(force_impedance_delay_s)
    )
    impedance_hold_s = (
        defaults.force_impedance_hold_s
        if force_impedance_hold_s is None
        else float(force_impedance_hold_s)
    )
    impedance_recovery_s = (
        defaults.force_impedance_recovery_s
        if force_impedance_recovery_s is None
        else float(force_impedance_recovery_s)
    )
    impedance_tail_kp_scale = (
        defaults.force_impedance_tail_kp_scale
        if force_impedance_tail_kp_scale is None
        else float(force_impedance_tail_kp_scale)
    )
    impedance_tail_kd_scale = (
        defaults.force_impedance_tail_kd_scale
        if force_impedance_tail_kd_scale is None
        else float(force_impedance_tail_kd_scale)
    )
    if (
        impedance_kp_scale < 0.0
        or impedance_kd_scale < 0.0
        or impedance_tail_kp_scale < 0.0
        or impedance_tail_kd_scale < 0.0
    ):
        raise ValueError("force impedance kp/kd scales must be non-negative")
    if impedance_delay_s < 0.0 or impedance_hold_s < 0.0 or impedance_recovery_s < 0.0:
        raise ValueError("force impedance hold/recovery durations must be non-negative")
    reference_governor_mode = (
        defaults.force_reference_governor_mode
        if force_reference_governor_mode is None
        else str(force_reference_governor_mode)
    )
    if reference_governor_mode not in FORCE_REFERENCE_GOVERNOR_MODES:
        choices = ", ".join(FORCE_REFERENCE_GOVERNOR_MODES)
        raise ValueError(
            f"unknown force reference governor mode {reference_governor_mode!r}; choose one of: {choices}"
        )
    reference_governor_admittance = (
        defaults.force_reference_governor_admittance_mps_per_n
        if force_reference_governor_admittance_mps_per_n is None
        else float(force_reference_governor_admittance_mps_per_n)
    )
    reference_governor_damping = (
        defaults.force_reference_governor_damping
        if force_reference_governor_damping is None
        else float(force_reference_governor_damping)
    )
    reference_governor_offset_clip = (
        defaults.force_reference_governor_offset_clip_m
        if force_reference_governor_offset_clip_m is None
        else float(force_reference_governor_offset_clip_m)
    )
    reference_governor_velocity_clip = (
        defaults.force_reference_governor_velocity_clip_mps
        if force_reference_governor_velocity_clip_mps is None
        else float(force_reference_governor_velocity_clip_mps)
    )
    reference_governor_delay_s = (
        defaults.force_reference_governor_delay_s
        if force_reference_governor_delay_s is None
        else float(force_reference_governor_delay_s)
    )
    reference_governor_hold_s = (
        defaults.force_reference_governor_hold_s
        if force_reference_governor_hold_s is None
        else float(force_reference_governor_hold_s)
    )
    reference_governor_recovery_s = (
        defaults.force_reference_governor_recovery_s
        if force_reference_governor_recovery_s is None
        else float(force_reference_governor_recovery_s)
    )
    reference_governor_tail_admittance_scale = (
        defaults.force_reference_governor_tail_admittance_scale
        if force_reference_governor_tail_admittance_scale is None
        else float(force_reference_governor_tail_admittance_scale)
    )
    reference_governor_tail_offset_clip_scale = (
        defaults.force_reference_governor_tail_offset_clip_scale
        if force_reference_governor_tail_offset_clip_scale is None
        else float(force_reference_governor_tail_offset_clip_scale)
    )
    reference_governor_tail_velocity_clip_scale = (
        defaults.force_reference_governor_tail_velocity_clip_scale
        if force_reference_governor_tail_velocity_clip_scale is None
        else float(force_reference_governor_tail_velocity_clip_scale)
    )
    if (
        reference_governor_admittance < 0.0
        or reference_governor_damping < 0.0
        or reference_governor_offset_clip < 0.0
        or reference_governor_velocity_clip < 0.0
        or reference_governor_tail_admittance_scale < 0.0
        or reference_governor_tail_offset_clip_scale < 0.0
        or reference_governor_tail_velocity_clip_scale < 0.0
    ):
        raise ValueError("force reference governor gains and clips must be non-negative")
    if reference_governor_delay_s < 0.0 or reference_governor_hold_s < 0.0 or reference_governor_recovery_s < 0.0:
        raise ValueError("force reference governor timing parameters must be non-negative")
    response_router_mode = (
        defaults.force_response_router_mode
        if force_response_router_mode is None
        else str(force_response_router_mode)
    )
    if response_router_mode not in FORCE_RESPONSE_ROUTER_MODES:
        choices = ", ".join(FORCE_RESPONSE_ROUTER_MODES)
        raise ValueError(f"unknown force response router mode {response_router_mode!r}; choose one of: {choices}")
    response_profile = (
        defaults.force_response_profile
        if force_response_profile is None
        else str(force_response_profile)
    )
    if response_profile not in FORCE_RESPONSE_PROFILES:
        choices = ", ".join(FORCE_RESPONSE_PROFILES)
        raise ValueError(f"unknown force response profile {response_profile!r}; choose one of: {choices}")
    response_foot_kp_scale = (
        defaults.force_response_foot_kp_scale
        if force_response_foot_kp_scale is None
        else float(force_response_foot_kp_scale)
    )
    response_foot_kd_scale = (
        defaults.force_response_foot_kd_scale
        if force_response_foot_kd_scale is None
        else float(force_response_foot_kd_scale)
    )
    if response_foot_kp_scale < 0.0 or response_foot_kd_scale < 0.0:
        raise ValueError("force response foot kp/kd scales must be non-negative")
    safety_trigger_source = (
        defaults.force_safety_trigger_source
        if force_safety_trigger_source is None
        else str(force_safety_trigger_source)
    )
    if safety_trigger_source not in FORCE_SAFETY_TRIGGER_SOURCES:
        choices = ", ".join(FORCE_SAFETY_TRIGGER_SOURCES)
        raise ValueError(
            f"unknown force safety trigger source {safety_trigger_source!r}; choose one of: {choices}"
        )
    safety_detector_linear_acceleration_threshold = (
        defaults.force_safety_detector_linear_acceleration_threshold
        if force_safety_detector_linear_acceleration_threshold is None
        else float(force_safety_detector_linear_acceleration_threshold)
    )
    safety_detector_angular_acceleration_threshold = (
        defaults.force_safety_detector_angular_acceleration_threshold
        if force_safety_detector_angular_acceleration_threshold is None
        else float(force_safety_detector_angular_acceleration_threshold)
    )
    safety_detector_joint_error_threshold = (
        defaults.force_safety_detector_joint_error_threshold
        if force_safety_detector_joint_error_threshold is None
        else float(force_safety_detector_joint_error_threshold)
    )
    safety_detector_joint_velocity_threshold = (
        defaults.force_safety_detector_joint_velocity_threshold
        if force_safety_detector_joint_velocity_threshold is None
        else float(force_safety_detector_joint_velocity_threshold)
    )
    safety_detector_contact_loss = (
        defaults.force_safety_detector_contact_loss
        if force_safety_detector_contact_loss is None
        else bool(force_safety_detector_contact_loss)
    )
    safety_detector_enable_after_s = (
        defaults.force_safety_detector_enable_after_s
        if force_safety_detector_enable_after_s is None
        else float(force_safety_detector_enable_after_s)
    )
    safety_detector_hold_s = (
        defaults.force_safety_detector_hold_s
        if force_safety_detector_hold_s is None
        else float(force_safety_detector_hold_s)
    )
    safety_detector_recovery_s = (
        defaults.force_safety_detector_recovery_s
        if force_safety_detector_recovery_s is None
        else float(force_safety_detector_recovery_s)
    )
    if (
        safety_detector_linear_acceleration_threshold < 0.0
        or safety_detector_angular_acceleration_threshold < 0.0
        or safety_detector_joint_error_threshold < 0.0
        or safety_detector_joint_velocity_threshold < 0.0
    ):
        raise ValueError("force safety detector thresholds must be non-negative")
    if (
        safety_detector_enable_after_s < 0.0
        or safety_detector_hold_s < 0.0
        or safety_detector_recovery_s < 0.0
    ):
        raise ValueError("force safety detector timing parameters must be non-negative")
    safety_history_estimator_path = (
        defaults.force_safety_history_estimator_path
        if force_safety_history_estimator_path is None
        else str(force_safety_history_estimator_path)
    )
    if safety_trigger_source in HISTORY_ESTIMATOR_FORCE_SAFETY_TRIGGER_SOURCES and not safety_history_estimator_path:
        raise ValueError(f"{safety_trigger_source} requires force_safety_history_estimator_path")

    return StandBalanceEnvConfig(
        task=task,
        control_dt=control_dt,
        episode_length_s=episode_length_s,
        action_scale=action_scale,
        action_smoothing=action_smoothing,
        reward_scale=reward_scale,
        target_base_z=0.4 if is_velocity_task else 0.36,
        init_base_z=0.4 if is_velocity_task else 0.36,
        fall_base_z=0.22 if is_velocity_task else 0.25,
        bad_orientation_limit_rad=MJLAB_BAD_ORIENTATION_LIMIT_RAD if is_velocity_task else 1.2132252231493863,
        reset_settle_s=reset_settle_s,
        include_command=include_command,
        randomize_commands=randomize_commands,
        standing_command_prob=(
            defaults.standing_command_prob
            if standing_command_prob is None
            else standing_command_prob
        ),
        command_lin_vel_x_range=command_lin_vel_x_range or defaults.command_lin_vel_x_range,
        command_lin_vel_y_range=command_lin_vel_y_range or defaults.command_lin_vel_y_range,
        command_yaw_rate_range=command_yaw_rate_range or defaults.command_yaw_rate_range,
        target_forward_velocity=target_forward_velocity,
        target_lateral_velocity=target_lateral_velocity,
        target_yaw_rate=target_yaw_rate,
        force_yielding_command_mode=yielding_command_mode,
        force_yielding_command_velocity_per_n=(
            defaults.force_yielding_command_velocity_per_n
            if force_yielding_command_velocity_per_n is None
            else force_yielding_command_velocity_per_n
        ),
        force_yielding_command_velocity_clip=(
            defaults.force_yielding_command_velocity_clip
            if force_yielding_command_velocity_clip is None
            else force_yielding_command_velocity_clip
        ),
        force_yielding_command_pulse_start_s=yielding_pulse_start_s,
        force_yielding_command_pulse_duration_s=yielding_pulse_duration_s,
        force_yielding_command_pulse_recovery_s=yielding_pulse_recovery_s,
        force_yielding_command_pulse_post_clip=yielding_pulse_post_clip,
        force_impedance_mode=impedance_mode,
        force_impedance_joint_scope=impedance_joint_scope,
        force_impedance_kp_scale=impedance_kp_scale,
        force_impedance_kd_scale=impedance_kd_scale,
        force_impedance_delay_s=impedance_delay_s,
        force_impedance_hold_s=impedance_hold_s,
        force_impedance_recovery_s=impedance_recovery_s,
        force_impedance_tail_kp_scale=impedance_tail_kp_scale,
        force_impedance_tail_kd_scale=impedance_tail_kd_scale,
        force_reference_governor_mode=reference_governor_mode,
        force_reference_governor_admittance_mps_per_n=reference_governor_admittance,
        force_reference_governor_damping=reference_governor_damping,
        force_reference_governor_offset_clip_m=reference_governor_offset_clip,
        force_reference_governor_velocity_clip_mps=reference_governor_velocity_clip,
        force_reference_governor_delay_s=reference_governor_delay_s,
        force_reference_governor_hold_s=reference_governor_hold_s,
        force_reference_governor_recovery_s=reference_governor_recovery_s,
        force_reference_governor_tail_admittance_scale=reference_governor_tail_admittance_scale,
        force_reference_governor_tail_offset_clip_scale=reference_governor_tail_offset_clip_scale,
        force_reference_governor_tail_velocity_clip_scale=reference_governor_tail_velocity_clip_scale,
        force_response_router_mode=response_router_mode,
        force_response_profile=response_profile,
        force_response_foot_kp_scale=response_foot_kp_scale,
        force_response_foot_kd_scale=response_foot_kd_scale,
        force_safety_trigger_source=safety_trigger_source,
        force_safety_detector_linear_acceleration_threshold=safety_detector_linear_acceleration_threshold,
        force_safety_detector_angular_acceleration_threshold=safety_detector_angular_acceleration_threshold,
        force_safety_detector_joint_error_threshold=safety_detector_joint_error_threshold,
        force_safety_detector_joint_velocity_threshold=safety_detector_joint_velocity_threshold,
        force_safety_detector_contact_loss=safety_detector_contact_loss,
        force_safety_detector_enable_after_s=safety_detector_enable_after_s,
        force_safety_detector_hold_s=safety_detector_hold_s,
        force_safety_detector_recovery_s=safety_detector_recovery_s,
        force_safety_history_estimator_path=safety_history_estimator_path,
        push_time_s=push_time_s,
        push_linear_velocity=push_linear_velocity or (0.0, 0.0, 0.0),
        push_angular_velocity=push_angular_velocity or (0.0, 0.0, 0.0),
        external_force_mode=defaults.external_force_mode if external_force_mode is None else external_force_mode,
        external_force_probability=(
            defaults.external_force_probability
            if external_force_probability is None
            else external_force_probability
        ),
        external_force_body_names=external_force_body_names or defaults.external_force_body_names,
        external_force_active_body_count=(
            defaults.external_force_active_body_count
            if external_force_active_body_count is None
            else external_force_active_body_count
        ),
        external_force_event_count_range=(
            defaults.external_force_event_count_range
            if external_force_event_count_range is None
            else external_force_event_count_range
        ),
        external_force_rest_s_range=(
            defaults.external_force_rest_s_range
            if external_force_rest_s_range is None
            else external_force_rest_s_range
        ),
        external_force_start_s_range=(
            defaults.external_force_start_s_range
            if external_force_start_s_range is None
            else external_force_start_s_range
        ),
        external_force_duration_s_range=(
            defaults.external_force_duration_s_range
            if external_force_duration_s_range is None
            else external_force_duration_s_range
        ),
        external_force_min_n=defaults.external_force_min_n if external_force_min_n is None else external_force_min_n,
        external_force_max_n=defaults.external_force_max_n if external_force_max_n is None else external_force_max_n,
        external_force_curriculum_start_n=external_force_curriculum_start_n,
        external_force_z_fraction=(
            defaults.external_force_z_fraction
            if external_force_z_fraction is None
            else external_force_z_fraction
        ),
        external_force_direction_angle_rad=external_force_direction_angle_rad,
        external_force_direction_mode=direction_mode,
        external_force_lateral_probability=lateral_probability,
        external_force_torque_max_nm=(
            defaults.external_force_torque_max_nm
            if external_force_torque_max_nm is None
            else external_force_torque_max_nm
        ),
        external_force_spring_stiffness_range=(
            defaults.external_force_spring_stiffness_range
            if external_force_spring_stiffness_range is None
            else external_force_spring_stiffness_range
        ),
        external_force_spring_damping=(
            defaults.external_force_spring_damping
            if external_force_spring_damping is None
            else external_force_spring_damping
        ),
        external_force_guiding_probability=(
            defaults.external_force_guiding_probability
            if external_force_guiding_probability is None
            else external_force_guiding_probability
        ),
        external_force_transition_s=(
            defaults.external_force_transition_s
            if external_force_transition_s is None
            else external_force_transition_s
        ),
        external_force_net_force_limit_n=(
            defaults.external_force_net_force_limit_n
            if external_force_net_force_limit_n is None
            else external_force_net_force_limit_n
        ),
        external_force_net_torque_limit_nm=(
            defaults.external_force_net_torque_limit_nm
            if external_force_net_torque_limit_nm is None
            else external_force_net_torque_limit_nm
        ),
        external_force_reference_mass=(
            defaults.external_force_reference_mass
            if external_force_reference_mass is None
            else external_force_reference_mass
        ),
        external_force_reference_damping=(
            defaults.external_force_reference_damping
            if external_force_reference_damping is None
            else external_force_reference_damping
        ),
        external_force_reference_velocity_clip=(
            defaults.external_force_reference_velocity_clip
            if external_force_reference_velocity_clip is None
            else external_force_reference_velocity_clip
        ),
        external_force_reference_acceleration_clip=(
            defaults.external_force_reference_acceleration_clip
            if external_force_reference_acceleration_clip is None
            else external_force_reference_acceleration_clip
        ),
        external_force_safe_limit_min_n=safe_limit_min_n,
        external_force_safe_limit_max_n=safe_limit_max_n,
        external_force_safe_margin_n=(
            defaults.external_force_safe_margin_n
            if external_force_safe_margin_n is None
            else external_force_safe_margin_n
        ),
        include_external_force_observation=include_external_force_observation,
        observation_mode=defaults.observation_mode if observation_mode is None else observation_mode,
        policy_action_mode=defaults.policy_action_mode if policy_action_mode is None else policy_action_mode,
        onnx_policy_path=defaults.onnx_policy_path if onnx_policy_path is None else onnx_policy_path,
        onnx_normalizer_checkpoint=(
            defaults.onnx_normalizer_checkpoint
            if onnx_normalizer_checkpoint is None
            else onnx_normalizer_checkpoint
        ),
        residual_action_scale=(
            defaults.residual_action_scale if residual_action_scale is None else residual_action_scale
        ),
        velocity_pose_profile=velocity_pose_profile,
        policy_leg_order=policy_leg_order or FEET,
        velocity_reward_profile=velocity_reward_profile,
        velocity_command_frame=(
            defaults.velocity_command_frame
            if velocity_command_frame is None
            else velocity_command_frame
        ),
        fullbody_reference_mode=(
            defaults.fullbody_reference_mode
            if fullbody_reference_mode is None
            else fullbody_reference_mode
        ),
        nominal_reference_dataset=nominal_reference_dataset,
        gait_frequency_hz=defaults.gait_frequency_hz if gait_frequency_hz is None else gait_frequency_hz,
        gait_step_length=defaults.gait_step_length if gait_step_length is None else gait_step_length,
        gait_swing_height=defaults.gait_swing_height if gait_swing_height is None else gait_swing_height,
        randomize_gait_parameters=(
            defaults.randomize_gait_parameters
            if randomize_gait_parameters is None
            else randomize_gait_parameters
        ),
        gait_frequency_range=gait_frequency_range or defaults.gait_frequency_range,
        gait_step_length_range=gait_step_length_range or defaults.gait_step_length_range,
        gait_swing_height_range=gait_swing_height_range or defaults.gait_swing_height_range,
        gait_joint_thigh_amplitude=(
            defaults.gait_joint_thigh_amplitude
            if gait_joint_thigh_amplitude is None
            else gait_joint_thigh_amplitude
        ),
        gait_joint_calf_amplitude=(
            defaults.gait_joint_calf_amplitude
            if gait_joint_calf_amplitude is None
            else gait_joint_calf_amplitude
        ),
        velocity_reward=velocity_reward,
        pd=pd,
    )
