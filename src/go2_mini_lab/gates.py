from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gentle30NGateSpec:
    min_no_force_success: float = 1.0
    min_no_force_straightness: float = 0.9
    max_no_force_forward_velocity: float = 0.45
    max_no_force_abs_lateral_velocity: float = 0.08
    max_no_force_abs_yaw_rate: float = 0.12
    max_no_force_abs_lateral_displacement: float = 0.35
    min_force_success: float = 0.9
    max_force_fall_rate: float = 0.1
    min_force_recovery_success: float = 0.9
    min_post_force_forward: float = 0.32
    max_post_force_abs_lateral: float = 0.30
    max_post_force_abs_yaw: float = 0.30
    max_force_abs_lateral_displacement: float = 0.45
    min_force_step_compliance: float = 0.9
    min_episode_compliance: float = 0.9


GENTLE_30N_GATE = Gentle30NGateSpec()
