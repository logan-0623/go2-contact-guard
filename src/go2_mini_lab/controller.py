from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin


LEGS = ("FR", "FL", "RR", "RL")
JOINTS_PER_LEG = ("hip", "thigh", "calf")

UNITREE_GO2_JOINTS = tuple(
    f"{leg}_{joint}_joint" for leg in LEGS for joint in JOINTS_PER_LEG
)


@dataclass(frozen=True)
class GaitConfig:
    stride_frequency: float = 1.2
    duty_factor: float = 0.62
    hip_swing: float = 0.03
    thigh_swing: float = 0.12
    calf_swing: float = 0.18
    nominal_hip: float = 0.0
    nominal_thigh: float = 0.608813
    nominal_calf: float = -1.21763


@dataclass(frozen=True)
class PDConfig:
    kp: float = 80.0
    kd: float = 4.5
    torque_limit: float = 45.0


@dataclass(frozen=True)
class ControllerPreset:
    gait: GaitConfig
    pd: PDConfig
    description: str


CONTROLLER_PRESETS: dict[str, ControllerPreset] = {
    "stand": ControllerPreset(
        gait=GaitConfig(hip_swing=0.0, thigh_swing=0.0, calf_swing=0.0),
        pd=PDConfig(kp=90.0, kd=5.0, torque_limit=45.0),
        description="Hold Unitree's stand-up pose for baseline state and contact checks.",
    ),
    "slow_trot": ControllerPreset(
        gait=GaitConfig(),
        pd=PDConfig(kp=80.0, kd=4.5, torque_limit=45.0),
        description="Conservative teaching trot with small joint excursions.",
    ),
    "fast_trot": ControllerPreset(
        gait=GaitConfig(
            stride_frequency=1.85,
            hip_swing=0.045,
            thigh_swing=0.16,
            calf_swing=0.22,
        ),
        pd=PDConfig(kp=90.0, kd=5.0, torque_limit=45.0),
        description="Faster open-loop trot for comparing timing and tracking stress.",
    ),
    "high_step": ControllerPreset(
        gait=GaitConfig(
            stride_frequency=1.05,
            hip_swing=0.035,
            thigh_swing=0.18,
            calf_swing=0.32,
        ),
        pd=PDConfig(kp=85.0, kd=5.0, torque_limit=45.0),
        description="Higher swing motion that makes gait phase easier to see.",
    ),
    "unstable_demo": ControllerPreset(
        gait=GaitConfig(
            stride_frequency=2.25,
            hip_swing=0.11,
            thigh_swing=0.38,
            calf_swing=0.5,
            nominal_thigh=0.78,
            nominal_calf=-1.55,
        ),
        pd=PDConfig(kp=34.0, kd=1.4, torque_limit=24.0),
        description="Intentionally aggressive and underdamped example for teaching failure modes.",
    ),
}


def get_controller_preset(name: str) -> ControllerPreset:
    try:
        return CONTROLLER_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(CONTROLLER_PRESETS))
        raise ValueError(f"unknown controller preset {name!r}; choose one of: {choices}") from exc


class TrotGaitController:
    """Readable open-loop trot target generator for a 12-joint quadruped."""

    def __init__(self, config: GaitConfig | None = None) -> None:
        self.config = config or GaitConfig()
        self._phase_offsets = {
            "FL": 0.0,
            "RR": 0.0,
            "FR": 0.5,
            "RL": 0.5,
        }

    @property
    def joint_names(self) -> tuple[str, ...]:
        return UNITREE_GO2_JOINTS

    def phase(self, t: float, leg: str) -> float:
        if leg not in self._phase_offsets:
            raise KeyError(f"unknown leg: {leg}")
        raw = t * self.config.stride_frequency + self._phase_offsets[leg]
        return raw % 1.0

    def gait_phase(self, t: float) -> dict[str, float]:
        return {leg: self.phase(t, leg) for leg in LEGS}

    def expected_contacts(self, t: float) -> dict[str, bool]:
        return {
            leg: self.phase(t, leg) < self.config.duty_factor
            for leg in LEGS
        }

    def joint_targets(self, t: float) -> dict[str, float]:
        targets: dict[str, float] = {}
        for leg in LEGS:
            phase = self.phase(t, leg)
            swing = _swing_progress(phase, self.config.duty_factor)
            stance = 1.0 - swing
            side = -1.0 if leg.endswith("R") else 1.0

            hip = self.config.nominal_hip + side * self.config.hip_swing * sin(2.0 * pi * phase)
            thigh = self.config.nominal_thigh + self.config.thigh_swing * (0.55 * swing - 0.22 * stance)
            calf = self.config.nominal_calf - self.config.calf_swing * swing + 0.16 * stance * cos(2.0 * pi * phase)

            targets[f"{leg}_hip_joint"] = hip
            targets[f"{leg}_thigh_joint"] = thigh
            targets[f"{leg}_calf_joint"] = calf

        return targets


def compute_pd_torque(
    current_angle: float,
    current_velocity: float,
    target_angle: float,
    target_velocity: float = 0.0,
    config: PDConfig | None = None,
) -> float:
    pd = config or PDConfig()
    torque = pd.kp * (target_angle - current_angle) + pd.kd * (
        target_velocity - current_velocity
    )
    return max(-pd.torque_limit, min(pd.torque_limit, torque))


def _swing_progress(phase: float, duty_factor: float) -> float:
    if phase < duty_factor:
        return 0.0
    swing_phase = (phase - duty_factor) / (1.0 - duty_factor)
    return 0.5 - 0.5 * cos(2.0 * pi * swing_phase)
