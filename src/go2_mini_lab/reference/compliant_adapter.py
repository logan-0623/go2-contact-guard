from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COMPLIANT_ADAPTER_MODES = ("none", "horizontal_only", "horizontal_yaw", "full_safe")


@dataclass(frozen=True)
class BoundedCompliantAdapterConfig:
    mode: str = "horizontal_only"
    force_to_offset: float = 0.01
    max_horizontal_offset: float = 0.05
    max_vertical_offset: float = 0.0
    max_offset_velocity: float = 0.20
    decay: float = 0.95
    allow_negative_z_offset: bool = False


@dataclass(frozen=True)
class BoundedCompliantAdapterSample:
    position: np.ndarray
    offset: np.ndarray
    offset_velocity: np.ndarray
    target_offset: np.ndarray


class BoundedCompliantAdapter:
    def __init__(self, config: BoundedCompliantAdapterConfig | None = None) -> None:
        self.config = config or BoundedCompliantAdapterConfig()
        if self.config.mode not in COMPLIANT_ADAPTER_MODES:
            choices = ", ".join(COMPLIANT_ADAPTER_MODES)
            raise ValueError(f"unknown compliant adapter mode {self.config.mode!r}; choose one of: {choices}")
        self._offset = np.zeros(3, dtype=np.float32)
        self._offset_velocity = np.zeros(3, dtype=np.float32)

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    @property
    def offset_velocity(self) -> np.ndarray:
        return self._offset_velocity.copy()

    def reset(self) -> None:
        self._offset[:] = 0.0
        self._offset_velocity[:] = 0.0

    def update(
        self,
        *,
        nominal_position: Any,
        external_force: Any,
        dt: float,
    ) -> BoundedCompliantAdapterSample:
        nominal = np.asarray(nominal_position, dtype=np.float32).reshape(3)
        dt = max(1e-6, float(dt))
        if self.config.mode == "none":
            self.reset()
            return BoundedCompliantAdapterSample(
                position=nominal.copy(),
                offset=self._offset.copy(),
                offset_velocity=self._offset_velocity.copy(),
                target_offset=np.zeros(3, dtype=np.float32),
            )

        force = np.asarray(external_force, dtype=np.float32).reshape(3)
        target_offset = self._target_offset(force)
        if float(np.linalg.norm(force)) <= 1e-8:
            target_offset = self._offset * max(0.0, min(1.0, float(self.config.decay)))

        desired_delta = target_offset - self._offset
        max_step = max(0.0, float(self.config.max_offset_velocity)) * dt
        if max_step > 0.0:
            desired_delta = _clip_norm(desired_delta, max_step)
        next_offset = self._offset + desired_delta
        next_offset = self._clip_offset(next_offset)
        self._offset_velocity = (next_offset - self._offset) / dt
        self._offset = next_offset.astype(np.float32)
        return BoundedCompliantAdapterSample(
            position=nominal + self._offset,
            offset=self._offset.copy(),
            offset_velocity=self._offset_velocity.copy(),
            target_offset=target_offset.astype(np.float32),
        )

    def _target_offset(self, force: np.ndarray) -> np.ndarray:
        target = float(self.config.force_to_offset) * force.astype(np.float32)
        if self.config.mode in {"horizontal_only", "horizontal_yaw"}:
            target[2] = 0.0
        return self._clip_offset(target)

    def _clip_offset(self, offset: np.ndarray) -> np.ndarray:
        clipped = np.asarray(offset, dtype=np.float32).copy()
        max_xy = max(0.0, float(self.config.max_horizontal_offset))
        if max_xy > 0.0:
            clipped[:2] = _clip_norm(clipped[:2], max_xy)
        else:
            clipped[:2] = 0.0

        max_z = max(0.0, float(self.config.max_vertical_offset))
        if max_z <= 0.0:
            clipped[2] = 0.0
        else:
            lower = -max_z if self.config.allow_negative_z_offset else 0.0
            clipped[2] = float(np.clip(clipped[2], lower, max_z))
        return clipped


def _clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
    limit = max(0.0, float(limit))
    norm = float(np.linalg.norm(value))
    if limit <= 0.0 or norm <= limit or norm <= 1e-8:
        return value.astype(np.float32)
    return (value * (limit / norm)).astype(np.float32)

