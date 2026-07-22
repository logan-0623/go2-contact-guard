from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_EPS = 1e-9


@dataclass
class _ForceEvent:
    index: int
    start_s: float | None
    end_s: float | None
    active_steps: int = 0
    compliant_steps: int = 0
    onset_active_steps: int = 0
    onset_compliant_steps: int = 0
    adaptation_active_steps: int = 0
    adaptation_compliant_steps: int = 0
    max_force_n: float = 0.0
    max_excess_n: float = 0.0
    sum_excess_n: float = 0.0
    excess_impulse_ns: float = 0.0
    onset_max_excess_n: float = 0.0
    onset_excess_impulse_ns: float = 0.0
    adaptation_max_excess_n: float = 0.0
    adaptation_excess_impulse_ns: float = 0.0
    time_to_first_excess_s: float | None = None
    time_to_recover_s: float | None = None
    elapsed_values_s: list[float] = field(default_factory=list)

    def observe(
        self,
        *,
        elapsed_s: float,
        force_n: float,
        excess_n: float,
        compliant: bool,
        dt_s: float,
        onset_window_s: float,
    ) -> None:
        self.active_steps += 1
        if compliant:
            self.compliant_steps += 1
        self.max_force_n = max(self.max_force_n, force_n)
        self.max_excess_n = max(self.max_excess_n, excess_n)
        self.sum_excess_n += excess_n
        self.excess_impulse_ns += excess_n * dt_s
        self.elapsed_values_s.append(elapsed_s)
        if elapsed_s <= onset_window_s + _EPS:
            self.onset_active_steps += 1
            if compliant:
                self.onset_compliant_steps += 1
            self.onset_max_excess_n = max(self.onset_max_excess_n, excess_n)
            self.onset_excess_impulse_ns += excess_n * dt_s
        else:
            self.adaptation_active_steps += 1
            if compliant:
                self.adaptation_compliant_steps += 1
            self.adaptation_max_excess_n = max(self.adaptation_max_excess_n, excess_n)
            self.adaptation_excess_impulse_ns += excess_n * dt_s
        if excess_n > _EPS and self.time_to_first_excess_s is None:
            self.time_to_first_excess_s = elapsed_s
        elif (
            excess_n <= _EPS
            and self.time_to_first_excess_s is not None
            and self.time_to_recover_s is None
            and elapsed_s >= self.time_to_first_excess_s
        ):
            self.time_to_recover_s = elapsed_s

    @property
    def compliance_rate(self) -> float:
        return self.compliant_steps / self.active_steps if self.active_steps else 1.0

    @property
    def onset_compliance_rate(self) -> float:
        return self.onset_compliant_steps / self.onset_active_steps if self.onset_active_steps else 1.0

    @property
    def adaptation_compliance_rate(self) -> float:
        return (
            self.adaptation_compliant_steps / self.adaptation_active_steps
            if self.adaptation_active_steps
            else 1.0
        )

    @property
    def compliant(self) -> bool:
        return self.max_excess_n <= _EPS

    @property
    def adaptation_compliant(self) -> bool:
        return self.adaptation_max_excess_n <= _EPS

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "event_index": self.index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "active_steps": self.active_steps,
            "compliant_steps": self.compliant_steps,
            "compliance_rate": self.compliance_rate,
            "compliant": self.compliant,
            "onset_active_steps": self.onset_active_steps,
            "onset_compliant_steps": self.onset_compliant_steps,
            "onset_compliance_rate": self.onset_compliance_rate,
            "adaptation_active_steps": self.adaptation_active_steps,
            "adaptation_compliant_steps": self.adaptation_compliant_steps,
            "adaptation_compliance_rate": self.adaptation_compliance_rate,
            "adaptation_compliant": self.adaptation_compliant,
            "max_force_n": self.max_force_n,
            "max_excess_n": self.max_excess_n,
            "mean_excess_n": self.sum_excess_n / self.active_steps if self.active_steps else 0.0,
            "excess_impulse_ns": self.excess_impulse_ns,
            "onset_max_excess_n": self.onset_max_excess_n,
            "onset_excess_impulse_ns": self.onset_excess_impulse_ns,
            "adaptation_max_excess_n": self.adaptation_max_excess_n,
            "adaptation_excess_impulse_ns": self.adaptation_excess_impulse_ns,
            "time_to_first_excess_s": self.time_to_first_excess_s,
            "time_to_recover_s": self.time_to_recover_s,
        }


class ForceComplianceTracker:
    def __init__(self, *, dt_s: float, onset_window_s: float) -> None:
        self.dt_s = max(0.0, float(dt_s))
        self.onset_window_s = max(0.0, float(onset_window_s))
        self._events: list[_ForceEvent] = []
        self._events_by_key: dict[tuple[float | None, float | None] | int, _ForceEvent] = {}
        self._fallback_active_key: int | None = None
        self._last_active_key: tuple[float | None, float | None] | int | None = None

    def observe(self, info: dict[str, Any]) -> None:
        if not bool(info.get("external_force_active", False)):
            self._fallback_active_key = None
            self._last_active_key = None
            return

        has_current_window_fields = (
            "external_force_current_window_start_s" in info
            or "external_force_current_window_end_s" in info
        )
        if has_current_window_fields:
            start_s = _optional_float(info.get("external_force_current_window_start_s"))
            end_s = _optional_float(info.get("external_force_current_window_end_s"))
        else:
            start_s = _optional_float(info.get("external_force_start_s"))
            end_s = _optional_float(info.get("external_force_end_s"))
        if start_s is None and end_s is None:
            if self._last_active_key is not None:
                key = self._last_active_key
            elif self._fallback_active_key is None:
                self._fallback_active_key = len(self._events) + 1
                key = self._fallback_active_key
            else:
                key = self._fallback_active_key
        else:
            key = (start_s, end_s)
        event = self._events_by_key.get(key)
        if event is None:
            event = _ForceEvent(index=len(self._events), start_s=start_s, end_s=end_s)
            self._events_by_key[key] = event
            self._events.append(event)
        self._last_active_key = key

        elapsed_s = _optional_float(info.get("external_force_current_window_elapsed_s"))
        if elapsed_s is None:
            t_s = _optional_float(info.get("t"))
            elapsed_s = t_s - start_s if t_s is not None and start_s is not None else event.active_steps * self.dt_s
        force_n = max(0.0, float(info.get("external_force_magnitude", 0.0) or 0.0))
        excess_n = max(0.0, float(info.get("external_force_excess_n", 0.0) or 0.0))
        compliant_value = info.get("external_force_step_compliant")
        compliant = bool(compliant_value) if compliant_value is not None else excess_n <= _EPS
        event.observe(
            elapsed_s=max(0.0, float(elapsed_s)),
            force_n=force_n,
            excess_n=excess_n,
            compliant=compliant,
            dt_s=self.dt_s,
            onset_window_s=self.onset_window_s,
        )

    def summary(self) -> dict[str, Any]:
        event_dicts = [event.as_dict() for event in self._events]
        count = len(event_dicts)
        if count == 0:
            return {
                "external_force_event_count_v2": 0,
                "external_force_events_v2": [],
                "external_force_event_compliance_rate": 1.0,
                "episode_all_events_compliant": True,
                "external_force_worst_event_max_excess_n": 0.0,
                "external_force_worst_event_onset_max_excess_n": 0.0,
                "external_force_worst_event_adaptation_max_excess_n": 0.0,
                "external_force_worst_event_excess_impulse_ns": 0.0,
                "external_force_worst_event_onset_excess_impulse_ns": 0.0,
                "external_force_worst_event_adaptation_excess_impulse_ns": 0.0,
                "external_force_mean_event_excess_impulse_ns": 0.0,
                "external_force_mean_event_adaptation_compliance_rate": 1.0,
                "external_force_min_time_to_first_excess_s": None,
                "external_force_max_time_to_recover_s": None,
            }
        return {
            "external_force_event_count_v2": count,
            "external_force_events_v2": event_dicts,
            "external_force_event_compliance_rate": sum(
                bool(event["compliant"]) for event in event_dicts
            )
            / count,
            "episode_all_events_compliant": all(bool(event["compliant"]) for event in event_dicts),
            "external_force_worst_event_max_excess_n": max(
                float(event["max_excess_n"]) for event in event_dicts
            ),
            "external_force_worst_event_onset_max_excess_n": max(
                float(event["onset_max_excess_n"]) for event in event_dicts
            ),
            "external_force_worst_event_adaptation_max_excess_n": max(
                float(event["adaptation_max_excess_n"]) for event in event_dicts
            ),
            "external_force_worst_event_excess_impulse_ns": max(
                float(event["excess_impulse_ns"]) for event in event_dicts
            ),
            "external_force_worst_event_onset_excess_impulse_ns": max(
                float(event["onset_excess_impulse_ns"]) for event in event_dicts
            ),
            "external_force_worst_event_adaptation_excess_impulse_ns": max(
                float(event["adaptation_excess_impulse_ns"]) for event in event_dicts
            ),
            "external_force_mean_event_excess_impulse_ns": _mean(
                event["excess_impulse_ns"] for event in event_dicts
            ),
            "external_force_mean_event_adaptation_compliance_rate": _mean(
                event["adaptation_compliance_rate"] for event in event_dicts
            ),
            "external_force_min_time_to_first_excess_s": _min_or_none(
                event["time_to_first_excess_s"] for event in event_dicts
            ),
            "external_force_max_time_to_recover_s": _max_or_none(
                event["time_to_recover_s"] for event in event_dicts
            ),
        }


def aggregate_force_compliance_v2(episodes: list[dict[str, Any]]) -> dict[str, float | int | None]:
    count = max(len(episodes), 1)
    return {
        "mean_external_force_event_count_v2": _mean(
            episode.get("external_force_event_count_v2") for episode in episodes
        ),
        "mean_external_force_event_compliance_rate": _mean(
            episode.get("external_force_event_compliance_rate", 1.0) for episode in episodes
        ),
        "episode_all_events_compliance_rate": sum(
            bool(episode.get("episode_all_events_compliant", True))
            for episode in episodes
        )
        / count,
        "max_external_force_event_max_excess_n": _max_or_default(
            episode.get("external_force_worst_event_max_excess_n", 0.0)
            for episode in episodes
        ),
        "max_external_force_event_onset_max_excess_n": _max_or_default(
            episode.get("external_force_worst_event_onset_max_excess_n", 0.0)
            for episode in episodes
        ),
        "max_external_force_event_adaptation_max_excess_n": _max_or_default(
            episode.get("external_force_worst_event_adaptation_max_excess_n", 0.0)
            for episode in episodes
        ),
        "max_external_force_event_excess_impulse_ns": _max_or_default(
            episode.get("external_force_worst_event_excess_impulse_ns", 0.0)
            for episode in episodes
        ),
        "max_external_force_event_onset_excess_impulse_ns": _max_or_default(
            episode.get("external_force_worst_event_onset_excess_impulse_ns", 0.0)
            for episode in episodes
        ),
        "max_external_force_event_adaptation_excess_impulse_ns": _max_or_default(
            episode.get("external_force_worst_event_adaptation_excess_impulse_ns", 0.0)
            for episode in episodes
        ),
        "mean_external_force_event_excess_impulse_ns": _mean(
            episode.get("external_force_mean_event_excess_impulse_ns", 0.0)
            for episode in episodes
        ),
        "mean_external_force_event_adaptation_compliance_rate": _mean(
            episode.get("external_force_mean_event_adaptation_compliance_rate", 1.0)
            for episode in episodes
        ),
        "min_external_force_time_to_first_excess_s": _min_or_none(
            episode.get("external_force_min_time_to_first_excess_s")
            for episode in episodes
        ),
        "max_external_force_time_to_recover_s": _max_or_none(
            episode.get("external_force_max_time_to_recover_s")
            for episode in episodes
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean(values: Any) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _min_or_none(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def _max_or_none(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def _max_or_default(values: Any, *, default: float = 0.0) -> float:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else default
