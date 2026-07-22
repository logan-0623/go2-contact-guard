from __future__ import annotations

from typing import Sequence


FEET = ("FR", "FL", "RR", "RL")
LEFT_FEET = ("FL", "RL")
RIGHT_FEET = ("FR", "RR")
FRONT_FEET = ("FR", "FL")
REAR_FEET = ("RR", "RL")
DIAGONAL_A = ("FR", "RL")
DIAGONAL_B = ("FL", "RR")

ACTION_BALANCE_KEYS = (
    "left_action_energy",
    "right_action_energy",
    "left_right_action_energy_delta",
    "front_action_energy",
    "rear_action_energy",
    "front_rear_action_energy_delta",
    "diagonal_action_energy_delta",
    "action_energy_imbalance_abs",
)

CONTACT_BALANCE_KEYS = (
    "left_contact_ratio",
    "right_contact_ratio",
    "left_right_contact_delta",
    "front_contact_ratio",
    "rear_contact_ratio",
    "front_rear_contact_delta",
    "diagonal_contact_delta",
)


def contact_balance_metrics(contact_ratio: dict[str, float]) -> dict[str, float]:
    left = _mean_named(contact_ratio, LEFT_FEET)
    right = _mean_named(contact_ratio, RIGHT_FEET)
    front = _mean_named(contact_ratio, FRONT_FEET)
    rear = _mean_named(contact_ratio, REAR_FEET)
    diagonal_a = _mean_named(contact_ratio, DIAGONAL_A)
    diagonal_b = _mean_named(contact_ratio, DIAGONAL_B)
    return {
        "left_contact_ratio": left,
        "right_contact_ratio": right,
        "left_right_contact_delta": left - right,
        "front_contact_ratio": front,
        "rear_contact_ratio": rear,
        "front_rear_contact_delta": front - rear,
        "diagonal_contact_delta": diagonal_a - diagonal_b,
    }


def action_balance_metrics(
    action: Sequence[float],
    action_names: Sequence[str],
) -> dict[str, float]:
    by_leg = _leg_energy(action, action_names)
    if not by_leg:
        return {}

    left = _mean_named(by_leg, LEFT_FEET)
    right = _mean_named(by_leg, RIGHT_FEET)
    front = _mean_named(by_leg, FRONT_FEET)
    rear = _mean_named(by_leg, REAR_FEET)
    diagonal_a = _mean_named(by_leg, DIAGONAL_A)
    diagonal_b = _mean_named(by_leg, DIAGONAL_B)
    left_right_delta = left - right
    front_rear_delta = front - rear
    diagonal_delta = diagonal_a - diagonal_b
    return {
        "left_action_energy": left,
        "right_action_energy": right,
        "left_right_action_energy_delta": left_right_delta,
        "front_action_energy": front,
        "rear_action_energy": rear,
        "front_rear_action_energy_delta": front_rear_delta,
        "diagonal_action_energy_delta": diagonal_delta,
        "action_energy_imbalance_abs": abs(left_right_delta),
    }


def _leg_energy(
    action: Sequence[float],
    action_names: Sequence[str],
) -> dict[str, float]:
    if len(action) != len(action_names):
        return {}

    totals = {foot: 0.0 for foot in FEET}
    counts = {foot: 0 for foot in FEET}
    for value, name in zip(action, action_names):
        foot = str(name).split("_", maxsplit=1)[0]
        if foot not in totals:
            continue
        totals[foot] += float(value) ** 2
        counts[foot] += 1

    return {
        foot: totals[foot] / counts[foot]
        for foot in FEET
        if counts[foot] > 0
    }


def _mean_named(values: dict[str, float], names: Sequence[str]) -> float:
    selected = [float(values[name]) for name in names if name in values]
    return sum(selected) / max(len(selected), 1)
