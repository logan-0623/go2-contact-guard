from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .controller import GaitConfig, LEGS, TrotGaitController


SCHEMA = "go2-mini-lab.trajectory.v1"


@dataclass(frozen=True)
class BaseState:
    position: list[float]
    quaternion: list[float]
    linear_velocity: list[float]
    angular_velocity: list[float]


def make_trajectory(
    *,
    frames: list[dict[str, Any]],
    joint_order: list[str],
    dt: float,
    source: str,
    robot: str = "unitree-go2-compatible",
    notes: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration = frames[-1]["t"] if frames else 0.0
    metadata: dict[str, Any] = {
        "robot": robot,
        "source": source,
        "duration": duration,
        "dt": dt,
        "joint_order": joint_order,
        "feet": list(LEGS),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if notes:
        metadata["notes"] = notes

    return {
        "schema": SCHEMA,
        "metadata": metadata,
        "frames": frames,
    }


def make_kinematic_demo(
    duration: float = 6.0,
    dt: float = 1.0 / 60.0,
    gait_config: GaitConfig | None = None,
    preset: str = "slow_trot",
) -> dict[str, Any]:
    """Generate a physics-free trajectory so the web replay works immediately."""

    controller = TrotGaitController(gait_config)
    frames: list[dict[str, Any]] = []
    speed = 0.22
    steps = max(1, int(duration / dt))

    previous_targets = controller.joint_targets(0.0)
    for i in range(steps + 1):
        t = round(i * dt, 6)
        targets = controller.joint_targets(t)
        qvel = []
        for name in controller.joint_names:
            qvel.append((targets[name] - previous_targets[name]) / dt if i else 0.0)

        base = BaseState(
            position=[round(speed * t, 6), 0.0, 0.36 + 0.012 * _soft_sin(t * 1.55)],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            linear_velocity=[speed, 0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )

        frames.append(
            {
                "t": t,
                "qpos": [*base.position, *base.quaternion, *[targets[n] for n in controller.joint_names]],
                "qvel": [*base.linear_velocity, *base.angular_velocity, *qvel],
                "ctrl": [0.0 for _ in controller.joint_names],
                "base": asdict(base),
                "joints": {name: round(targets[name], 6) for name in controller.joint_names},
                "contacts": controller.expected_contacts(t),
                "gait_phase": {
                    leg: round(phase, 6)
                    for leg, phase in controller.gait_phase(t).items()
                },
            }
        )
        previous_targets = targets

    return make_trajectory(
        frames=frames,
        joint_order=list(controller.joint_names),
        dt=dt,
        source="kinematic-demo",
        extra_metadata={"preset": preset, "mode": "kinematic-replay"},
        notes="Generated without physics. Use --model to export a MuJoCo rollout.",
    )


def _soft_sin(x: float) -> float:
    # A tiny deterministic bob for the browser sample; not a physics signal.
    import math

    return math.sin(2.0 * math.pi * x)
