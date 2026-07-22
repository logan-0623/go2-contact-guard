from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .controller import CONTROLLER_PRESETS, get_controller_preset
from .mujoco_rollout import run_mujoco_rollout
from .trajectory import make_kinematic_demo
from .trajectory_analysis import metrics_output_path, validate_trajectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a Go2 Mini Lab trajectory JSON file."
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to a MuJoCo MJCF model. If omitted, writes a kinematic browser demo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trajectory.json"),
        help="Output trajectory JSON path.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Rollout duration in seconds.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 60.0,
        help="Export timestep in seconds.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help=(
            "MuJoCo-only warmup seconds before recording frames. "
            "Use 0 to export from the initial pose."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(CONTROLLER_PRESETS),
        default="slow_trot",
        help="Controller preset for gait and PD parameters.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Metrics JSON path. Defaults to trajectory_metrics.json next to --output.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not write rollout metrics JSON.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help=(
            "Keep replay fields but omit raw qpos/qvel/ctrl arrays. "
            "Useful when the JSON is only loaded in the browser dashboard."
        ),
    )
    args = parser.parse_args(argv)
    preset = get_controller_preset(args.preset)

    if args.model:
        trajectory = run_mujoco_rollout(
            model_path=args.model,
            duration=args.duration,
            export_dt=args.dt,
            warmup_duration=args.warmup,
            gait_config=preset.gait,
            pd_config=preset.pd,
            preset=args.preset,
        )
    else:
        trajectory = make_kinematic_demo(
            duration=args.duration,
            dt=args.dt,
            gait_config=preset.gait,
            preset=args.preset,
        )

    if args.compact_json:
        _strip_raw_state_arrays(trajectory)

    result = validate_trajectory(trajectory)
    metrics = result["metrics"]
    trajectory.setdefault("metadata", {})["metrics"] = metrics
    trajectory["metadata"]["validation"] = {
        "passed": result["passed"],
        "health_status": metrics["health_status"],
        "warnings": result["warnings"],
        "errors": result["errors"],
        "warning_events": metrics["warning_events"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(trajectory['frames'])} frames)")

    if result["warnings"]:
        print(f"validation warnings: {len(result['warnings'])}")
        for warning in result["warnings"][:5]:
            print(f"- {warning}")
    if result["errors"]:
        print(f"validation errors: {len(result['errors'])}")
        for error in result["errors"][:5]:
            print(f"- {error}")

    if not args.no_metrics:
        metrics_path = args.metrics_output or metrics_output_path(args.output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {metrics_path}")

    return 0


def _strip_raw_state_arrays(trajectory: dict) -> None:
    trajectory.setdefault("metadata", {})["raw_state_included"] = False
    trajectory["metadata"]["raw_state_note"] = (
        "qpos, qvel, and ctrl were omitted with --compact-json. "
        "Use base, joints, contacts, and gait_phase for replay."
    )
    for frame in trajectory.get("frames", []):
        frame["qpos"] = []
        frame["qvel"] = []
        frame["ctrl"] = []


if __name__ == "__main__":
    raise SystemExit(main())
