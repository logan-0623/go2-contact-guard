from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .trajectory_analysis import (
    analyze_trajectory,
    format_validation_report,
    load_trajectory,
    validate_trajectory,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Go2 Mini Lab trajectory JSON file."
    )
    parser.add_argument("trajectory", type=Path, help="Path to trajectory.json")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional path for writing metrics JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print validation result as JSON instead of text.",
    )
    args = parser.parse_args(argv)

    trajectory = load_trajectory(args.trajectory)
    result = validate_trajectory(trajectory)

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(analyze_trajectory(trajectory), indent=2),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_validation_report(result, args.trajectory))

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
