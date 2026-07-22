from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .replay_bank import load_trace_rows


def audit_detector_trace_rows(
    rows: Sequence[dict[str, Any]],
    *,
    false_positive_threshold: float = 0.05,
    recall_threshold: float = 0.90,
    latency_threshold_s: float = 0.12,
) -> dict[str, Any]:
    inactive_rows = [row for row in rows if not bool(row.get("external_force_active", False))]
    active_rows = [row for row in rows if bool(row.get("external_force_active", False))]
    inactive_detector_rows = [
        row for row in inactive_rows if bool(row.get("force_safety_detector_active", False))
    ]
    active_detector_rows = [
        row for row in active_rows if bool(row.get("force_safety_detector_active", False))
    ]
    latencies = _detection_latencies(rows)

    false_positive_rate = len(inactive_detector_rows) / len(inactive_rows) if inactive_rows else 0.0
    recall = len(active_detector_rows) / len(active_rows) if active_rows else 1.0
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    max_latency = max(latencies) if latencies else None

    return {
        "row_count": len(rows),
        "event_count": len(latencies),
        "no_force_rows": len(inactive_rows),
        "force_active_rows": len(active_rows),
        "no_force_false_positive_rate": false_positive_rate,
        "force_active_recall": recall,
        "mean_detection_latency_s": mean_latency,
        "max_detection_latency_s": max_latency,
        "passes_false_positive_rate": false_positive_rate < false_positive_threshold,
        "passes_force_active_recall": recall >= recall_threshold,
        "passes_detection_latency": (
            True if max_latency is None else max_latency <= latency_threshold_s
        ),
        "thresholds": {
            "false_positive_rate": false_positive_threshold,
            "force_active_recall": recall_threshold,
            "detection_latency_s": latency_threshold_s,
        },
    }


def _detection_latencies(rows: Sequence[dict[str, Any]]) -> list[float]:
    latencies: list[float] = []
    previous_active = False
    pending_start_s: float | None = None

    for row in rows:
        active = bool(row.get("external_force_active", False))
        detector_active = bool(row.get("force_safety_detector_active", False))
        t_s = float(row.get("t", 0.0))
        if active and not previous_active:
            pending_start_s = t_s
        if active and pending_start_s is not None and detector_active:
            latencies.append(max(0.0, t_s - pending_start_s))
            pending_start_s = None
        if not active:
            pending_start_s = None
        previous_active = active
    return latencies


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit deployable force-event detector traces.")
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.05)
    parser.add_argument("--min-force-active-recall", type=float, default=0.90)
    parser.add_argument("--max-detection-latency", type=float, default=0.12)
    args = parser.parse_args(argv)

    report = audit_detector_trace_rows(
        load_trace_rows(args.trace_jsonl),
        false_positive_threshold=args.max_false_positive_rate,
        recall_threshold=args.min_force_active_recall,
        latency_threshold_s=args.max_detection_latency,
    )
    _write_json(args.output, report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
