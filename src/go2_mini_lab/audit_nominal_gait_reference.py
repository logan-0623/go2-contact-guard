from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .reference.reference_audit import (
    ReferenceAuditThresholds,
    audit_nominal_gait_file,
    write_audit_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a phase-conditioned nominal Go2 gait reference.")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("reference_datasets/go2_nominal_gait_040_reference.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/go2_nominal_reference_audit.json"),
    )
    parser.add_argument("--min-mean-vx", type=float, default=0.35)
    parser.add_argument("--max-mean-vx", type=float, default=0.45)
    parser.add_argument("--max-mean-abs-vy", type=float, default=0.08)
    parser.add_argument("--max-mean-abs-yaw-rate", type=float, default=0.12)
    parser.add_argument("--min-base-z", type=float, default=0.26)
    parser.add_argument("--max-joint-abs", type=float, default=3.2)
    parser.add_argument("--max-action-rate-mean", type=float, default=4.0)
    parser.add_argument("--min-foot-clearance", type=float, default=0.015)
    parser.add_argument("--min-contact-duty", type=float, default=0.10)
    parser.add_argument("--max-contact-duty", type=float, default=0.90)
    parser.add_argument("--max-left-right-contact-delta", type=float, default=0.35)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Return success even if the audit fails.",
    )
    args = parser.parse_args(argv)

    thresholds = ReferenceAuditThresholds(
        min_mean_vx=args.min_mean_vx,
        max_mean_vx=args.max_mean_vx,
        max_mean_abs_vy=args.max_mean_abs_vy,
        max_mean_abs_yaw_rate=args.max_mean_abs_yaw_rate,
        min_base_z=args.min_base_z,
        max_joint_abs=args.max_joint_abs,
        max_action_rate_mean=args.max_action_rate_mean,
        min_foot_clearance=args.min_foot_clearance,
        min_contact_duty=args.min_contact_duty,
        max_contact_duty=args.max_contact_duty,
        max_left_right_contact_delta=args.max_left_right_contact_delta,
    )
    report = audit_nominal_gait_file(args.dataset, thresholds=thresholds)
    write_audit_report(args.output, report)
    print(f"dataset: {args.dataset}")
    print(f"audit: {'PASS' if report.passed else 'FAIL'}")
    for name, value in sorted(report.metrics.items()):
        print(f"{name}: {value:.6g}")
    for name, reason in report.failures.items():
        print(f"FAIL {name}: {reason}")
    print(f"wrote {args.output}")
    return 0 if report.passed or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

