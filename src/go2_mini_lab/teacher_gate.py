from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .gates import GENTLE_30N_GATE


SCENARIOS = ("no_force", "lateral_plus_90", "lateral_minus_90", "random_3d")
HORIZONTAL_SCENARIOS = ("no_force", "lateral_plus_90", "lateral_minus_90", "random_horizontal")
ALL_SCENARIOS = (
    "no_force",
    "lateral_plus_90",
    "lateral_minus_90",
    "random_horizontal",
    "random_3d",
)
DEFAULT_GATE_SPEC = GENTLE_30N_GATE


@dataclass(frozen=True)
class GateCheck:
    scenario: str
    metric: str
    expected: str
    value: float | None
    passed: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check disturbance diagnostics before a teacher is allowed to "
            "produce student distillation data."
        )
    )
    parser.add_argument(
        "report_dir",
        type=Path,
        help=(
            "Directory containing no_force.json, lateral_plus_90.json, "
            "lateral_minus_90.json, and random_3d.json."
        ),
    )
    parser.add_argument("--min-no-force-success", type=float, default=DEFAULT_GATE_SPEC.min_no_force_success)
    parser.add_argument("--min-no-force-straightness", type=float, default=DEFAULT_GATE_SPEC.min_no_force_straightness)
    parser.add_argument(
        "--max-no-force-forward-velocity",
        type=float,
        default=DEFAULT_GATE_SPEC.max_no_force_forward_velocity,
    )
    parser.add_argument(
        "--max-no-force-abs-lateral-velocity",
        type=float,
        default=DEFAULT_GATE_SPEC.max_no_force_abs_lateral_velocity,
    )
    parser.add_argument(
        "--max-no-force-abs-yaw-rate",
        type=float,
        default=DEFAULT_GATE_SPEC.max_no_force_abs_yaw_rate,
    )
    parser.add_argument(
        "--max-no-force-abs-lateral-displacement",
        type=float,
        default=DEFAULT_GATE_SPEC.max_no_force_abs_lateral_displacement,
    )
    parser.add_argument("--min-force-success", type=float, default=DEFAULT_GATE_SPEC.min_force_success)
    parser.add_argument("--max-force-fall-rate", type=float, default=DEFAULT_GATE_SPEC.max_force_fall_rate)
    parser.add_argument(
        "--min-force-recovery-success",
        type=float,
        default=DEFAULT_GATE_SPEC.min_force_recovery_success,
    )
    parser.add_argument(
        "--recovery-first",
        action="store_true",
        help=(
            "Gate force scenarios by recovery behavior instead of strict transient force compliance. "
            "Compliance metrics remain diagnostic unless explicit compliance thresholds are passed."
        ),
    )
    parser.add_argument("--min-force-recovery-survival", type=float, default=0.9)
    parser.add_argument("--min-force-recovery-posture", type=float, default=0.9)
    parser.add_argument("--min-force-recovery-locomotion", type=float, default=0.9)
    parser.add_argument("--min-force-min-base-z-after", type=float, default=0.26)
    parser.add_argument("--max-force-orientation-error", type=float, default=0.65)
    parser.add_argument("--min-post-force-forward", type=float, default=DEFAULT_GATE_SPEC.min_post_force_forward)
    parser.add_argument(
        "--max-post-force-abs-lateral",
        type=float,
        default=DEFAULT_GATE_SPEC.max_post_force_abs_lateral,
    )
    parser.add_argument("--max-post-force-abs-yaw", type=float, default=DEFAULT_GATE_SPEC.max_post_force_abs_yaw)
    parser.add_argument(
        "--max-force-abs-lateral-displacement",
        type=float,
        default=DEFAULT_GATE_SPEC.max_force_abs_lateral_displacement,
    )
    parser.add_argument("--min-force-step-compliance", type=float)
    parser.add_argument("--min-episode-compliance", type=float)
    parser.add_argument(
        "--skip-force-scenarios",
        action="store_true",
        help="Only check no_force.json. Use for no-force warmup stages before disturbance training.",
    )
    parser.add_argument(
        "--horizontal-force-only",
        action="store_true",
        help="Check no_force plus horizontal 5N scenarios; random_3d remains diagnostic-only.",
    )
    parser.add_argument(
        "--all-force-scenarios",
        action="store_true",
        help="Check both random-horizontal and random-3D scenarios in addition to fixed lateral pushes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the gate report. Defaults to <report_dir>/gate.json.",
    )
    args = parser.parse_args(argv)

    if args.skip_force_scenarios:
        scenarios = ("no_force",)
    elif args.all_force_scenarios:
        scenarios = ALL_SCENARIOS
    elif args.horizontal_force_only:
        scenarios = HORIZONTAL_SCENARIOS
    else:
        scenarios = SCENARIOS
    reports = {
        scenario: _load_report(args.report_dir / f"{scenario}.json")
        for scenario in scenarios
    }
    checks = _build_checks(reports, args)
    passed = all(check.passed for check in checks)
    output = {
        "passed": passed,
        "report_dir": str(args.report_dir),
        "checks": [
            {
                "scenario": check.scenario,
                "metric": check.metric,
                "expected": check.expected,
                "value": check.value,
                "passed": check.passed,
            }
            for check in checks
        ],
    }
    output_path = args.output or args.report_dir / "gate.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _print_summary(checks=checks, output_path=output_path, passed=passed)
    return 0 if passed else 1


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing diagnostics report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"diagnostics report has no aggregate object: {path}")
    return aggregate


def _build_checks(reports: dict[str, dict[str, Any]], args: Any) -> list[GateCheck]:
    checks: list[GateCheck] = []
    no_force = reports["no_force"]
    checks.extend(
        [
            _min_check("no_force", no_force, "success_rate", args.min_no_force_success),
            _min_check("no_force", no_force, "straightness_success_rate", args.min_no_force_straightness),
            _max_check(
                "no_force",
                no_force,
                "mean_forward_velocity",
                args.max_no_force_forward_velocity,
            ),
            _max_check(
                "no_force",
                no_force,
                "mean_abs_lateral_velocity",
                args.max_no_force_abs_lateral_velocity,
            ),
            _max_check("no_force", no_force, "mean_abs_yaw_rate", args.max_no_force_abs_yaw_rate),
            _max_check(
                "no_force",
                no_force,
                "mean_abs_final_lateral_displacement",
                args.max_no_force_abs_lateral_displacement,
            ),
        ]
    )

    if getattr(args, "skip_force_scenarios", False):
        return checks

    if getattr(args, "all_force_scenarios", False):
        force_scenarios = ALL_SCENARIOS[1:]
    elif getattr(args, "horizontal_force_only", False):
        force_scenarios = HORIZONTAL_SCENARIOS[1:]
    else:
        force_scenarios = SCENARIOS[1:]
    for scenario in force_scenarios:
        aggregate = reports[scenario]
        if getattr(args, "recovery_first", False):
            scenario_checks = [
                _min_check(scenario, aggregate, "success_rate", args.min_force_success),
                _max_check(scenario, aggregate, "fall_rate", args.max_force_fall_rate),
                _min_check(
                    scenario,
                    aggregate,
                    "force_recovery_survival_rate",
                    args.min_force_recovery_survival,
                ),
                _min_check(
                    scenario,
                    aggregate,
                    "force_recovery_posture_success_rate",
                    args.min_force_recovery_posture,
                ),
                _min_check(
                    scenario,
                    aggregate,
                    "force_recovery_locomotion_success_rate",
                    args.min_force_recovery_locomotion,
                ),
                _min_check(
                    scenario,
                    aggregate,
                    "force_recovery_min_base_z_after_force",
                    args.min_force_min_base_z_after,
                ),
                _max_check(
                    scenario,
                    aggregate,
                    "force_recovery_max_orientation_error_during_force",
                    args.max_force_orientation_error,
                ),
                _min_check(scenario, aggregate, "post_force_mean_forward_velocity", args.min_post_force_forward),
                _max_check(
                    scenario,
                    aggregate,
                    "post_force_mean_abs_lateral_velocity",
                    args.max_post_force_abs_lateral,
                ),
                _max_check(scenario, aggregate, "post_force_mean_abs_yaw_rate", args.max_post_force_abs_yaw),
                _max_check(
                    scenario,
                    aggregate,
                    "mean_abs_final_lateral_displacement",
                    args.max_force_abs_lateral_displacement,
                ),
            ]
        else:
            scenario_checks = [
                _min_check(scenario, aggregate, "success_rate", args.min_force_success),
                _max_check(scenario, aggregate, "fall_rate", args.max_force_fall_rate),
                _min_check(
                    scenario,
                    aggregate,
                    "external_force_recovery_success_rate",
                    args.min_force_recovery_success,
                ),
                _min_check(scenario, aggregate, "post_force_mean_forward_velocity", args.min_post_force_forward),
                _max_check(
                    scenario,
                    aggregate,
                    "post_force_mean_abs_lateral_velocity",
                    args.max_post_force_abs_lateral,
                ),
                _max_check(scenario, aggregate, "post_force_mean_abs_yaw_rate", args.max_post_force_abs_yaw),
                _max_check(
                    scenario,
                    aggregate,
                    "mean_abs_final_lateral_displacement",
                    args.max_force_abs_lateral_displacement,
                ),
            ]
        if args.min_force_step_compliance is not None:
            scenario_checks.append(
                _min_check(
                    scenario,
                    aggregate,
                    "force_active_step_compliance_rate",
                    args.min_force_step_compliance,
                )
            )
        if args.min_episode_compliance is not None:
            scenario_checks.append(
                _min_check(
                    scenario,
                    aggregate,
                    "episode_compliance_rate",
                    args.min_episode_compliance,
                )
            )
        checks.extend(scenario_checks)
    return checks


def _metric_value(aggregate: dict[str, Any], metric: str) -> float | None:
    value = aggregate.get(metric)
    if value is None:
        return None
    return float(value)


def _min_check(scenario: str, aggregate: dict[str, Any], metric: str, threshold: float) -> GateCheck:
    value = _metric_value(aggregate, metric)
    return GateCheck(
        scenario=scenario,
        metric=metric,
        expected=f">= {threshold:g}",
        value=value,
        passed=value is not None and value >= threshold,
    )


def _max_check(scenario: str, aggregate: dict[str, Any], metric: str, threshold: float) -> GateCheck:
    value = _metric_value(aggregate, metric)
    return GateCheck(
        scenario=scenario,
        metric=metric,
        expected=f"<= {threshold:g}",
        value=value,
        passed=value is not None and value <= threshold,
    )


def _print_summary(*, checks: list[GateCheck], output_path: Path, passed: bool) -> None:
    print(f"teacher_gate: {'PASS' if passed else 'FAIL'}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        value = "None" if check.value is None else f"{check.value:.4g}"
        print(f"{status} {check.scenario}.{check.metric}: {value} {check.expected}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
