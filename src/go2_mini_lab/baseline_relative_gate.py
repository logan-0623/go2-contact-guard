from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCENARIOS = ("no_force", "lateral_plus_90", "lateral_minus_90", "random_3d")
HORIZONTAL_SCENARIOS = ("no_force", "lateral_plus_90", "lateral_minus_90", "random_horizontal")
ALL_SCENARIOS = (
    "no_force",
    "lateral_plus_90",
    "lateral_minus_90",
    "random_horizontal",
    "random_3d",
)
FORCE_SCENARIOS = tuple(scenario for scenario in ALL_SCENARIOS if scenario != "no_force")


@dataclass(frozen=True)
class RelativeRule:
    scenario: str
    metric: str
    direction: str
    tolerance: float
    kind: str = "non_regression"


@dataclass(frozen=True)
class RelativeCheck:
    kind: str
    scenario: str
    metric: str
    direction: str
    baseline: float | None
    candidate: float | None
    tolerance: float
    delta: float | None
    passed: bool
    skipped: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a candidate Go2 disturbance report against an ONNX baseline "
            "under the same evaluation contract."
        )
    )
    parser.add_argument("baseline_report_dir", type=Path)
    parser.add_argument("candidate_report_dir", type=Path)
    parser.add_argument(
        "--all-force-scenarios",
        action="store_true",
        help="Compare no_force, fixed lateral, random-horizontal, and random-3D scenarios.",
    )
    parser.add_argument(
        "--horizontal-force-only",
        action="store_true",
        help="Compare no_force plus horizontal scenarios; random_3d remains diagnostic.",
    )
    parser.add_argument(
        "--min-primary-improvements",
        type=int,
        default=1,
        help=(
            "Minimum number of force metrics that must improve over ONNX by their "
            "configured margin. Use 0 for pure non-regression."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to <candidate_report_dir>/baseline_relative_gate.json.",
    )
    args = parser.parse_args(argv)

    scenarios = _selected_scenarios(args)
    baseline = _load_reports(args.baseline_report_dir, scenarios)
    candidate = _load_reports(args.candidate_report_dir, scenarios)

    rules = _default_non_regression_rules(scenarios)
    checks = [_evaluate_rule(rule, baseline, candidate) for rule in rules]
    non_regression_passed = all(check.passed for check in checks if not check.skipped)

    primary_rules = _default_primary_improvement_rules(scenarios)
    primary_checks = [_evaluate_rule(rule, baseline, candidate) for rule in primary_rules]
    primary_improvements = [
        check
        for check in primary_checks
        if not check.skipped and check.passed
    ]
    primary_improvement_passed = len(primary_improvements) >= args.min_primary_improvements

    passed = non_regression_passed and primary_improvement_passed
    output = {
        "passed": passed,
        "baseline_report_dir": str(args.baseline_report_dir),
        "candidate_report_dir": str(args.candidate_report_dir),
        "scenarios": list(scenarios),
        "non_regression_passed": non_regression_passed,
        "primary_improvement_passed": primary_improvement_passed,
        "primary_improvement_count": len(primary_improvements),
        "min_primary_improvements": args.min_primary_improvements,
        "checks": [_check_to_dict(check) for check in checks],
        "primary_improvement_checks": [_check_to_dict(check) for check in primary_checks],
    }
    output_path = args.output or args.candidate_report_dir / "baseline_relative_gate.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _print_summary(
        checks=checks,
        primary_checks=primary_checks,
        output_path=output_path,
        passed=passed,
        min_primary_improvements=args.min_primary_improvements,
    )
    return 0 if passed else 1


def _selected_scenarios(args: Any) -> tuple[str, ...]:
    if getattr(args, "all_force_scenarios", False):
        return ALL_SCENARIOS
    if getattr(args, "horizontal_force_only", False):
        return HORIZONTAL_SCENARIOS
    return SCENARIOS


def _load_reports(report_dir: Path, scenarios: Sequence[str]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        path = report_dir / f"{scenario}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing diagnostics report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        aggregate = report.get("aggregate")
        if not isinstance(aggregate, dict):
            raise ValueError(f"diagnostics report has no aggregate object: {path}")
        reports[scenario] = aggregate
    return reports


def _default_non_regression_rules(scenarios: Sequence[str]) -> list[RelativeRule]:
    rules = [
        RelativeRule("no_force", "success_rate", "higher", 0.0),
        RelativeRule("no_force", "straightness_success_rate", "higher", 0.0),
        RelativeRule("no_force", "fall_rate", "lower", 0.0),
        RelativeRule("no_force", "mean_abs_lateral_velocity", "lower", 0.02),
        RelativeRule("no_force", "mean_abs_yaw_rate", "lower", 0.03),
        RelativeRule("no_force", "mean_abs_final_lateral_displacement", "lower", 0.05),
        RelativeRule("no_force", "base_z_min", "higher", 0.03),
    ]
    for scenario in scenarios:
        if scenario == "no_force":
            continue
        rules.extend(
            [
                RelativeRule(scenario, "success_rate", "higher", 0.0),
                RelativeRule(scenario, "fall_rate", "lower", 0.0),
                RelativeRule(scenario, "external_force_recovery_success_rate", "higher", 0.0),
                RelativeRule(scenario, "post_force_mean_forward_velocity", "higher", 0.03),
                RelativeRule(scenario, "post_force_mean_abs_lateral_velocity", "lower", 0.03),
                RelativeRule(scenario, "post_force_mean_abs_yaw_rate", "lower", 0.05),
                RelativeRule(scenario, "mean_abs_final_lateral_displacement", "lower", 0.05),
                RelativeRule(scenario, "base_z_min", "higher", 0.03),
                RelativeRule(scenario, "force_active_step_compliance_rate", "higher", 0.02),
                RelativeRule(scenario, "episode_compliance_rate", "higher", 0.10),
                RelativeRule(scenario, "force_recovery_survival_rate", "higher", 0.0),
                RelativeRule(scenario, "force_recovery_posture_success_rate", "higher", 0.0),
                RelativeRule(scenario, "force_recovery_locomotion_success_rate", "higher", 0.0),
                RelativeRule(scenario, "force_recovery_min_base_z_after_force", "higher", 0.03),
                RelativeRule(
                    scenario,
                    "force_recovery_max_orientation_error_during_force",
                    "lower",
                    0.05,
                ),
            ]
        )
    return rules


def _default_primary_improvement_rules(scenarios: Sequence[str]) -> list[RelativeRule]:
    rules: list[RelativeRule] = []
    for scenario in scenarios:
        if scenario == "no_force":
            continue
        rules.extend(
            [
                RelativeRule(
                    scenario,
                    "mean_abs_final_lateral_displacement",
                    "lower",
                    0.05,
                    kind="primary_improvement",
                ),
                RelativeRule(
                    scenario,
                    "post_force_mean_forward_velocity",
                    "higher",
                    0.02,
                    kind="primary_improvement",
                ),
                RelativeRule(
                    scenario,
                    "post_force_mean_abs_lateral_velocity",
                    "lower",
                    0.02,
                    kind="primary_improvement",
                ),
                RelativeRule(
                    scenario,
                    "post_force_mean_abs_yaw_rate",
                    "lower",
                    0.02,
                    kind="primary_improvement",
                ),
                RelativeRule(
                    scenario,
                    "force_active_step_compliance_rate",
                    "higher",
                    0.02,
                    kind="primary_improvement",
                ),
                RelativeRule(
                    scenario,
                    "episode_compliance_rate",
                    "higher",
                    0.10,
                    kind="primary_improvement",
                ),
            ]
        )
    return rules


def _evaluate_rule(
    rule: RelativeRule,
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> RelativeCheck:
    baseline_value = _metric_value(baseline.get(rule.scenario, {}), rule.metric)
    candidate_value = _metric_value(candidate.get(rule.scenario, {}), rule.metric)
    if baseline_value is None and candidate_value is None:
        return RelativeCheck(
            kind=rule.kind,
            scenario=rule.scenario,
            metric=rule.metric,
            direction=rule.direction,
            baseline=None,
            candidate=None,
            tolerance=rule.tolerance,
            delta=None,
            passed=True,
            skipped=True,
        )
    if baseline_value is None or candidate_value is None:
        return RelativeCheck(
            kind=rule.kind,
            scenario=rule.scenario,
            metric=rule.metric,
            direction=rule.direction,
            baseline=baseline_value,
            candidate=candidate_value,
            tolerance=rule.tolerance,
            delta=None,
            passed=False,
        )
    delta = candidate_value - baseline_value
    if rule.direction == "higher":
        passed = candidate_value + rule.tolerance >= baseline_value
        if rule.kind == "primary_improvement":
            passed = candidate_value >= baseline_value + rule.tolerance
    elif rule.direction == "lower":
        passed = candidate_value <= baseline_value + rule.tolerance
        if rule.kind == "primary_improvement":
            passed = candidate_value <= baseline_value - rule.tolerance
    else:
        raise ValueError(f"unknown direction {rule.direction!r}")
    return RelativeCheck(
        kind=rule.kind,
        scenario=rule.scenario,
        metric=rule.metric,
        direction=rule.direction,
        baseline=baseline_value,
        candidate=candidate_value,
        tolerance=rule.tolerance,
        delta=delta,
        passed=passed,
    )


def _metric_value(aggregate: dict[str, Any], metric: str) -> float | None:
    value = aggregate.get(metric)
    if value is None:
        return None
    return float(value)


def _check_to_dict(check: RelativeCheck) -> dict[str, Any]:
    return {
        "kind": check.kind,
        "scenario": check.scenario,
        "metric": check.metric,
        "direction": check.direction,
        "baseline": check.baseline,
        "candidate": check.candidate,
        "tolerance": check.tolerance,
        "delta": check.delta,
        "passed": check.passed,
        "skipped": check.skipped,
    }


def _print_summary(
    *,
    checks: Sequence[RelativeCheck],
    primary_checks: Sequence[RelativeCheck],
    output_path: Path,
    passed: bool,
    min_primary_improvements: int,
) -> None:
    print(f"baseline_relative_gate: {'PASS' if passed else 'FAIL'}")
    for check in checks:
        if check.skipped:
            continue
        status = "PASS" if check.passed else "FAIL"
        print(
            f"{status} {check.scenario}.{check.metric}: "
            f"candidate={_fmt(check.candidate)} baseline={_fmt(check.baseline)} "
            f"direction={check.direction} tol={check.tolerance:g}"
        )
    primary_improvements = [
        check
        for check in primary_checks
        if not check.skipped and check.passed
    ]
    print(
        "primary_improvements: "
        f"{len(primary_improvements)} >= {min_primary_improvements}"
    )
    for check in primary_improvements:
        print(
            f"IMPROVED {check.scenario}.{check.metric}: "
            f"candidate={_fmt(check.candidate)} baseline={_fmt(check.baseline)}"
        )
    print(f"wrote {output_path}")


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
