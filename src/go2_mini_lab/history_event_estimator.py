from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .event_detector_audit import audit_detector_trace_rows
from .replay_bank import load_trace_rows


PROPRIOCEPTIVE_FEATURE_WIDTH = 45


@dataclass(frozen=True)
class HistoryDataset:
    features: list[list[float]]
    labels: list[int]
    history_steps: int
    label_mode: str = "active"
    onset_window_s: float = 0.25


@dataclass(frozen=True)
class LinearHistoryEventEstimator:
    weights: list[float]
    bias: float
    feature_mean: list[float]
    feature_std: list[float]
    history_steps: int
    threshold: float
    release_threshold: float
    feature_width: int = PROPRIOCEPTIVE_FEATURE_WIDTH
    training_label_mode: str = "active"
    training_onset_window_s: float = 0.25

    def score_feature(self, feature: Sequence[float]) -> float:
        if len(feature) != len(self.weights):
            raise ValueError(f"feature width mismatch: {len(feature)} != {len(self.weights)}")
        total = self.bias
        for value, weight, mean, std in zip(
            feature,
            self.weights,
            self.feature_mean,
            self.feature_std,
            strict=True,
        ):
            total += ((float(value) - mean) / std) * weight
        return _sigmoid(total)

    def score_features(self, features: Sequence[Sequence[float]]) -> list[float]:
        return [self.score_feature(feature) for feature in features]

    def predict_features(self, features: Sequence[Sequence[float]]) -> list[bool]:
        active = False
        predictions: list[bool] = []
        for score in self.score_features(features):
            if active:
                active = score >= self.release_threshold
            else:
                active = score >= self.threshold
            predictions.append(active)
        return predictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "go2_history_event_estimator_v1",
            "weights": self.weights,
            "bias": self.bias,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "history_steps": self.history_steps,
            "threshold": self.threshold,
            "release_threshold": self.release_threshold,
            "feature_width": self.feature_width,
            "training_label_mode": self.training_label_mode,
            "training_onset_window_s": self.training_onset_window_s,
            "feature_contract": {
                "uses_true_external_force": False,
                "row_feature_order": [
                    "command[3]",
                    "base_angular_velocity_from_qvel_or_row[3]",
                    "projected_gravity[3]",
                    "joint_positions[12]",
                    "joint_velocities[12]",
                    "last_action[12]",
                ],
                "history_feature": "flattened padded history plus current-minus-oldest delta",
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearHistoryEventEstimator":
        if payload.get("format") != "go2_history_event_estimator_v1":
            raise ValueError(f"unsupported estimator format: {payload.get('format')}")
        return cls(
            weights=[float(value) for value in payload["weights"]],
            bias=float(payload["bias"]),
            feature_mean=[float(value) for value in payload["feature_mean"]],
            feature_std=[float(value) for value in payload["feature_std"]],
            history_steps=int(payload["history_steps"]),
            threshold=float(payload["threshold"]),
            release_threshold=float(payload["release_threshold"]),
            feature_width=int(payload.get("feature_width", PROPRIOCEPTIVE_FEATURE_WIDTH)),
            training_label_mode=str(payload.get("training_label_mode", "active")),
            training_onset_window_s=float(payload.get("training_onset_window_s", 0.25)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearHistoryEventEstimator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def build_history_dataset(
    traces: Sequence[Sequence[dict[str, Any]]],
    *,
    history_steps: int,
    stride: int = 1,
    label_mode: str = "active",
    onset_window_s: float = 0.25,
) -> HistoryDataset:
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if label_mode not in {"active", "onset"}:
        raise ValueError("label_mode must be 'active' or 'onset'")
    if onset_window_s < 0.0:
        raise ValueError("onset_window_s must be non-negative")

    features: list[list[float]] = []
    labels: list[int] = []
    for trace in traces:
        for episode_rows in _rows_by_episode(trace):
            row_features = [_row_proprioceptive_feature(row) for row in episode_rows]
            onset_labels = _onset_labels(episode_rows, onset_window_s=onset_window_s)
            for index, row in enumerate(episode_rows):
                if index % stride != 0:
                    continue
                history = _padded_history(row_features, index, history_steps)
                features.append(_history_feature(history))
                if label_mode == "onset":
                    labels.append(onset_labels[index])
                else:
                    labels.append(1 if bool(row.get("external_force_active", False)) else 0)
    return HistoryDataset(
        features=features,
        labels=labels,
        history_steps=history_steps,
        label_mode=label_mode,
        onset_window_s=onset_window_s,
    )


def fit_linear_event_estimator(
    dataset: HistoryDataset,
    *,
    epochs: int = 400,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    positive_weight: float | None = None,
    max_false_positive_rate: float = 0.05,
    release_ratio: float = 0.90,
    progress_interval: int = 0,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> LinearHistoryEventEstimator:
    if not dataset.features:
        raise ValueError("cannot fit estimator from an empty dataset")
    if len(set(dataset.labels)) < 2:
        raise ValueError("history estimator training needs both positive and negative rows")

    np = _require_numpy()
    x = np.asarray(dataset.features, dtype=float)
    y = np.asarray(dataset.labels, dtype=float)
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    x_norm = (x - feature_mean) / feature_std

    pos_count = float(y.sum())
    neg_count = float(len(y) - pos_count)
    if positive_weight is None:
        positive_weight = max(1.0, neg_count / max(1.0, pos_count))
    sample_weight = np.where(y > 0.5, positive_weight, 1.0)
    weight_sum = float(sample_weight.sum())

    weights = np.zeros(x_norm.shape[1], dtype=float)
    prior = min(max(pos_count / float(len(y)), 1e-4), 1.0 - 1e-4)
    bias = math.log(prior / (1.0 - prior))

    for epoch in range(1, epochs + 1):
        logits = x_norm @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        error = (probs - y) * sample_weight
        grad_weights = (x_norm.T @ error) / weight_sum + l2 * weights
        grad_bias = float(error.sum() / weight_sum)
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
        if progress_callback is not None and _should_report_progress(
            epoch,
            epochs,
            progress_interval,
        ):
            loss = _weighted_log_loss(probs, y, sample_weight, weight_sum, weights, l2)
            progress_callback(epoch, epochs, loss)

    raw_scores = 1.0 / (1.0 + np.exp(-np.clip(x_norm @ weights + bias, -60.0, 60.0)))
    threshold = _calibrate_threshold(
        [float(value) for value in raw_scores],
        dataset.labels,
        max_false_positive_rate=max_false_positive_rate,
    )
    release_threshold = max(0.0, min(1.0, threshold * release_ratio))

    return LinearHistoryEventEstimator(
        weights=[float(value) for value in weights],
        bias=float(bias),
        feature_mean=[float(value) for value in feature_mean],
        feature_std=[float(value) for value in feature_std],
        history_steps=dataset.history_steps,
        threshold=threshold,
        release_threshold=release_threshold,
        training_label_mode=dataset.label_mode,
        training_onset_window_s=dataset.onset_window_s,
    )


def evaluate_history_estimator(
    rows: Sequence[dict[str, Any]],
    estimator: LinearHistoryEventEstimator | None,
    *,
    onset_window_s: float = 0.25,
    recovery_grace_s: float = 0.20,
) -> dict[str, Any]:
    annotated_rows: list[dict[str, Any]] = []
    if estimator is None:
        scores = [
            float(row.get("force_safety_detector_score", 1.0))
            if bool(row.get("force_safety_detector_active", False))
            else float(row.get("force_safety_detector_score", 0.0))
            for row in rows
        ]
        predictions = [bool(row.get("force_safety_detector_active", False)) for row in rows]
        annotated_rows = [dict(row) for row in rows]
        history_steps = None
        threshold = None
        release_threshold = None
    else:
        dataset = build_history_dataset([rows], history_steps=estimator.history_steps)
        scores = estimator.score_features(dataset.features)
        predictions = estimator.predict_features(dataset.features)
        for row, score, active in zip(rows, scores, predictions, strict=True):
            annotated = dict(row)
            annotated["force_safety_detector_active"] = bool(active)
            annotated["force_safety_detector_score"] = float(score)
            annotated_rows.append(annotated)
        history_steps = estimator.history_steps
        threshold = estimator.threshold
        release_threshold = estimator.release_threshold

    report = audit_detector_trace_rows(annotated_rows)
    onset_report = _audit_onset_detector_rows(
        annotated_rows,
        onset_window_s=onset_window_s,
        recovery_grace_s=recovery_grace_s,
    )
    report.update(
        {
            **onset_report,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_mean": sum(scores) / len(scores) if scores else None,
            "predicted_active_rate": (
                sum(1 for value in predictions if value) / len(predictions)
                if predictions
                else 0.0
            ),
            "history_steps": history_steps,
            "threshold": threshold,
            "release_threshold": release_threshold,
        }
    )
    return report


def summarize_history_audits(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {}
    keys = (
        "pure_no_force_false_positive_rate",
        "force_active_recall",
        "onset_recall",
        "event_detection_recall",
        "max_onset_detection_latency_s",
        "predicted_active_rate",
    )
    summary: dict[str, Any] = {"trace_count": len(reports)}
    for key in keys:
        values = [
            float(report[key])
            for report in reports
            if report.get(key) is not None
        ]
        if not values:
            continue
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    summary["passes_deployable_detector_target_count"] = sum(
        1
        for report in reports
        if bool(report.get("passes_pure_false_positive_rate"))
        and bool(report.get("passes_onset_recall"))
        and bool(report.get("passes_onset_latency"))
    )
    return summary


def _onset_labels(rows: Sequence[dict[str, Any]], *, onset_window_s: float) -> list[int]:
    labels = [0] * len(rows)
    for event in _force_events(rows):
        start_s = event["start_s"]
        for index in range(event["start_index"], event["end_index"] + 1):
            t_s = float(rows[index].get("t", 0.0))
            if 0.0 <= t_s - start_s <= onset_window_s + 1e-9:
                labels[index] = 1
    return labels


def _audit_onset_detector_rows(
    rows: Sequence[dict[str, Any]],
    *,
    onset_window_s: float,
    recovery_grace_s: float,
    false_positive_threshold: float = 0.05,
    onset_recall_threshold: float = 0.90,
    latency_threshold_s: float = 0.12,
) -> dict[str, Any]:
    events = _force_events(rows)
    pure_no_force_indices = _pure_no_force_indices(rows, events, recovery_grace_s=recovery_grace_s)
    pure_no_force_detector_count = sum(
        1 for index in pure_no_force_indices if bool(rows[index].get("force_safety_detector_active", False))
    )
    pure_false_positive_rate = (
        pure_no_force_detector_count / len(pure_no_force_indices)
        if pure_no_force_indices
        else 0.0
    )
    recovery_grace_rows = sum(
        1
        for index, row in enumerate(rows)
        if not bool(row.get("external_force_active", False))
        and index not in pure_no_force_indices
    )

    event_detection_latencies: list[float] = []
    onset_detection_latencies: list[float] = []
    for event in events:
        first_event_detection = _first_detection_latency(rows, event, max_latency_s=None)
        if first_event_detection is not None:
            event_detection_latencies.append(first_event_detection)
        first_onset_detection = _first_detection_latency(
            rows,
            event,
            max_latency_s=onset_window_s,
        )
        if first_onset_detection is not None:
            onset_detection_latencies.append(first_onset_detection)

    event_count = len(events)
    onset_recall = len(onset_detection_latencies) / event_count if event_count else 1.0
    event_detection_recall = len(event_detection_latencies) / event_count if event_count else 1.0
    mean_onset_latency = (
        sum(onset_detection_latencies) / len(onset_detection_latencies)
        if onset_detection_latencies
        else None
    )
    max_onset_latency = max(onset_detection_latencies) if onset_detection_latencies else None

    return {
        "event_count_total": event_count,
        "event_detection_count": len(event_detection_latencies),
        "event_detection_recall": event_detection_recall,
        "onset_window_s": onset_window_s,
        "onset_detection_count": len(onset_detection_latencies),
        "onset_recall": onset_recall,
        "mean_onset_detection_latency_s": mean_onset_latency,
        "max_onset_detection_latency_s": max_onset_latency,
        "recovery_grace_s": recovery_grace_s,
        "recovery_grace_rows": recovery_grace_rows,
        "pure_no_force_rows": len(pure_no_force_indices),
        "pure_no_force_false_positive_count": pure_no_force_detector_count,
        "pure_no_force_false_positive_rate": pure_false_positive_rate,
        "passes_pure_false_positive_rate": pure_false_positive_rate < false_positive_threshold,
        "passes_onset_recall": onset_recall >= onset_recall_threshold,
        "passes_onset_latency": (
            True if max_onset_latency is None else max_onset_latency <= latency_threshold_s
        ),
        "onset_thresholds": {
            "pure_false_positive_rate": false_positive_threshold,
            "onset_recall": onset_recall_threshold,
            "onset_latency_s": latency_threshold_s,
        },
    }


def _force_events(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_active = False
    previous_episode: int | None = None
    start_index: int | None = None
    start_s: float | None = None

    for index, row in enumerate(rows):
        episode = int(row.get("episode", 0))
        active = bool(row.get("external_force_active", False))
        if previous_episode is not None and episode != previous_episode:
            if previous_active and start_index is not None and start_s is not None:
                end_index = index - 1
                events.append(
                    {
                        "start_index": start_index,
                        "end_index": end_index,
                        "start_s": start_s,
                        "end_s": float(rows[end_index].get("t", start_s)),
                    }
                )
            previous_active = False
            start_index = None
            start_s = None

        if active and not previous_active:
            start_index = index
            start_s = float(row.get("t", 0.0))
        if not active and previous_active and start_index is not None and start_s is not None:
            end_index = index - 1
            events.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_s": start_s,
                    "end_s": float(rows[end_index].get("t", start_s)),
                }
            )
            start_index = None
            start_s = None

        previous_active = active
        previous_episode = episode

    if previous_active and start_index is not None and start_s is not None:
        end_index = len(rows) - 1
        events.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "start_s": start_s,
                "end_s": float(rows[end_index].get("t", start_s)),
            }
        )
    return events


def _pure_no_force_indices(
    rows: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    recovery_grace_s: float,
) -> set[int]:
    pure: set[int] = set()
    for index, row in enumerate(rows):
        if bool(row.get("external_force_active", False)):
            continue
        t_s = float(row.get("t", 0.0))
        episode = int(row.get("episode", 0))
        in_recovery_grace = False
        for event in events:
            event_episode = int(rows[event["start_index"]].get("episode", 0))
            if event_episode != episode:
                continue
            if 0.0 < t_s - float(event["end_s"]) <= recovery_grace_s + 1e-9:
                in_recovery_grace = True
                break
        if not in_recovery_grace:
            pure.add(index)
    return pure


def _first_detection_latency(
    rows: Sequence[dict[str, Any]],
    event: dict[str, Any],
    *,
    max_latency_s: float | None,
) -> float | None:
    start_s = float(event["start_s"])
    for index in range(event["start_index"], event["end_index"] + 1):
        t_s = float(rows[index].get("t", 0.0))
        latency = max(0.0, t_s - start_s)
        if max_latency_s is not None and latency > max_latency_s + 1e-9:
            break
        if bool(rows[index].get("force_safety_detector_active", False)):
            return latency
    return None


def _rows_by_episode(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_episode: int | None = None
    for row in rows:
        episode = int(row.get("episode", 0))
        if current and episode != current_episode:
            grouped.append(current)
            current = []
        current.append(row)
        current_episode = episode
    if current:
        grouped.append(current)
    return grouped


def _row_proprioceptive_feature(row: dict[str, Any]) -> list[float]:
    qvel = _fixed_vector(row.get("qvel"), 6)
    base_angular_velocity = _fixed_vector(row.get("base_angular_velocity"), 3)
    if not any(base_angular_velocity):
        base_angular_velocity = qvel[3:6]
    feature = build_proprioceptive_feature(
        command=_fixed_vector(row.get("command"), 3),
        base_angular_velocity=base_angular_velocity,
        projected_gravity=_fixed_vector(row.get("projected_gravity"), 3),
        joint_positions=_fixed_vector(row.get("joint_positions"), 12),
        joint_velocities=_fixed_vector(row.get("joint_velocities"), 12),
        last_action=_fixed_vector(row.get("last_action"), 12),
    )
    if len(feature) != PROPRIOCEPTIVE_FEATURE_WIDTH:
        raise AssertionError(f"unexpected proprioceptive feature width: {len(feature)}")
    return feature


def build_proprioceptive_feature(
    *,
    command: Sequence[float],
    base_angular_velocity: Sequence[float],
    projected_gravity: Sequence[float],
    joint_positions: Sequence[float],
    joint_velocities: Sequence[float],
    last_action: Sequence[float],
) -> list[float]:
    feature = (
        [float(value) for value in command[:3]]
        + [float(value) for value in base_angular_velocity[:3]]
        + [float(value) for value in projected_gravity[:3]]
        + [float(value) for value in joint_positions[:12]]
        + [float(value) for value in joint_velocities[:12]]
        + [float(value) for value in last_action[:12]]
    )
    if len(feature) != PROPRIOCEPTIVE_FEATURE_WIDTH:
        raise ValueError(f"proprioceptive feature must have width {PROPRIOCEPTIVE_FEATURE_WIDTH}, got {len(feature)}")
    return feature


def _fixed_vector(value: Any, width: int) -> list[float]:
    if value is None:
        return [0.0] * width
    values = [float(item) for item in value]
    if len(values) >= width:
        return values[:width]
    return values + [0.0] * (width - len(values))


def _padded_history(
    row_features: Sequence[Sequence[float]],
    index: int,
    history_steps: int,
) -> list[list[float]]:
    start = index - history_steps + 1
    history: list[list[float]] = []
    first = list(row_features[0])
    for history_index in range(start, index + 1):
        if history_index < 0:
            history.append(first)
        else:
            history.append(list(row_features[history_index]))
    return history


def _history_feature(history: Sequence[Sequence[float]]) -> list[float]:
    return build_history_feature(history)


def build_history_feature(history: Sequence[Sequence[float]]) -> list[float]:
    if not history:
        raise ValueError("history must not be empty")
    flattened: list[float] = []
    for row_feature in history:
        if len(row_feature) != PROPRIOCEPTIVE_FEATURE_WIDTH:
            raise ValueError(
                f"history row feature must have width {PROPRIOCEPTIVE_FEATURE_WIDTH}, got {len(row_feature)}"
            )
        flattened.extend(row_feature)
    current = history[-1]
    oldest = history[0]
    flattened.extend(float(a) - float(b) for a, b in zip(current, oldest, strict=True))
    return flattened


def _calibrate_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    max_false_positive_rate: float,
) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return 0.5
    epsilon = 1e-9
    candidates = sorted({max(0.0, min(1.0, score + epsilon)) for score in scores}, reverse=True)
    candidates.append(0.0)
    best_threshold = 1.0
    best_recall = -1.0
    for threshold in candidates:
        false_positive_rate = sum(1 for score in negatives if score >= threshold) / len(negatives)
        if false_positive_rate > max_false_positive_rate:
            continue
        recall = sum(1 for score in positives if score >= threshold) / len(positives)
        if recall > best_recall or (math.isclose(recall, best_recall) and threshold < best_threshold):
            best_recall = recall
            best_threshold = threshold
    return best_threshold


def _should_report_progress(epoch: int, total_epochs: int, progress_interval: int) -> bool:
    if progress_interval <= 0:
        return False
    return epoch == 1 or epoch == total_epochs or epoch % progress_interval == 0


def _weighted_log_loss(
    probs: Any,
    labels: Any,
    sample_weight: Any,
    weight_sum: float,
    weights: Any,
    l2: float,
) -> float:
    np = _require_numpy()
    clipped = np.clip(probs, 1e-9, 1.0 - 1e-9)
    data_loss = -float(
        np.sum(sample_weight * (labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
        / weight_sum
    )
    reg_loss = 0.5 * l2 * float(np.sum(weights * weights))
    return data_loss + reg_loss


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "history event estimator training requires numpy; install the train or sim extras"
        ) from exc
    return np


def _resolve_trace_paths(paths: Sequence[str], globs: Sequence[str]) -> list[Path]:
    resolved = [Path(path) for path in paths]
    for pattern in globs:
        resolved.extend(Path(path) for path in sorted(glob.glob(pattern)))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise FileNotFoundError("no trace JSONL files matched")
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing trace JSONL files: {missing}")
    return unique


def _split_train_validation_paths(
    paths: Sequence[Path],
    *,
    validation_fraction: float,
) -> tuple[list[Path], list[Path]]:
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0.0 or len(paths) < 2:
        return list(paths), []
    validation_count = max(1, int(round(len(paths) * validation_fraction)))
    validation_count = min(validation_count, len(paths) - 1)
    return list(paths[:-validation_count]), list(paths[-validation_count:])


def _compact_audit_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "row_count",
        "event_count_total",
        "pure_no_force_false_positive_rate",
        "force_active_recall",
        "event_detection_recall",
        "onset_recall",
        "max_onset_detection_latency_s",
        "predicted_active_rate",
        "passes_pure_false_positive_rate",
        "passes_onset_recall",
        "passes_onset_latency",
    )
    compact = {"trace": str(path)}
    compact.update({key: report.get(key) for key in keys})
    return compact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and audit a deployable history-based Go2 force-event estimator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("trace_jsonl", nargs="*")
    train_parser.add_argument("--train-glob", action="append", default=[])
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--report-output", type=Path)
    train_parser.add_argument("--history-steps", type=int, default=15)
    train_parser.add_argument("--label-mode", choices=("active", "onset"), default="onset")
    train_parser.add_argument("--onset-window-s", type=float, default=0.25)
    train_parser.add_argument("--recovery-grace-s", type=float, default=0.20)
    train_parser.add_argument("--epochs", type=int, default=400)
    train_parser.add_argument("--learning-rate", type=float, default=0.05)
    train_parser.add_argument("--l2", type=float, default=1e-3)
    train_parser.add_argument("--max-false-positive-rate", type=float, default=0.05)
    train_parser.add_argument("--release-ratio", type=float, default=0.90)
    train_parser.add_argument("--validation-fraction", type=float, default=0.20)
    train_parser.add_argument("--progress-interval", type=int, default=50)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("model", type=Path)
    audit_parser.add_argument("trace_jsonl", type=Path)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--onset-window-s", type=float, default=0.25)
    audit_parser.add_argument("--recovery-grace-s", type=float, default=0.20)

    args = parser.parse_args(argv)
    if args.command == "train":
        paths = _resolve_trace_paths(args.trace_jsonl, args.train_glob)
        train_paths, validation_paths = _split_train_validation_paths(
            paths,
            validation_fraction=args.validation_fraction,
        )
        traces = [load_trace_rows(path) for path in train_paths]
        dataset = build_history_dataset(
            traces,
            history_steps=args.history_steps,
            label_mode=args.label_mode,
            onset_window_s=args.onset_window_s,
        )
        print(
            "history_event_estimator train: "
            f"traces={len(train_paths)} validation_traces={len(validation_paths)} "
            f"rows={len(dataset.labels)} positives={int(sum(dataset.labels))} "
            f"label_mode={args.label_mode} history_steps={args.history_steps}",
            flush=True,
        )
        estimator = fit_linear_event_estimator(
            dataset,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            max_false_positive_rate=args.max_false_positive_rate,
            release_ratio=args.release_ratio,
            progress_interval=args.progress_interval,
            progress_callback=lambda epoch, total, loss: print(
                f"history_event_estimator train: epoch {epoch}/{total} loss={loss:.6f}",
                flush=True,
            ),
        )
        print("history_event_estimator train: calibrating threshold and writing outputs", flush=True)
        estimator.save(args.output)
        validation_audits = []
        for path in validation_paths:
            audit_report = evaluate_history_estimator(
                load_trace_rows(path),
                estimator,
                onset_window_s=args.onset_window_s,
                recovery_grace_s=args.recovery_grace_s,
            )
            validation_audits.append(_compact_audit_report(path, audit_report))
        report = {
            "trace_count": len(paths),
            "train_trace_count": len(train_paths),
            "validation_trace_count": len(validation_paths),
            "row_count": len(dataset.labels),
            "positive_rows": int(sum(dataset.labels)),
            "negative_rows": int(len(dataset.labels) - sum(dataset.labels)),
            "history_steps": args.history_steps,
            "label_mode": args.label_mode,
            "onset_window_s": args.onset_window_s,
            "recovery_grace_s": args.recovery_grace_s,
            "threshold": estimator.threshold,
            "release_threshold": estimator.release_threshold,
            "model": str(args.output),
            "train_traces": [str(path) for path in train_paths],
            "validation_traces": [str(path) for path in validation_paths],
            "validation_summary": summarize_history_audits(validation_audits),
            "validation_audits": validation_audits,
        }
        if args.report_output is not None:
            _write_json(args.report_output, report)
        print(f"wrote {args.output}")
        if args.report_output is not None:
            print(f"wrote {args.report_output}")
        return 0

    if args.command == "audit":
        estimator = LinearHistoryEventEstimator.load(args.model)
        rows = load_trace_rows(args.trace_jsonl)
        report = evaluate_history_estimator(
            rows,
            estimator,
            onset_window_s=args.onset_window_s,
            recovery_grace_s=args.recovery_grace_s,
        )
        report["model"] = str(args.model)
        report["trace"] = str(args.trace_jsonl)
        _write_json(args.output, report)
        print(f"wrote {args.output}")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
