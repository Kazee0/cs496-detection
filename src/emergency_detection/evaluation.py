"""Metrics and reporting for classifier and live-detection evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from emergency_detection.config import ALERT_LABELS


@dataclass(frozen=True)
class EvaluationReport:
    """Structured metrics aligned with README evaluation plan."""

    class_names: list[str]
    accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    emergency_precision: float
    emergency_recall: float
    emergency_f1: float
    normal_false_positive_rate: float
    confusion_matrix: list[list[int]]
    test_size: int
    alert_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _emergency_label_indices(class_names: list[str]) -> list[int]:
    return [index for index, name in enumerate(class_names) if name in ALERT_LABELS]


def _normal_label_index(class_names: list[str]) -> int | None:
    try:
        return class_names.index("normal_background")
    except ValueError:
        return None


def emergency_alert_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    *,
    threshold: float,
) -> tuple[float, float]:
    """Return (emergency recall, normal false positive rate) using alert rules."""
    emergency_indices = _emergency_label_indices(class_names)
    normal_index = _normal_label_index(class_names)

    if not emergency_indices:
        return 0.0, 0.0

    emergency_true = np.isin(y_true, emergency_indices)
    if emergency_true.any():
        triggered = []
        for index in np.where(emergency_true)[0]:
            label_index = int(np.argmax(y_proba[index]))
            label = class_names[label_index]
            confidence = float(y_proba[index, label_index])
            triggered.append(label in ALERT_LABELS and confidence >= threshold)
        emergency_recall = float(np.mean(triggered))
    else:
        emergency_recall = 0.0

    normal_fpr = 0.0
    if normal_index is not None:
        normal_mask = y_true == normal_index
        if normal_mask.any():
            false_alarms = []
            for index in np.where(normal_mask)[0]:
                label_index = int(np.argmax(y_proba[index]))
                label = class_names[label_index]
                confidence = float(y_proba[index, label_index])
                false_alarms.append(label in ALERT_LABELS and confidence >= threshold)
            normal_fpr = float(np.mean(false_alarms))

    return emergency_recall, normal_fpr


def build_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    *,
    alert_threshold: float | None = None,
) -> EvaluationReport:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    emergency_indices = _emergency_label_indices(class_names)
    if emergency_indices:
        emergency_mask = np.isin(y_true, emergency_indices)
        emergency_pred = np.isin(y_pred, emergency_indices)
        emergency_precision, emergency_recall, emergency_f1, _ = precision_recall_fscore_support(
            emergency_mask,
            emergency_pred,
            average="binary",
            zero_division=0,
        )
    else:
        emergency_precision = emergency_recall = emergency_f1 = 0.0

    normal_fpr = 0.0
    normal_index = _normal_label_index(class_names)
    if normal_index is not None:
        normal_mask = y_true == normal_index
        if normal_mask.any():
            normal_fpr = float(np.mean(y_pred[normal_mask] != normal_index))

    if alert_threshold is not None:
        threshold_recall, threshold_fpr = emergency_alert_rate(
            y_true, y_pred, y_proba, class_names, threshold=alert_threshold
        )
        emergency_recall = threshold_recall
        normal_fpr = threshold_fpr

    return EvaluationReport(
        class_names=class_names,
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class_precision={
            name: float(precision[index]) for index, name in enumerate(class_names)
        },
        per_class_recall={
            name: float(recall[index]) for index, name in enumerate(class_names)
        },
        per_class_f1={name: float(f1[index]) for index, name in enumerate(class_names)},
        emergency_precision=float(emergency_precision),
        emergency_recall=float(emergency_recall),
        emergency_f1=float(emergency_f1),
        normal_false_positive_rate=float(normal_fpr),
        confusion_matrix=confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
        test_size=int(y_true.size),
        alert_threshold=alert_threshold,
    )


def format_report_text(report: EvaluationReport) -> str:
    lines = [
        "=== Evaluation summary ===",
        f"Test samples: {report.test_size}",
        f"Accuracy: {report.accuracy:.4f}",
        "",
        "Emergency classes (grouped binary):",
        f"  Precision: {report.emergency_precision:.4f}",
        f"  Recall:    {report.emergency_recall:.4f}",
        f"  F1:        {report.emergency_f1:.4f}",
        "",
        f"Normal false positive rate: {report.normal_false_positive_rate:.4f}",
    ]
    if report.alert_threshold is not None:
        lines.append(f"(Alert threshold: {report.alert_threshold:.2f})")
    lines.extend(
        [
            "",
            "Per-class metrics:",
        ]
    )
    for name in report.class_names:
        lines.append(
            f"  {name}: P={report.per_class_precision[name]:.3f} "
            f"R={report.per_class_recall[name]:.3f} "
            f"F1={report.per_class_f1[name]:.3f}"
        )
    lines.extend(["", "Confusion matrix (rows=true, cols=pred):"])
    header = "             " + " ".join(f"{name[:8]:>8}" for name in report.class_names)
    lines.append(header)
    for index, row in enumerate(report.confusion_matrix):
        label = report.class_names[index][:12]
        counts = " ".join(f"{value:8d}" for value in row)
        lines.append(f"{label:12} {counts}")
    return "\n".join(lines)


def sklearn_classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
