"""Evaluate the trained classifier on a held-out test split.

Usage:
    python scripts/evaluate_classifier.py
    python scripts/evaluate_classifier.py --threshold 0.75 --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import DEFAULT_ALERT_THRESHOLD
from emergency_detection.evaluation import (
    build_evaluation_report,
    format_report_text,
    sklearn_classification_report_text,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out fraction (must match train_classifier.py default).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (must match train_classifier.py).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ALERT_THRESHOLD,
        help="Confidence threshold for alert-style false positive rate.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save confusion matrix heatmap to outputs/confusion_matrix.png",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=OUTPUTS_DIR / "evaluation_report.json",
        help="Path for JSON metrics export.",
    )
    return parser.parse_args()


def load_test_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices_path = PROCESSED_DIR / "test_indices.npy"
    if indices_path.exists():
        test_indices = np.load(indices_path)
        return embeddings[test_indices], labels[test_indices], test_indices

    _, X_test, _, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_test, y_test, np.array([], dtype=np.int64)


def save_confusion_plot(report, output_path: Path) -> None:
    matrix = np.array(report.confusion_matrix)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(report.class_names)))
    axis.set_yticks(range(len(report.class_names)))
    axis.set_xticklabels(report.class_names, rotation=45, ha="right")
    axis.set_yticklabels(report.class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix (test set)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def predict_proba_by_class(clf, X_test_scaled: np.ndarray, class_count: int) -> np.ndarray:
    raw_proba = clf.predict_proba(X_test_scaled)
    proba = np.zeros((raw_proba.shape[0], class_count), dtype=raw_proba.dtype)
    for column_index, class_index in enumerate(clf.classes_):
        if 0 <= class_index < class_count:
            proba[:, int(class_index)] = raw_proba[:, column_index]
    return proba


def main() -> int:
    args = parse_args()

    model_path = MODELS_DIR / "classifier.pkl"
    if not model_path.exists():
        print(f"No model at {model_path}. Run scripts/train_classifier.py first.")
        return 1

    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    labels = np.load(PROCESSED_DIR / "labels.npy")
    class_names = (PROCESSED_DIR / "class_names.txt").read_text().strip().splitlines()

    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    scaler = bundle["scaler"]

    X_test, y_test, _ = load_test_split(
        embeddings,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    y_proba = predict_proba_by_class(clf, X_test_scaled, len(class_names))

    report = build_evaluation_report(
        y_test,
        y_pred,
        y_proba,
        class_names,
        alert_threshold=args.threshold,
    )

    print(format_report_text(report))
    print()
    print(sklearn_classification_report_text(y_test, y_pred, class_names))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\nJSON report: {args.json}")

    if args.plot:
        plot_path = OUTPUTS_DIR / "confusion_matrix.png"
        save_confusion_plot(report, plot_path)
        print(f"Confusion matrix plot: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
