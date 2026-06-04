"""Train a classifier on top of YAMNet embeddings.

Usage:
    python scripts/train_classifier.py

Reads embeddings and labels from data/processed/.
Saves the trained model to models/classifier.pkl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import LABEL_ALERT_THRESHOLDS
from emergency_detection.inference import predict_with_class_thresholds

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
MINORITY_TARGET_RATIO = 0.75
FIRE_ALARM_WEIGHT_MULTIPLIER = 3.0
MAX_NORMAL_FALSE_ALARM_RATE = 0.10
EMERGENCY_MIN_THRESHOLD = 0.05
EMERGENCY_MAX_THRESHOLD = 0.75
EMERGENCY_THRESHOLD_STEP = 0.05


def proba_by_class(clf, X_scaled: np.ndarray, class_count: int) -> np.ndarray:
    raw_proba = clf.predict_proba(X_scaled)
    proba = np.zeros((raw_proba.shape[0], class_count), dtype=raw_proba.dtype)
    for column_index, class_index in enumerate(clf.classes_):
        if 0 <= int(class_index) < class_count:
            proba[:, int(class_index)] = raw_proba[:, column_index]
    return proba


def predict_with_thresholds(
    clf,
    X_scaled: np.ndarray,
    class_names: list[str],
    class_thresholds: dict[str, float],
) -> np.ndarray:
    """Return labels using emergency-friendly per-class probability thresholds."""
    raw_proba = clf.predict_proba(X_scaled)
    predictions: list[int] = []
    for row in raw_proba:
        proba = {name: 0.0 for name in class_names}
        for column_index, class_index in enumerate(clf.classes_):
            if 0 <= int(class_index) < len(class_names):
                proba[class_names[int(class_index)]] = float(row[column_index])
        label, _ = predict_with_class_thresholds(proba, class_thresholds=class_thresholds)
        predictions.append(class_names.index(label) if label in class_names else int(np.argmax(row)))
    return np.array(predictions, dtype=np.int32)


def make_class_weights(y_train: np.ndarray, class_names: list[str]) -> dict[int, float]:
    """Balanced class weights plus an explicit fire-alarm recall boost."""
    total = y_train.size
    class_count = len(class_names)
    weights: dict[int, float] = {}
    for class_index, name in enumerate(class_names):
        count = int(np.sum(y_train == class_index))
        if count == 0:
            continue
        weight = total / (class_count * count)
        if name == "fire_or_smoke_alarm":
            weight *= FIRE_ALARM_WEIGHT_MULTIPLIER
        weights[class_index] = float(weight)
    return weights


def augment_minority_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample rare emergency classes with embedding-space mixup and jitter."""
    rng = np.random.default_rng(RANDOM_STATE)
    normal_index = class_names.index("normal_background") if "normal_background" in class_names else None
    class_counts = np.bincount(y_train, minlength=len(class_names))
    non_normal_counts = [
        int(count) for index, count in enumerate(class_counts)
        if index != normal_index and count > 0
    ]
    if not non_normal_counts:
        return X_train, y_train

    target_count = max(non_normal_counts)
    if normal_index is not None:
        target_count = max(target_count, int(class_counts[normal_index] * MINORITY_TARGET_RATIO))

    augmented_x = [X_train]
    augmented_y = [y_train]
    feature_std = np.std(X_train, axis=0)
    noise_scale = np.maximum(feature_std * 0.015, 1e-4)

    for class_index, name in enumerate(class_names):
        if name == "normal_background":
            continue
        class_mask = y_train == class_index
        class_samples = X_train[class_mask]
        if class_samples.size == 0 or class_samples.shape[0] >= target_count:
            continue

        needed = target_count - class_samples.shape[0]
        synthetic = []
        for _ in range(needed):
            first = class_samples[rng.integers(0, class_samples.shape[0])]
            second = class_samples[rng.integers(0, class_samples.shape[0])]
            mix = rng.uniform(0.35, 0.65)
            noise = rng.normal(0.0, noise_scale, size=first.shape)
            synthetic.append((mix * first + (1.0 - mix) * second + noise).astype(np.float32))

        augmented_x.append(np.array(synthetic, dtype=np.float32))
        augmented_y.append(np.full(needed, class_index, dtype=y_train.dtype))
        print(f"  embedding aug: {name} {class_samples.shape[0]} -> {target_count}")

    return np.vstack(augmented_x), np.concatenate(augmented_y)


def tune_emergency_thresholds(
    y_val: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Choose per-emergency thresholds that favor recall while limiting normal FPR."""
    thresholds = dict(LABEL_ALERT_THRESHOLDS)
    normal_index = class_names.index("normal_background") if "normal_background" in class_names else None
    candidates = np.arange(
        EMERGENCY_MIN_THRESHOLD,
        EMERGENCY_MAX_THRESHOLD + EMERGENCY_THRESHOLD_STEP / 2,
        EMERGENCY_THRESHOLD_STEP,
    )

    for class_index, name in enumerate(class_names):
        if name == "normal_background":
            continue

        positives = y_val == class_index
        if not positives.any():
            continue

        best_score: tuple[float, float, float] | None = None
        best_threshold = thresholds.get(name, 0.35)
        for threshold in candidates:
            predicted_positive = y_proba[:, class_index] >= threshold
            recall = float(np.mean(predicted_positive[positives]))

            if normal_index is not None:
                normal_mask = y_val == normal_index
                normal_fpr = (
                    float(np.mean(predicted_positive[normal_mask]))
                    if normal_mask.any()
                    else 0.0
                )
            else:
                normal_fpr = 0.0

            precision = (
                float(np.mean(y_val[predicted_positive] == class_index))
                if predicted_positive.any()
                else 0.0
            )
            if normal_fpr > MAX_NORMAL_FALSE_ALARM_RATE:
                continue

            score = (recall, precision, threshold)
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = float(threshold)

        thresholds[name] = best_threshold
        print(f"  tuned threshold: {name}={best_threshold:.2f}")

    return thresholds


def main() -> int:
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    labels = np.load(PROCESSED_DIR / "labels.npy")
    class_names = (PROCESSED_DIR / "class_names.txt").read_text().strip().splitlines()

    print(f"Loaded {embeddings.shape[0]} samples, {len(class_names)} classes: {class_names}")

    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    fit_indices, val_indices = train_test_split(
        train_indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels[train_indices],
    )
    X_fit, X_val, X_test = embeddings[fit_indices], embeddings[val_indices], embeddings[test_indices]
    y_fit, y_val, y_test = labels[fit_indices], labels[val_indices], labels[test_indices]
    print(f"Fit: {len(X_fit)}  Validation: {len(X_val)}  Test: {len(X_test)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "test_indices.npy", test_indices)

    print("\nAugmenting minority emergency embeddings...")
    X_fit_aug, y_fit_aug = augment_minority_embeddings(X_fit, y_fit, class_names)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_fit_aug)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\nTraining logistic regression classifier...")
    class_weights = make_class_weights(y_fit_aug, class_names)
    clf = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight=class_weights,
        C=0.5,
    )
    clf.fit(X_train, y_fit_aug)

    print("\nTuning emergency thresholds on validation split...")
    y_val_proba = proba_by_class(clf, X_val, len(class_names))
    class_thresholds = tune_emergency_thresholds(y_val, y_val_proba, class_names)

    y_pred = predict_with_thresholds(clf, X_test, class_names, class_thresholds)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=list(range(len(class_names)))))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": clf,
            "scaler": scaler,
            "class_names": class_names,
            "class_thresholds": class_thresholds,
            "decision_rule": "minority_augmented_cost_sensitive_thresholds",
            "class_weights": class_weights,
        },
        MODELS_DIR / "classifier.pkl",
    )
    print(f"\nModel saved to {MODELS_DIR / 'classifier.pkl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
