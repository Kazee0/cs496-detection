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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> int:
    embeddings = np.load(PROCESSED_DIR / "embeddings.npy")
    labels = np.load(PROCESSED_DIR / "labels.npy")
    class_names = (PROCESSED_DIR / "class_names.txt").read_text().strip().splitlines()

    print(f"Loaded {embeddings.shape[0]} samples, {len(class_names)} classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining logistic regression classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "scaler": scaler, "class_names": class_names}, MODELS_DIR / "classifier.pkl")
    print(f"\nModel saved to {MODELS_DIR / 'classifier.pkl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
