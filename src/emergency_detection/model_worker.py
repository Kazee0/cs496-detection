"""Separate-process model worker for live dashboard inference."""

from __future__ import annotations

import queue
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np

from emergency_detection.config import LABEL_ALERT_THRESHOLDS
from emergency_detection.inference import classify_window, waveform_rms
from emergency_detection.yamnet import load_yamnet_model


def run_model_worker(input_queue, output_queue, models_dir: str, min_signal_rms: float) -> None:
    """Load YAMNet/classifier once, then classify waveform windows from a queue."""
    try:
        output_queue.put({"type": "status", "message": "Worker loading classifier..."})
        bundle = joblib.load(Path(models_dir) / "classifier.pkl")

        output_queue.put({"type": "status", "message": "Worker loading YAMNet..."})
        yamnet = load_yamnet_model()

        clf = bundle["classifier"]
        scaler = bundle["scaler"]
        class_names: list[str] = bundle["class_names"]
        class_thresholds: dict[str, float] = bundle.get("class_thresholds", LABEL_ALERT_THRESHOLDS)
        output_queue.put({"type": "ready", "message": "Model worker ready."})
    except Exception:
        output_queue.put({
            "type": "error",
            "message": "Model worker failed during startup.",
            "traceback": traceback.format_exc(),
        })
        return

    while True:
        try:
            item = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:
            return

        try:
            waveform = np.asarray(item, dtype=np.float32)
            label, confidence, proba = classify_window(
                waveform,
                yamnet,
                clf,
                scaler,
                class_names,
                min_signal_rms=min_signal_rms,
                class_thresholds=class_thresholds,
            )
            output_queue.put({
                "type": "prediction",
                "label": label,
                "confidence": confidence,
                "proba": proba,
                "rms": waveform_rms(waveform),
            })
        except Exception:
            output_queue.put({
                "type": "error",
                "message": "Model worker failed during inference.",
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    # This module is intended to be launched through multiprocessing by the dashboard.
    sys.exit("Use run_model_worker() from multiprocessing.Process.")
