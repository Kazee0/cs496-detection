"""Live microphone emergency sound detection.

Usage:
    python scripts/live_detection.py
    python scripts/live_detection.py --device 1 --threshold 0.75
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
    ALERT_LABELS,
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_ALERT_COOLDOWN_SECONDS,
)
from emergency_detection.yamnet import load_yamnet_model, run_yamnet

MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None, help="Microphone device index.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_ALERT_THRESHOLD, help="Alert confidence threshold.")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_ALERT_COOLDOWN_SECONDS, help="Seconds between repeated alerts.")
    return parser.parse_args()


def load_model(models_dir: Path):
    model_path = models_dir / "classifier.pkl"
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("Run scripts/train_classifier.py first.")
        sys.exit(1)
    data = joblib.load(model_path)
    return data["classifier"], data["scaler"], data["class_names"]


def predict(embedding: np.ndarray, clf, scaler, class_names: list[str]) -> tuple[str, float]:
    x = scaler.transform(embedding.reshape(1, -1))
    proba = clf.predict_proba(x)[0]
    index = int(np.argmax(proba))
    return class_names[index], float(proba[index])


def main() -> int:
    args = parse_args()

    print("Loading classifier...")
    clf, scaler, class_names = load_model(MODELS_DIR)
    print(f"Classes: {class_names}")

    print("Loading YAMNet...")
    yamnet = load_yamnet_model()

    # Rolling buffer to accumulate audio from the microphone stream
    buffer: deque[float] = deque(maxlen=WINDOW_SAMPLES)
    samples_since_hop = 0
    last_alert_time = 0.0

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal samples_since_hop, last_alert_time

        if status:
            print(f"[stream status] {status}", file=sys.stderr)

        chunk = indata[:, 0].astype(np.float32)
        buffer.extend(chunk)
        samples_since_hop += len(chunk)

        if len(buffer) < WINDOW_SAMPLES:
            return

        if samples_since_hop < HOP_SAMPLES:
            return
        samples_since_hop = 0

        waveform = np.array(buffer, dtype=np.float32)
        output = run_yamnet(waveform, yamnet_model=yamnet)
        embedding = output.embeddings.mean(axis=0)
        label, confidence = predict(embedding, clf, scaler, class_names)

        now = time.time()
        is_alert = label in ALERT_LABELS and confidence >= args.threshold
        cooldown_ok = (now - last_alert_time) >= args.cooldown

        if is_alert and cooldown_ok:
            last_alert_time = now
            print(f"*** ALERT *** {label.upper()} ({confidence:.0%})", flush=True)
        else:
            print(f"  {label} ({confidence:.0%})", flush=True)

    print(f"\nListening... (threshold={args.threshold:.0%}, cooldown={args.cooldown}s)")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            device=args.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=HOP_SAMPLES,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
