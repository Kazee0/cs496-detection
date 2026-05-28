"""Load YAMNet and run one synthetic-audio inference smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import SAMPLE_RATE
from emergency_detection.yamnet import run_yamnet, summarize_top_classes


def main() -> int:
    seconds = 1.0
    frequency_hz = 440.0
    timeline = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    waveform = 0.1 * np.sin(2.0 * np.pi * frequency_hz * timeline)

    print("Loading YAMNet and running synthetic waveform inference...")
    output = run_yamnet(waveform.astype(np.float32))

    print(f"scores shape: {output.scores.shape}")
    print(f"embeddings shape: {output.embeddings.shape}")
    print(f"spectrogram shape: {output.spectrogram.shape}")
    print()
    print("Top YAMNet classes:")
    for class_name, score in summarize_top_classes(output):
        print(f"- {class_name}: {score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
