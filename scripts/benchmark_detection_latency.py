"""Benchmark end-to-end inference latency on local WAV clips.

Usage:
    python scripts/benchmark_detection_latency.py
    python scripts/benchmark_detection_latency.py --wav samples/mixkit-alert-alarm-1005.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.audio import load_wav_mono
from emergency_detection.config import WINDOW_SAMPLES
from emergency_detection.yamnet import load_yamnet_model, run_yamnet

MODELS_DIR = PROJECT_ROOT / "models"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wav",
        type=Path,
        action="append",
        help="WAV file to benchmark (repeatable). Defaults to all data/raw/**/*.wav",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum files when scanning data/raw.",
    )
    return parser.parse_args()


def collect_wav_paths(args: argparse.Namespace) -> list[Path]:
    if args.wav:
        return [path for path in args.wav if path.exists()]
    paths = sorted(RAW_DIR.rglob("*.wav"))
    if not paths:
        demo = PROJECT_ROOT / "mixkit-alert-alarm-1005.wav"
        if demo.exists():
            return [demo]
    return paths[: args.max_files]


def load_classifier():
    model_path = MODELS_DIR / "classifier.pkl"
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    return bundle["classifier"], bundle["scaler"], bundle["class_names"]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), pct))


def main() -> int:
    args = parse_args()
    wav_paths = collect_wav_paths(args)
    if not wav_paths:
        print("No WAV files found. Add clips under data/raw/ or pass --wav.")
        return 1

    print("Loading YAMNet...")
    t0 = time.perf_counter()
    yamnet = load_yamnet_model()
    yamnet_load_s = time.perf_counter() - t0
    print(f"YAMNet load: {yamnet_load_s:.2f}s")

    classifier_bundle = load_classifier()
    latencies_ms: list[float] = []

    for wav_path in wav_paths:
        waveform = load_wav_mono(wav_path)
        if waveform.size < WINDOW_SAMPLES:
            waveform = np.pad(waveform, (0, WINDOW_SAMPLES - waveform.size))

        start = time.perf_counter()
        output = run_yamnet(waveform[:WINDOW_SAMPLES], yamnet_model=yamnet)
        embedding = output.embeddings.mean(axis=0)
        if classifier_bundle is not None:
            clf, scaler, class_names = classifier_bundle
            scaled = scaler.transform(embedding.reshape(1, -1))
            clf.predict_proba(scaled)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        print(f"  {wav_path.name}: {elapsed_ms:.1f} ms")

    print("\n=== Latency (per 1s window, excluding first-time model load) ===")
    print(f"Files: {len(latencies_ms)}")
    print(f"Mean:   {np.mean(latencies_ms):.1f} ms")
    print(f"Median: {np.median(latencies_ms):.1f} ms")
    print(f"P95:    {percentile(latencies_ms, 95):.1f} ms")
    print(f"Max:    {max(latencies_ms):.1f} ms")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "latency_ms.txt"
    out_path.write_text(
        "\n".join(f"{value:.2f}" for value in latencies_ms),
        encoding="utf-8",
    )
    print(f"\nPer-file timings: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
