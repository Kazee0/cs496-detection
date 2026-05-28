"""Extract YAMNet embeddings from a WAV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.audio import describe_waveform, load_wav_mono
from emergency_detection.yamnet import run_yamnet, summarize_top_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path, help="Path to a WAV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npy path for saving embeddings.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of YAMNet AudioSet classes to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.wav_path.exists():
        print(f"File not found: {args.wav_path}")
        return 1

    waveform = load_wav_mono(args.wav_path)
    info = describe_waveform(waveform)

    print("Audio:")
    for key, value in info.items():
        print(f"- {key}: {value:.4f}")

    print()
    print("Running YAMNet...")
    output = run_yamnet(waveform)

    print(f"scores shape: {output.scores.shape}")
    print(f"embeddings shape: {output.embeddings.shape}")
    print(f"spectrogram shape: {output.spectrogram.shape}")

    print()
    print("Top YAMNet classes:")
    for class_name, score in summarize_top_classes(output, limit=args.top_k):
        print(f"- {class_name}: {score:.4f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, output.embeddings)
        print()
        print(f"Saved embeddings: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
