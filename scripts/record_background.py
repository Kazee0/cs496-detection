"""Record ambient background noise and split into 1-second WAV clips.

Usage:
    python scripts/record_background.py
    python scripts/record_background.py --duration 60 --device 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import SAMPLE_RATE, WINDOW_SAMPLES

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "normal_background"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=40, help="Total seconds to record.")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Recording {args.duration} seconds of background noise...")
    print("Stay quiet — this is capturing ambient room sound.")
    print("Starting in 3 seconds...")

    sd.sleep(3000)
    print("Recording now...")

    audio = sd.rec(
        frames=args.duration * SAMPLE_RATE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=args.device,
    )
    sd.wait()
    print("Done recording.")

    waveform = audio[:, 0]

    # Split into 1-second clips
    num_clips = len(waveform) // WINDOW_SAMPLES
    saved = 0
    for i in range(num_clips):
        clip = waveform[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
        clip_int16 = (clip * 32767).astype(np.int16)
        out_path = OUTPUT_DIR / f"background_{i:03d}.wav"
        wavfile.write(out_path, SAMPLE_RATE, clip_int16)
        saved += 1

    print(f"Saved {saved} clips to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
