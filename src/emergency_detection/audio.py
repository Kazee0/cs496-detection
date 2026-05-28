"""Audio loading utilities for model training and inference."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

from emergency_detection.config import SAMPLE_RATE


def load_wav_mono(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file as mono float32 audio at the target sample rate."""
    source_rate, audio = wavfile.read(path)
    waveform = np.asarray(audio)

    if np.issubdtype(waveform.dtype, np.integer):
        max_value = np.iinfo(waveform.dtype).max
        waveform = waveform.astype(np.float32) / max_value
    else:
        waveform = waveform.astype(np.float32)
        peak = np.max(np.abs(waveform)) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if source_rate != sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=source_rate,
            target_sr=sample_rate,
        ).astype(np.float32)

    return waveform


def describe_waveform(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, float]:
    """Return basic information useful for debugging audio input."""
    if waveform.size == 0:
        return {
            "sample_rate": float(sample_rate),
            "samples": 0.0,
            "duration_seconds": 0.0,
            "peak_amplitude": 0.0,
            "rms_amplitude": 0.0,
        }

    return {
        "sample_rate": float(sample_rate),
        "samples": float(waveform.size),
        "duration_seconds": float(waveform.size / sample_rate),
        "peak_amplitude": float(np.max(np.abs(waveform))),
        "rms_amplitude": float(np.sqrt(np.mean(np.square(waveform)))),
    }
