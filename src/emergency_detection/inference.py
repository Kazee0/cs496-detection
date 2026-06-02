"""Shared inference logic: energy-weighted embedding + classifier."""

from __future__ import annotations

import numpy as np

from emergency_detection.config import SAMPLE_RATE

# YAMNet frame parameters (fixed by the model architecture)
_YAMNET_FRAME_SECONDS = 0.96
_YAMNET_HOP_SECONDS = 0.48


def waveform_rms(waveform: np.ndarray) -> float:
    """Return RMS amplitude for a float waveform."""
    if waveform.size == 0:
        return 0.0
    samples = waveform.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(np.square(samples))))


def normal_background_proba(class_names: list[str]) -> dict[str, float]:
    """Return a deterministic normal-background probability vector."""
    proba = {name: 0.0 for name in class_names}
    if "normal_background" in proba:
        proba["normal_background"] = 1.0
    return proba


def energy_weighted_embedding(waveform: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Average YAMNet frame embeddings weighted by each frame's RMS energy.

    ESC-50 glass_breaking clips typically have 60-90% silent frames — simple
    mean averaging drowns the actual glass sound in silence, pulling the
    embedding toward normal_background. Energy weighting lets the loud frames
    dominate, giving glass_breaking a distinct representation.
    """
    hop = int(_YAMNET_HOP_SECONDS * SAMPLE_RATE)
    fsize = int(_YAMNET_FRAME_SECONDS * SAMPLE_RATE)
    n_frames = embeddings.shape[0]

    energies = np.array([
        float(np.sqrt(np.mean(waveform[i * hop: min(i * hop + fsize, len(waveform))] ** 2)))
        for i in range(n_frames)
    ], dtype=np.float32)

    total = energies.sum()
    if total > 1e-8:
        weights = energies / total
        return (embeddings * weights[:, np.newaxis]).sum(axis=0)
    return embeddings.mean(axis=0)


def classify_window(
    waveform: np.ndarray,
    yamnet_model,
    clf,
    scaler,
    class_names: list[str],
    *,
    min_signal_rms: float | None = None,
) -> tuple[str, float, dict[str, float]]:
    """Run full inference on a 1-second audio window.

    Returns (label, confidence, per_class_proba).

    Uses energy-weighted embedding aggregation to suppress silent frames in
    training clips (e.g. glass_breaking ESC-50 clips are 60-90% silence),
    giving each class a more representative embedding.
    """
    from emergency_detection.yamnet import run_yamnet

    if min_signal_rms is not None and waveform_rms(waveform) < min_signal_rms:
        return "normal_background", 1.0, normal_background_proba(class_names)

    output = run_yamnet(waveform, yamnet_model=yamnet_model)
    embedding = energy_weighted_embedding(waveform, output.embeddings)

    x = scaler.transform(embedding.reshape(1, -1))
    raw_proba = clf.predict_proba(x)[0]

    proba: dict[str, float] = {name: 0.0 for name in class_names}
    for col_i, class_i in enumerate(clf.classes_):
        if 0 <= int(class_i) < len(class_names):
            proba[class_names[int(class_i)]] = float(raw_proba[col_i])

    best_col = int(np.argmax(raw_proba))
    best_class = int(clf.classes_[best_col])
    label = class_names[best_class] if 0 <= best_class < len(class_names) else "—"
    confidence = float(raw_proba[best_col])

    return label, confidence, proba
