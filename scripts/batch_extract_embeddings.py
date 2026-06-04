"""Extract YAMNet embeddings from all WAV files in labeled class folders.

Usage:
    python scripts/batch_extract_embeddings.py

Expects data/raw/<class_name>/*.wav folders.
Saves embeddings and labels to data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.audio import load_wav_mono
from emergency_detection.config import ALERT_LABELS, SAMPLE_RATE
from emergency_detection.inference import energy_weighted_embedding
from emergency_detection.yamnet import load_yamnet_model, run_yamnet

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RNG = np.random.default_rng(42)


def _match_length(waveform: np.ndarray, target_length: int) -> np.ndarray:
    if waveform.size == target_length:
        return waveform.astype(np.float32)
    if waveform.size > target_length:
        return waveform[:target_length].astype(np.float32)
    return np.pad(waveform, (0, target_length - waveform.size)).astype(np.float32)


def augment_waveform(waveform: np.ndarray, variant_index: int) -> np.ndarray:
    """Return a lightweight emergency-audio augmentation."""
    original_length = waveform.size
    variant = variant_index % 3

    if variant == 0:
        shifted = librosa.effects.pitch_shift(
            y=waveform.astype(np.float32),
            sr=SAMPLE_RATE,
            n_steps=float(RNG.choice([-1.0, -0.5, 0.5, 1.0])),
        )
        augmented = shifted
    elif variant == 1:
        stretched = librosa.effects.time_stretch(
            y=waveform.astype(np.float32),
            rate=float(RNG.choice([0.92, 0.96, 1.04, 1.08])),
        )
        augmented = stretched
    else:
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        noise_rms = max(rms * 0.08, 0.002)
        noise = RNG.normal(0.0, noise_rms, size=waveform.shape).astype(np.float32)
        augmented = waveform.astype(np.float32) + noise

    augmented = _match_length(np.asarray(augmented, dtype=np.float32), original_length)
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


def extract_embedding(waveform: np.ndarray, model) -> np.ndarray:
    output = run_yamnet(waveform, yamnet_model=model)
    return energy_weighted_embedding(waveform, output.embeddings)


def main() -> int:
    class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"No class folders found in {RAW_DIR}")
        return 1

    class_names = [d.name for d in class_dirs]
    class_counts = {class_dir.name: len(list(class_dir.glob("*.wav"))) for class_dir in class_dirs}
    emergency_counts = [
        count for name, count in class_counts.items()
        if name in ALERT_LABELS and count > 0
    ]
    emergency_target_count = max(emergency_counts) if emergency_counts else 0

    print(f"Classes found: {class_names}")
    if emergency_target_count:
        print(f"Emergency augmentation target: {emergency_target_count} samples/class")
    print("Loading YAMNet...")
    model = load_yamnet_model()

    all_embeddings = []
    all_labels = []

    for label_index, class_dir in enumerate(class_dirs):
        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"\n[{class_dir.name}] {len(wav_files)} files")

        class_waveforms: list[tuple[Path, np.ndarray]] = []
        for wav_path in wav_files:
            try:
                waveform = load_wav_mono(wav_path)
                class_waveforms.append((wav_path, waveform))
                all_embeddings.append(extract_embedding(waveform, model))
                all_labels.append(label_index)
                print(f"  ok: {wav_path.name}")
            except Exception as exc:
                print(f"  skip: {wav_path.name} ({exc})")

        if class_dir.name in ALERT_LABELS and class_waveforms:
            needed = max(0, emergency_target_count - len(class_waveforms))
            for aug_index in range(needed):
                wav_path, waveform = class_waveforms[aug_index % len(class_waveforms)]
                try:
                    augmented = augment_waveform(waveform, aug_index)
                    all_embeddings.append(extract_embedding(augmented, model))
                    all_labels.append(label_index)
                    print(f"  aug: {wav_path.stem}#{aug_index + 1}")
                except Exception as exc:
                    print(f"  aug skip: {wav_path.name} ({exc})")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "embeddings.npy", embeddings_array)
    np.save(PROCESSED_DIR / "labels.npy", labels_array)

    class_names_path = PROCESSED_DIR / "class_names.txt"
    class_names_path.write_text("\n".join(class_names))

    print(f"\nDone.")
    print(f"  embeddings: {embeddings_array.shape}")
    print(f"  labels:     {labels_array.shape}")
    print(f"  classes:    {class_names}")
    print(f"  saved to:   {PROCESSED_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
