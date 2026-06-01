"""Extract YAMNet embeddings from all WAV files in labeled class folders.

Usage:
    python scripts/batch_extract_embeddings.py

Expects data/raw/<class_name>/*.wav folders.
Saves embeddings and labels to data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.audio import load_wav_mono
from emergency_detection.inference import energy_weighted_embedding
from emergency_detection.yamnet import load_yamnet_model, run_yamnet

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> int:
    class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"No class folders found in {RAW_DIR}")
        return 1

    class_names = [d.name for d in class_dirs]
    print(f"Classes found: {class_names}")
    print("Loading YAMNet...")
    model = load_yamnet_model()

    all_embeddings = []
    all_labels = []

    for label_index, class_dir in enumerate(class_dirs):
        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"\n[{class_dir.name}] {len(wav_files)} files")

        for wav_path in wav_files:
            try:
                waveform = load_wav_mono(wav_path)
                output = run_yamnet(waveform, yamnet_model=model)
                embedding = energy_weighted_embedding(waveform, output.embeddings)
                all_embeddings.append(embedding)
                all_labels.append(label_index)
                print(f"  ok: {wav_path.name}")
            except Exception as exc:
                print(f"  skip: {wav_path.name} ({exc})")

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
