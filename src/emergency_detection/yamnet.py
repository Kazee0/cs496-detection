"""YAMNet embedding extraction and baseline inference helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"


@dataclass(frozen=True)
class YamnetOutput:
    """Structured output from one YAMNet model call."""

    scores: np.ndarray
    embeddings: np.ndarray
    spectrogram: np.ndarray
    class_names: list[str]


def load_yamnet_model():
    """Load the pretrained YAMNet model from TensorFlow Hub."""
    import tensorflow_hub as hub

    return hub.load(YAMNET_MODEL_HANDLE)


def load_yamnet_class_names(yamnet_model) -> list[str]:
    """Read AudioSet display names bundled with the TensorFlow Hub model."""
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as class_map_file:
        reader = csv.DictReader(class_map_file)
        return [row["display_name"] for row in reader]


def run_yamnet(waveform: np.ndarray, yamnet_model=None) -> YamnetOutput:
    """Run YAMNet on a mono waveform and return scores, embeddings, and spectrogram."""
    import tensorflow as tf

    model = yamnet_model or load_yamnet_model()
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform_tensor)

    return YamnetOutput(
        scores=scores.numpy(),
        embeddings=embeddings.numpy(),
        spectrogram=spectrogram.numpy(),
        class_names=load_yamnet_class_names(model),
    )


def summarize_top_classes(output: YamnetOutput, limit: int = 5) -> list[tuple[str, float]]:
    """Return top YAMNet AudioSet classes averaged across model frames."""
    mean_scores = output.scores.mean(axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:limit]
    return [(output.class_names[index], float(mean_scores[index])) for index in top_indices]
