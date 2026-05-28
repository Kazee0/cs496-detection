"""Shared configuration for audio capture and classification."""

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SECONDS = 0.5
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)

TARGET_LABELS = [
    "fire_or_smoke_alarm",
    "siren",
    "glass_breaking",
    "scream",
    "normal_background",
]

ALERT_LABELS = {
    "fire_or_smoke_alarm",
    "siren",
    "glass_breaking",
    "scream",
}

DEFAULT_ALERT_THRESHOLD = 0.75
DEFAULT_ALERT_COOLDOWN_SECONDS = 3.0
