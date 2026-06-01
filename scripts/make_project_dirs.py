"""Create local directories used during development."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import TARGET_LABELS

DIRECTORIES = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "external",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "logs",
]


def main() -> int:
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"ready: {directory.relative_to(PROJECT_ROOT)}")

    raw_root = PROJECT_ROOT / "data" / "raw"
    for label in TARGET_LABELS:
        class_dir = raw_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"ready: {class_dir.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
