"""Copy ESC-50 clips into data/raw/ class folders.

Usage:
    python scripts/copy_esc50_clips.py glass_breaking siren
    python scripts/copy_esc50_clips.py normal_background
    python scripts/copy_esc50_clips.py --all
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import TARGET_LABELS

ESC50_ROOT = PROJECT_ROOT / "data" / "ESC-50-master"
ESC50_META = ESC50_ROOT / "meta" / "esc50.csv"
ESC50_AUDIO = ESC50_ROOT / "audio"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"

# ESC-50 target IDs (filename suffix *-{target}.wav) for emergency classes.
ESC50_TARGET_BY_CLASS: dict[str, int] = {
    "glass_breaking": 39,
    "siren": 42,
}

# Diverse everyday sounds — hard negatives, not alarm-like.
ESC50_NORMAL_CATEGORIES: tuple[str, ...] = (
    "rain",
    "sea_waves",
    "wind",
    "footsteps",
    "keyboard_typing",
    "dog",
    "cat",
    "chirping_birds",
    "engine",
    "vacuum_cleaner",
    "washing_machine",
    "crow",
    "crickets",
    "pouring_water",
    "mouse_click",
    "train",
    # Human voice — common in video playback; helps avoid false alarms during TV/speech
    "laughing",
    "crying_baby",
    "sneezing",
    "coughing",
    "breathing",
    # Short transient impacts — similar embedding neighborhood to glass_breaking
    "clapping",
    "door_wood_knock",
    "can_opening",
    "clock_tick",
    # Ambient non-emergency backgrounds
    "crackling_fire",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "classes",
        nargs="*",
        choices=[*TARGET_LABELS, "all"],
        help="Project labels to copy from ESC-50 (or --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Copy glass_breaking, siren, and normal_background.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def copy_file(source: Path, destination: Path, *, dry_run: bool) -> str:
    if destination.exists():
        return "skipped"
    if dry_run:
        print(f"  copy: {source.name} -> {destination.relative_to(PROJECT_ROOT)}")
        return "copied"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"  copy: {source.name}")
    return "copied"


def copy_by_target_id(label: str, target_id: int, *, dry_run: bool) -> dict[str, int]:
    counts = {"copied": 0, "skipped": 0}
    destination_dir = RAW_ROOT / label
    pattern = f"*-{target_id}.wav"
    files = sorted(ESC50_AUDIO.glob(pattern))
    print(f"\n[{label}] target={target_id} ({len(files)} files)")
    for source in files:
        result = copy_file(source, destination_dir / source.name, dry_run=dry_run)
        counts[result] += 1
    return counts


def copy_normal_background(*, dry_run: bool) -> dict[str, int]:
    counts = {"copied": 0, "skipped": 0}
    destination_dir = RAW_ROOT / "normal_background"
    print(f"\n[normal_background] categories: {', '.join(ESC50_NORMAL_CATEGORIES)}")

    with ESC50_META.open(newline="", encoding="utf-8") as meta_file:
        reader = csv.DictReader(meta_file)
        for row in reader:
            if row["category"] not in ESC50_NORMAL_CATEGORIES:
                continue
            source = ESC50_AUDIO / row["filename"]
            if not source.exists():
                print(f"  missing: {row['filename']}")
                continue
            result = copy_file(source, destination_dir / row["filename"], dry_run=dry_run)
            counts[result] += 1

    return counts


def main() -> int:
    args = parse_args()
    if not ESC50_META.exists():
        print(f"ESC-50 not found. Run README Data Setup step 1 first.")
        print(f"Expected: {ESC50_META}")
        return 1

    labels = list(TARGET_LABELS) if args.all else args.classes
    if not labels:
        print("Specify class names or --all")
        return 1

    totals = {"copied": 0, "skipped": 0}
    for label in labels:
        if label == "normal_background":
            counts = copy_normal_background(dry_run=args.dry_run)
        elif label in ESC50_TARGET_BY_CLASS:
            counts = copy_by_target_id(
                label, ESC50_TARGET_BY_CLASS[label], dry_run=args.dry_run
            )
        else:
            print(f"\n[{label}] no ESC-50 mapping (use import_team_audio.py)")
            continue
        for key in totals:
            totals[key] += counts.get(key, 0)

    print(f"\nSummary: {totals}")
    print("Class counts:")
    for name in TARGET_LABELS:
        print(f"  {name}: {len(list((RAW_ROOT / name).glob('*.wav')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
