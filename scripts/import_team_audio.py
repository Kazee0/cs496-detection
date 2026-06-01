"""Import team audio into data/raw/ and normalize to mono 16 kHz WAV.

Examples:
    python scripts/import_team_audio.py --folder scream ~/Downloads/screams \\
        --folder fire_or_smoke_alarm ~/Downloads/smoke_alarm

    python scripts/import_team_audio.py --source ~/Downloads/CS396
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import SAMPLE_RATE, TARGET_LABELS

SOURCE_ALIASES: dict[str, str] = {
    "screams": "scream",
    "scream": "scream",
    "smoke_alarm": "fire_or_smoke_alarm",
    "smoke_alarms": "fire_or_smoke_alarm",
    "fire_or_smoke_alarm": "fire_or_smoke_alarm",
    "alarm": "fire_or_smoke_alarm",
    "alarms": "fire_or_smoke_alarm",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".wma"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Parent folder with class subfolders (e.g. CS396/screams, CS396/smoke_alarm).",
    )
    parser.add_argument(
        "--folder",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        default=[],
        help="Import one class folder, e.g. --folder scream ~/Downloads/screams",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files.",
    )
    return parser.parse_args()


def resolve_label(name: str) -> str | None:
    key = name.strip().lower()
    if key in SOURCE_ALIASES:
        return SOURCE_ALIASES[key]
    if key in TARGET_LABELS:
        return key
    return None


def list_audio_files(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and not path.name.startswith(".")
        and "__MACOSX" not in path.parts
    ]
    return sorted(files)


def load_mono_16k(path: Path) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32)


def write_wav(path: Path, waveform: np.ndarray) -> None:
    peak = np.max(np.abs(waveform)) if waveform.size else 0.0
    if peak > 1.0:
        waveform = waveform / peak
    pcm = np.clip(waveform, -1.0, 1.0)
    wavfile.write(path, SAMPLE_RATE, (pcm * 32767).astype(np.int16))


def import_file(source: Path, destination_dir: Path, *, dry_run: bool) -> str:
    destination = destination_dir / f"{source.stem}.wav"
    if destination.exists():
        return "skipped"

    if dry_run:
        action = "converted" if source.suffix.lower() != ".wav" else "imported"
        print(f"  {action}: {source.name} -> {destination.name}")
        return action

    destination_dir.mkdir(parents=True, exist_ok=True)
    waveform = load_mono_16k(source)
    write_wav(destination, waveform)
    action = "converted" if source.suffix.lower() != ".wav" else "imported"
    print(f"  {action}: {source.name} -> {destination.name}")
    return action


def import_folder(label: str, folder: Path, raw_root: Path, *, dry_run: bool) -> dict[str, int]:
    counts = {"imported": 0, "converted": 0, "skipped": 0}
    resolved = resolve_label(label)
    if resolved is None:
        print(f"Unknown label: {label}")
        return counts

    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        return counts

    destination_dir = raw_root / resolved
    audio_files = list_audio_files(folder)
    print(f"\n[{resolved}] {len(audio_files)} file(s) from {folder}")

    for audio_path in audio_files:
        result = import_file(audio_path, destination_dir, dry_run=dry_run)
        counts[result] = counts.get(result, 0) + 1

    return counts


def main() -> int:
    args = parse_args()
    raw_root = PROJECT_ROOT / "data" / "raw"
    totals = {"imported": 0, "converted": 0, "skipped": 0}

    if not args.folder and args.source is None:
        print("Provide --source or at least one --folder LABEL PATH")
        return 1

    for label, path_str in args.folder:
        folder = Path(path_str).expanduser().resolve()
        counts = import_folder(label, folder, raw_root, dry_run=args.dry_run)
        for key in totals:
            totals[key] += counts.get(key, 0)

    if args.source is not None:
        source_root = args.source.expanduser().resolve()
        if not source_root.is_dir():
            print(f"Source not found: {source_root}")
            return 1
        for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            label = resolve_label(class_dir.name)
            if label is None:
                print(f"skip unknown folder: {class_dir.name}")
                continue
            counts = import_folder(label, class_dir, raw_root, dry_run=args.dry_run)
            for key in totals:
                totals[key] += counts.get(key, 0)

    print("\nSummary:", totals)
    print("Class counts:")
    for name in TARGET_LABELS:
        wav_count = len(list((raw_root / name).glob("*.wav")))
        print(f"  {name}: {wav_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
