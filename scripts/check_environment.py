"""Check whether the local development environment is ready."""

from __future__ import annotations

import importlib.util
import sys


REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sounddevice", "sounddevice"),
    ("librosa", "librosa"),
    ("tensorflow", "tensorflow"),
    ("tensorflow_hub", "tensorflow-hub"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("streamlit", "streamlit"),
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")

    missing: list[str] = []
    for module_name, package_name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
            print(f"[missing] {package_name}")
        else:
            print(f"[ok]      {package_name}")

    if missing:
        print()
        print("Install missing packages with:")
        print("python -m pip install -r requirements.txt")
        return 1

    print()
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
