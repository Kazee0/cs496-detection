"""List audio input devices available to sounddevice."""

from __future__ import annotations

import sounddevice as sd


def main() -> int:
    devices = sd.query_devices()
    input_devices = [
        (index, device)
        for index, device in enumerate(devices)
        if device.get("max_input_channels", 0) > 0
    ]

    if not input_devices:
        print("No audio input devices found.")
        return 1

    print("Audio input devices:")
    for index, device in input_devices:
        channels = device["max_input_channels"]
        sample_rate = int(device["default_samplerate"])
        print(f"[{index}] {device['name']} ({channels} channels, default {sample_rate} Hz)")

    default_input = sd.default.device[0]
    print()
    print(f"Default input device index: {default_input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
