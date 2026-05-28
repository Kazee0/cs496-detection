# Data

Audio data should be organized by split and class:

```text
data/
  raw/
    train/
      fire_or_smoke_alarm/
      siren/
      glass_breaking/
      scream/
      normal_background/
    val/
      ...
    test/
      ...
  processed/
  external/
```

Use short `.wav` clips when possible. The model pipeline will resample audio to 16 kHz and process it in 1 second windows.

Large audio files are ignored by git. Keep dataset source notes in this folder so the team can reproduce the data collection process.
