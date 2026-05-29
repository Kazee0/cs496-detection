# Sound-Based Emergency Event Detection

Team 2: Irena Liu, Andrew Smart, Jiadong Li

## Overview

This project plans to build a smart sensing system that detects emergency sound events before a user visually notices them. The system will listen through a microphone, classify short audio windows in real time, and notify the user when emergency-like sounds are detected.

Target events include:

- Fire alarm or smoke alarm
- Siren
- Glass breaking
- Screaming
- Loud impact sounds
- Normal background noise

The system is intended to improve smart home safety, support elderly people or people living alone, and help users with hearing difficulties.

## Planned System

The planned pipeline is:

```text
Microphone -> Audio Windows -> Mel Spectrogram / YAMNet Embeddings -> Classifier -> UI Alert
```

The sensing component will use a laptop or phone microphone to capture a continuous audio stream. Audio will be processed in fixed-size windows, with an initial target of 1 second per window at a 16 kHz sample rate.

The machine learning component will perform multi-class emergency sound classification. The proposal uses YAMNet as the pretrained audio model because it is trained on AudioSet and can produce useful audio embeddings for downstream classification.

## Project Setup

This repository is organized as a Python project for audio capture, model training, and a live dashboard demo.

Recommended setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\make_project_dirs.py
python scripts\check_environment.py
python scripts\check_microphone.py
```

Project structure:

```text
data/                  Local audio data and dataset notes
models/                Trained model artifacts
notebooks/             Experiments and analysis
scripts/               Utility scripts
src/emergency_detection/
                       Reusable project code
```

Large local audio files and trained model artifacts are ignored by git by default.

## Data Setup

Audio datasets are not stored in the repository. Run the following steps to set up the training data locally after cloning.

**1. Download ESC-50**

```bash
cd data
curl -L -o ESC-50.zip https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
unzip ESC-50.zip
cd ..
```

**2. Copy siren and glass breaking clips**

```bash
cp data/ESC-50-master/audio/*-39.wav data/raw/glass_breaking/
cp data/ESC-50-master/audio/*-42.wav data/raw/siren/
```

**3. Add fire/smoke alarm and scream clips**

Download WAV files from FreeSound or a similar source and place them in:

```text
data/raw/fire_or_smoke_alarm/
data/raw/scream/
```

**4. Record background noise**

```bash
python scripts/record_background.py
```

Stay quiet and let it record 40 seconds of ambient room noise. Move around, type, or talk normally — varied background noise improves the model.

**5. Create class folders (if needed)**

```bash
python scripts/make_project_dirs.py
```

**6. Extract embeddings and train the classifier**

```bash
python scripts/batch_extract_embeddings.py
python scripts/train_classifier.py
```

**7. Evaluate the model**

```bash
python scripts/evaluate_classifier.py --plot
python scripts/benchmark_detection_latency.py
```

**8. Run live detection (terminal)**

```bash
python scripts/live_detection.py
```

Press `Ctrl+C` to stop.

## ML Plan

Initial model design:

```text
Audio Waveform -> Pretrained YAMNet -> Audio Embeddings -> Classification Layers -> Event Label
```

Planned classes:

- Fire alarm / smoke alarm
- Siren
- Glass breaking
- Scream
- Normal background noise

Implementation steps:

1. Collect or select audio samples for each target class.
2. Convert audio to the input format required by YAMNet.
3. Extract embeddings from YAMNet.
4. Train lightweight classification layers on top of the embeddings.
5. Evaluate classification quality on held-out audio clips.
6. Tune confidence thresholds to reduce false alarms during live use.

## Sensing Plan

The sensing component will capture live audio from the microphone and prepare it for model inference.

Planned preprocessing:

- Capture audio at 16 kHz.
- Split continuous audio into fixed-size overlapping windows.
- Convert each window into a mel spectrogram or YAMNet-compatible waveform input.
- Normalize input to improve consistency.
- Send each processed window to the model.

The live system should prioritize low latency, stable microphone capture, and avoiding repeated duplicate alerts for the same sound event.

## Demo Plan

The live demo will run on a laptop with a dashboard interface.

Demo components:

- Laptop running the dashboard
- Laptop or phone microphone
- Sound sources, such as smoke alarm audio and background noise
- Real-time waveform visualization
- Visual and/or sound notification when an emergency event is detected

Demo workflow:

```text
Microphone -> Mel Spectrogram -> Model -> UI Alert
```

Expected demo behavior:

1. The dashboard shows the incoming audio waveform.
2. The system classifies recent audio windows in real time.
3. When an emergency class exceeds the alert threshold, the UI displays a warning.
4. Background noise should be classified as normal and should not trigger alerts.

## Evaluation Plan

Scripts implement the metrics below on a held-out 20% test split (same seed as training).

| Metric | Script |
|--------|--------|
| Accuracy, per-class P/R/F1, confusion matrix | `scripts/evaluate_classifier.py` |
| Emergency-class precision/recall (grouped) | `scripts/evaluate_classifier.py` |
| Normal false positive rate (incl. alert threshold) | `scripts/evaluate_classifier.py --threshold 0.75` |
| Detection latency per 1s window | `scripts/benchmark_detection_latency.py` |

Outputs: `outputs/evaluation_report.json`, optional `outputs/confusion_matrix.png`, `outputs/latency_ms.txt`.

Suggested test cases:

- Clean emergency audio clips
- Emergency sounds mixed with background noise
- Normal indoor background noise
- Speech or music that should not trigger alerts
- Multiple volume levels and distances from the microphone

## Implementation Roadmap

### Phase 1: Project Setup

- Create the project structure.
- Decide the runtime stack for model training and live demo.
- Prepare a small labeled audio dataset.
- Confirm microphone capture works locally.

### Phase 2: Baseline Model

- Load YAMNet.
- Extract embeddings from labeled audio samples.
- Train a baseline classifier.
- Save the trained classifier for inference.

Current Phase 2 utilities:

```powershell
.\.venv\Scripts\python.exe scripts\yamnet_smoke_test.py
.\.venv\Scripts\python.exe scripts\extract_yamnet_embeddings.py path\to\audio.wav --output outputs\sample_embeddings.npy
```

The smoke test loads YAMNet from TensorFlow Hub and runs inference on a synthetic 1 second waveform. The extraction script loads a WAV file, converts it to mono 16 kHz audio, runs YAMNet, prints the top AudioSet classes, and optionally saves the embedding matrix as `.npy`.

### Phase 3: Real-Time Detection

- Implement microphone streaming.
- Process audio in 1 second overlapping windows.
- Run model inference on each window.
- Apply confidence thresholds and cooldown logic for alerts.

### Phase 4: Dashboard

- Show live waveform visualization.
- Display current predicted class and confidence.
- Show visible alert state for emergency events.
- Add simple logging for detected events.

### Phase 5: Testing and Demo Polish

- [x] Held-out evaluation: `scripts/evaluate_classifier.py`
- [x] Latency benchmark: `scripts/benchmark_detection_latency.py`
- Test live detection with demo sound sources.
- Tune thresholds for fewer false positives.
- Prepare final demo flow and presentation notes.

## Repository Status

| Area | Status |
|------|--------|
| Data setup | Manual ESC-50 download + copy (`README` Data Setup); `record_background.py` for normal class |
| Embeddings / train | `batch_extract_embeddings.py`, `train_classifier.py` |
| Live detection | `live_detection.py` (mic + classifier + cooldown) |
| Evaluation | `evaluate_classifier.py`, `benchmark_detection_latency.py` |
| Dashboard POC | Stashed locally (`git stash list` → `WIP: dashboard POC UI`) |

ESC-50 on `main` uses category-specific filenames (`*-39.wav` glass, `*-42.wav` siren) instead of the removed `prepare_esc50_subset.py` script (fold-based train/val/test splits).
