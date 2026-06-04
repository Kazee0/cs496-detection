"""Live emergency sound detection dashboard.

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import joblib
import streamlit as st

TELEGRAM_TOKEN = "8783807236:AAFdtEk41BcHM4gsqkCUQbhyD_pantKxt1c"
TELEGRAM_CHAT_ID = "7701400056"


def send_telegram_alert(label: str, confidence: float) -> None:
    msg = f"🚨 ALERT: {label.replace('_', ' ').title()} detected ({confidence:.0%})"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=5,
        )
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
    ALERT_LABELS,
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_ALERT_COOLDOWN_SECONDS,
)
from emergency_detection.yamnet import load_yamnet_model, run_yamnet
from emergency_detection.inference import energy_weighted_embedding

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Emergency Sound Detection", layout="wide")


@st.cache_resource
def load_models():
    model_path = MODELS_DIR / "classifier.pkl"
    if not model_path.exists():
        st.error("No trained model found. Run scripts/train_classifier.py first.")
        st.stop()
    data = joblib.load(model_path)
    yamnet = load_yamnet_model()
    return yamnet, data["classifier"], data["scaler"], data["class_names"]


@st.cache_resource
def get_shared_state():
    return {
        "lock": threading.Lock(),
        "running": False,
        "current_label": "normal_background",
        "current_confidence": 0.0,
        "waveform": np.zeros(WINDOW_SAMPLES, dtype=np.float32),
        "is_alert": False,
        "last_alert_time": 0.0,
        "event_log": [],
        "stream": None,
    }


yamnet, clf, scaler, class_names = load_models()
state = get_shared_state()

st.sidebar.title("Settings")
threshold = st.sidebar.slider("Alert threshold", 0.5, 1.0, DEFAULT_ALERT_THRESHOLD, 0.05)
cooldown = st.sidebar.slider("Alert cooldown (s)", 1.0, 10.0, DEFAULT_ALERT_COOLDOWN_SECONDS, 0.5)
consecutive = st.sidebar.slider("Consecutive windows", 1, 5, 3, 1,
                                 help="How many consecutive detections before alerting.")

devices = sd.query_devices()
input_devices = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
device_labels = [f"[{i}] {name}" for i, name in input_devices]
device_indices = [i for i, _ in input_devices]
default_idx = next(
    (i for i, (_, name) in enumerate(input_devices) if "macbook" in name.lower() or "built-in" in name.lower()), 0)
selected = st.sidebar.selectbox("Microphone", device_labels, index=default_idx)
selected_device = device_indices[device_labels.index(selected)]

st.title("Emergency Sound Detection")

col_start, col_stop = st.columns(2)


def start_stream():
    buffer = deque(maxlen=WINDOW_SAMPLES)
    samples_since_hop = [0]
    pending_label = [None]
    pending_count = [0]

    def audio_callback(indata, frames, time_info, cb_status):
        chunk = indata[:, 0].astype(np.float32)
        buffer.extend(chunk)
        samples_since_hop[0] += len(chunk)

        if len(buffer) < WINDOW_SAMPLES or samples_since_hop[0] < HOP_SAMPLES:
            return
        samples_since_hop[0] = 0

        waveform = np.array(buffer, dtype=np.float32)
        output = run_yamnet(waveform, yamnet_model=yamnet)
        embedding = energy_weighted_embedding(waveform, output.embeddings)
        x = scaler.transform(embedding.reshape(1, -1))
        proba = clf.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = class_names[idx]
        confidence = float(proba[idx])

        is_alert_candidate = label in ALERT_LABELS and confidence >= threshold
        if is_alert_candidate and label == pending_label[0]:
            pending_count[0] += 1
        elif is_alert_candidate:
            pending_label[0] = label
            pending_count[0] = 1
        else:
            pending_label[0] = None
            pending_count[0] = 0

        now = time.time()
        is_alert = is_alert_candidate and pending_count[0] >= consecutive
        cooldown_ok = (now - state["last_alert_time"]) >= cooldown

        with state["lock"]:
            state["waveform"] = waveform
            state["current_label"] = label
            state["current_confidence"] = confidence
            if is_alert and cooldown_ok:
                state["is_alert"] = True
                state["last_alert_time"] = now
                state["event_log"].append({
                    "Time": time.strftime("%H:%M:%S"),
                    "Event": label.replace("_", " ").title(),
                    "Confidence": f"{confidence:.0%}",
                })
                send_telegram_alert(label, confidence)
            else:
                state["is_alert"] = (now - state["last_alert_time"]) < cooldown

    stream = sd.InputStream(
        device=selected_device,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=HOP_SAMPLES,
        callback=audio_callback,
    )
    stream.start()
    with state["lock"]:
        state["stream"] = stream
        state["running"] = True


def stop_stream():
    with state["lock"]:
        if state["stream"] is not None:
            state["stream"].stop()
            state["stream"].close()
            state["stream"] = None
        state["running"] = False
        state["is_alert"] = False


with col_start:
    if st.button("Start Listening", disabled=state["running"]):
        start_stream()
        st.rerun()

with col_stop:
    if st.button("Stop", disabled=not state["running"]):
        stop_stream()
        st.rerun()

st.divider()

with state["lock"]:
    label = state["current_label"]
    confidence = state["current_confidence"]
    is_alert = state["is_alert"]
    waveform = state["waveform"].copy()
    event_log = list(state["event_log"])
    running = state["running"]

if is_alert:
    st.error(f"ALERT: {label.replace('_', ' ').upper()} detected!")
elif running:
    st.success("Listening — no emergency detected")
else:
    st.info("Not running — press Start Listening")

st.metric(label="Current Detection",
          value=label.replace("_", " ").title(),
          delta=f"{confidence:.0%} confidence")

st.line_chart(waveform, height=150, width="stretch")

st.subheader("Event Log")
if event_log:
    st.table(event_log[-10:][::-1])
else:
    st.caption("No events yet.")

if running:
    time.sleep(0.5)
    st.rerun()
