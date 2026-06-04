"""Mac-friendly Streamlit dashboard with isolated model execution.

Usage:
    streamlit run dashboard_mac.py
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import (
    ALERT_LABELS,
    DEFAULT_ALERT_COOLDOWN_SECONDS,
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_MIN_SIGNAL_RMS,
    HOP_SAMPLES,
    LABEL_ALERT_THRESHOLDS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from emergency_detection.model_worker import run_model_worker

MODELS_DIR = PROJECT_ROOT / "models"
REFRESH_SECONDS = 0.35
WAVEFORM_SECONDS = 3
WAVEFORM_MAXLEN = SAMPLE_RATE * WAVEFORM_SECONDS

LABEL_DISPLAY = {
    "fire_or_smoke_alarm": "Fire / Smoke Alarm",
    "scream": "Scream",
    "siren": "Siren",
    "glass_breaking": "Glass Breaking",
    "normal_background": "Normal Background",
}


def make_state() -> dict:
    return {
        "running": False,
        "status": "Ready.",
        "error": "",
        "label": "normal_background",
        "confidence": 0.0,
        "proba": {},
        "waveform": deque(maxlen=WAVEFORM_MAXLEN),
        "events": deque(maxlen=20),
        "audio_chunks": 0,
        "predictions": 0,
        "worker_started_at": 0.0,
        "last_prediction_at": "Never",
        "last_alert_at": 0.0,
    }


def init_session() -> None:
    if "mac_dash" not in st.session_state:
        st.session_state.mac_dash = make_state()
        st.session_state.audio_thread = None
        st.session_state.result_thread = None
        st.session_state.model_process = None
        st.session_state.model_input = None
        st.session_state.model_output = None


def stop_detection() -> None:
    state = st.session_state.mac_dash
    state["running"] = False
    state["status"] = "Stopped."

    input_queue = st.session_state.model_input
    if input_queue is not None:
        try:
            input_queue.put_nowait(None)
        except Exception:
            pass

    process = st.session_state.model_process
    if process is not None and process.is_alive():
        process.terminate()


def start_detection(threshold: float, cooldown: float, min_signal_rms: float) -> None:
    state = st.session_state.mac_dash
    if state["running"]:
        return

    state.update({
        "running": True,
        "status": "Starting isolated YAMNet worker...",
        "error": "",
        "worker_started_at": time.time(),
        "predictions": 0,
    })

    ctx = mp.get_context("spawn")
    st.session_state.model_input = ctx.Queue(maxsize=4)
    st.session_state.model_output = ctx.Queue(maxsize=16)
    st.session_state.model_process = ctx.Process(
        target=run_model_worker,
        args=(
            st.session_state.model_input,
            st.session_state.model_output,
            str(MODELS_DIR),
            min_signal_rms,
        ),
        daemon=True,
        name="mac-yamnet-worker",
    )
    st.session_state.model_process.start()
    input_queue = st.session_state.model_input
    output_queue = st.session_state.model_output
    process = st.session_state.model_process

    def audio_loop() -> None:
        try:
            import sounddevice as sd

            buffer: deque[float] = deque(maxlen=WINDOW_SAMPLES)
            samples_since_hop = 0

            def callback(indata, frames, time_info, status) -> None:
                nonlocal samples_since_hop
                if status:
                    state["status"] = f"Audio status: {status}"
                chunk = indata[:, 0].astype(np.float32)
                buffer.extend(chunk)
                state["waveform"].extend(chunk)
                state["audio_chunks"] += 1
                samples_since_hop += len(chunk)

                if samples_since_hop < HOP_SAMPLES or len(buffer) < WINDOW_SAMPLES:
                    return

                samples_since_hop = 0
                try:
                    input_queue.put_nowait(np.array(buffer, dtype=np.float32))
                except queue.Full:
                    pass

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP_SAMPLES,
                callback=callback,
            ):
                state["status"] = "Microphone running; waiting for predictions..."
                while state["running"]:
                    time.sleep(0.05)
        except Exception:
            state["error"] = traceback.format_exc()
            state["status"] = "Audio failed."
            state["running"] = False

    def result_loop() -> None:
        while state["running"]:
            try:
                message = output_queue.get(timeout=0.5)
            except queue.Empty:
                if process.exitcode is not None and process.exitcode != 0:
                    state["status"] = f"Worker exited with code {process.exitcode}."
                    state["running"] = False
                elif state["predictions"] == 0 and time.time() - state["worker_started_at"] > 25:
                    state["status"] = "YAMNet worker timed out."
                    state["error"] = (
                        "The UI is responsive, but TensorFlow/YAMNet did not return from "
                        "the isolated worker within 25 seconds."
                    )
                    state["running"] = False
                continue

            msg_type = message.get("type")
            if msg_type == "status":
                state["status"] = message.get("message", "Worker starting.")
            elif msg_type == "ready":
                state["status"] = "Worker ready."
            elif msg_type == "error":
                state["status"] = message.get("message", "Worker failed.")
                state["error"] = message.get("traceback", "")
                state["running"] = False
            elif msg_type == "prediction":
                label = message["label"]
                confidence = float(message["confidence"])
                now = time.time()
                is_alert = (
                    label in ALERT_LABELS
                    and confidence >= LABEL_ALERT_THRESHOLDS.get(label, threshold)
                    and now - state["last_alert_at"] >= cooldown
                )
                state["label"] = label
                state["confidence"] = confidence
                state["proba"] = message["proba"]
                state["predictions"] += 1
                state["last_prediction_at"] = time.strftime("%H:%M:%S")
                state["status"] = "Listening."
                if is_alert:
                    state["last_alert_at"] = now
                    state["events"].appendleft({
                        "Time": state["last_prediction_at"],
                        "Event": LABEL_DISPLAY.get(label, label),
                        "Confidence": f"{confidence:.0%}",
                    })

    st.session_state.audio_thread = threading.Thread(target=audio_loop, daemon=True)
    st.session_state.result_thread = threading.Thread(target=result_loop, daemon=True)
    st.session_state.audio_thread.start()
    st.session_state.result_thread.start()


def render() -> None:
    init_session()
    state = st.session_state.mac_dash

    st.set_page_config(page_title="Emergency Sound Detection", layout="wide")
    st.title("Emergency Sound Detection")

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Alert threshold", 0.0, 1.0, DEFAULT_ALERT_THRESHOLD, 0.05)
        cooldown = st.slider("Alert cooldown (s)", 0.0, 10.0, DEFAULT_ALERT_COOLDOWN_SECONDS, 0.5)
        min_signal_rms = st.slider("Min signal RMS", 0.0, 0.10, DEFAULT_MIN_SIGNAL_RMS, 0.005)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Listening", disabled=state["running"], use_container_width=True):
            start_detection(threshold, cooldown, min_signal_rms)
            st.rerun()
    with col2:
        if st.button("Stop", disabled=not state["running"], use_container_width=True):
            stop_detection()
            st.rerun()
    with col3:
        st.metric("Predictions", state["predictions"])

    st.info(state["status"])
    if state["error"]:
        st.error("Background worker reported a problem.")
        st.code(state["error"], language="text")

    label = state["label"]
    confidence = state["confidence"]
    st.metric("Current Detection", LABEL_DISPLAY.get(label, label), f"{confidence:.0%}")

    waveform = np.array(state["waveform"], dtype=np.float32)
    if waveform.size:
        step = max(1, waveform.size // 800)
        st.line_chart(waveform[::step], height=160)
    else:
        st.caption("Waveform will appear after listening starts.")

    if state["proba"]:
        rows = [
            {"Class": LABEL_DISPLAY.get(k, k), "Confidence": float(v)}
            for k, v in state["proba"].items()
        ]
        st.bar_chart(pd.DataFrame(rows), x="Class", y="Confidence", height=240)

    st.subheader("Event Log")
    events = list(state["events"])
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts yet.")

    if state["running"]:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    render()
