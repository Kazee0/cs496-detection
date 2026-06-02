"""Streamlit real-time emergency sound detection dashboard.

Usage:
    streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sounddevice as sd
import streamlit as st
import altair as alt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergency_detection.config import (
    ALERT_LABELS,
    DEFAULT_ALERT_COOLDOWN_SECONDS,
    DEFAULT_ALERT_CONSECUTIVE_WINDOWS,
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_MIN_SIGNAL_RMS,
    HOP_SAMPLES,
    LABEL_ALERT_THRESHOLDS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from emergency_detection.inference import classify_window, waveform_rms
from emergency_detection.telegram_notifications import TelegramNotifier
from emergency_detection.yamnet import load_yamnet_model

MODELS_DIR = PROJECT_ROOT / "models"
WAVEFORM_SECONDS = 3
WAVEFORM_MAXLEN = SAMPLE_RATE * WAVEFORM_SECONDS
DISPLAY_POINTS = 800
REFRESH_INTERVAL_S = 0.3
LOG_MAXLEN = 15
ALERT_DISPLAY_HOLD_SECONDS = 4.0

LABEL_DISPLAY: dict[str, str] = {
    "fire_or_smoke_alarm": "Fire / Smoke Alarm",
    "scream": "Scream",
    "siren": "Siren",
    "glass_breaking": "Glass Breaking",
    "normal_background": "Normal Background",
}
LABEL_COLOR: dict[str, str] = {
    "fire_or_smoke_alarm": "#ff4444",
    "scream": "#ff8800",
    "siren": "#ffcc00",
    "glass_breaking": "#ff6688",
    "normal_background": "#44bb66",
}


@dataclass
class LogEntry:
    time_str: str
    label: str
    confidence: float
    is_alert: bool


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.waveform: deque[float] = deque(maxlen=WAVEFORM_MAXLEN)
        self.label: str = "—"
        self.confidence: float = 0.0
        self.proba: dict[str, float] = {}
        self.is_alert: bool = False
        self.display_alert_until: float = 0.0
        self.display_alert_label: str = "—"
        self.display_alert_confidence: float = 0.0
        self.last_alert_time: float = 0.0
        self.log: deque[LogEntry] = deque(maxlen=LOG_MAXLEN)
        self.started: bool = False
        self.pending_alert_label: str | None = None
        self.pending_alert_count: int = 0
        self.active_alarm_label: str | None = None
        self.last_window_rms: float = 0.0
        self.audio_chunks: int = 0
        self.inference_count: int = 0
        self.last_prediction_time: str = "—"
        self.status_message: str = "Waiting to start."
        self.error_message: str = ""


# Module-level singletons — persist across Streamlit reruns in the same process
_app_state: AppState | None = None
_audio_thread: threading.Thread | None = None
_infer_thread: threading.Thread | None = None
_telegram_thread: threading.Thread | None = None
_audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
_telegram_notifier: TelegramNotifier | None = None


def get_state() -> AppState:
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state


def get_telegram_notifier() -> TelegramNotifier:
    global _telegram_notifier
    if _telegram_notifier is None:
        _telegram_notifier = TelegramNotifier()
    return _telegram_notifier


@st.cache_resource(show_spinner="Loading YAMNet and classifier…")
def load_bundle() -> tuple:
    bundle = joblib.load(MODELS_DIR / "classifier.pkl")
    yamnet = load_yamnet_model()
    return bundle, yamnet


def _make_audio_callback(state: AppState):
    samples_since_hop = [0]

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            with state.lock:
                state.status_message = f"Audio stream status: {status}"
        chunk = indata[:, 0].astype(np.float32)
        with state.lock:
            state.waveform.extend(chunk)
            state.audio_chunks += 1
        samples_since_hop[0] += len(chunk)
        if samples_since_hop[0] >= HOP_SAMPLES and len(state.waveform) >= WINDOW_SAMPLES:
            samples_since_hop[0] = 0
            snap = np.array(list(state.waveform)[-WINDOW_SAMPLES:], dtype=np.float32)
            try:
                _audio_queue.put_nowait(snap)
            except queue.Full:
                pass

    return callback


def _inference_worker(
    state: AppState,
    threshold: float,
    cooldown: float,
    consecutive: int,
    min_signal_rms: float,
    bundle: dict,
    yamnet,
) -> None:
    clf = bundle["classifier"]
    scaler = bundle["scaler"]
    class_names: list[str] = bundle["class_names"]

    while state.started:
        try:
            snap = _audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            label, confidence, proba_dict = classify_window(
                snap,
                yamnet,
                clf,
                scaler,
                class_names,
                min_signal_rms=min_signal_rms,
            )
        except Exception:
            with state.lock:
                state.error_message = traceback.format_exc()
                state.status_message = "Inference failed. See error details below."
            state.started = False
            break

        now = time.time()
        label_threshold = LABEL_ALERT_THRESHOLDS.get(label, threshold)
        is_alert_candidate = label in ALERT_LABELS and confidence >= label_threshold
        cooldown_ok = (now - state.last_alert_time) >= cooldown
        prediction_time = datetime.now().strftime("%H:%M:%S")
        rms = waveform_rms(snap)
        alarm_to_send: tuple[str, float] | None = None

        with state.lock:
            if is_alert_candidate and label == state.pending_alert_label:
                state.pending_alert_count += 1
            elif is_alert_candidate:
                state.pending_alert_label = label
                state.pending_alert_count = 1
            else:
                state.pending_alert_label = None
                state.pending_alert_count = 0
                state.active_alarm_label = None

            is_alert = is_alert_candidate and state.pending_alert_count >= consecutive
            is_new_alarm = is_alert and state.active_alarm_label != label
            should_show_alert = is_new_alarm and cooldown_ok
            state.label = label
            state.confidence = confidence
            state.proba = proba_dict
            state.is_alert = should_show_alert or now < state.display_alert_until
            state.inference_count += 1
            state.last_window_rms = rms
            state.last_prediction_time = prediction_time
            state.status_message = "Detection running."
            if should_show_alert:
                state.active_alarm_label = label
                state.last_alert_time = now
                state.display_alert_until = now + ALERT_DISPLAY_HOLD_SECONDS
                state.display_alert_label = label
                state.display_alert_confidence = confidence
                state.log.appendleft(
                    LogEntry(
                        time_str=prediction_time,
                        label=label,
                        confidence=confidence,
                        is_alert=True,
                    )
                )
                alarm_to_send = (label, confidence)

        if alarm_to_send:
            get_telegram_notifier().send_alarm(*alarm_to_send)


def start_telegram_registration_worker(state: AppState) -> None:
    global _telegram_thread
    if _telegram_thread and _telegram_thread.is_alive():
        return

    notifier = get_telegram_notifier()

    def telegram_run() -> None:
        while state.started:
            notifier.poll_register_commands()
            time.sleep(5.0)

    _telegram_thread = threading.Thread(
        target=telegram_run,
        daemon=True,
        name="telegram-registration",
    )
    _telegram_thread.start()


def start_detection(threshold: float, cooldown: float, consecutive: int, min_signal_rms: float) -> None:
    global _audio_thread, _infer_thread
    state = get_state()
    if state.started:
        return
    state.started = True
    with state.lock:
        state.status_message = "Loading model and starting microphone..."
        state.error_message = ""
    bundle, yamnet = load_bundle()

    callback = _make_audio_callback(state)

    def audio_run() -> None:
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP_SAMPLES,
                callback=callback,
            ):
                with state.lock:
                    state.status_message = "Microphone stream running."
                while state.started:
                    time.sleep(0.05)
        except Exception:
            with state.lock:
                state.error_message = traceback.format_exc()
                state.status_message = "Audio stream failed. Check microphone permissions/device."
            state.started = False

    _audio_thread = threading.Thread(target=audio_run, daemon=True, name="audio-capture")
    _infer_thread = threading.Thread(
        target=_inference_worker,
        args=(state, threshold, cooldown, consecutive, min_signal_rms, bundle, yamnet),
        daemon=True,
        name="inference",
    )
    _audio_thread.start()
    _infer_thread.start()
    start_telegram_registration_worker(state)


def _snapshot(state: AppState) -> dict:
    with state.lock:
        now = time.time()
        display_alert_active = now < state.display_alert_until
        return {
            "waveform": list(state.waveform),
            "label": state.label,
            "confidence": state.confidence,
            "proba": dict(state.proba),
            "is_alert": display_alert_active,
            "alert_label": state.display_alert_label if display_alert_active else state.label,
            "alert_confidence": state.display_alert_confidence if display_alert_active else state.confidence,
            "log": list(state.log),
            "started": state.started,
            "audio_chunks": state.audio_chunks,
            "inference_count": state.inference_count,
            "last_prediction_time": state.last_prediction_time,
            "status_message": state.status_message,
            "error_message": state.error_message,
            "pending_alert_label": state.pending_alert_label,
            "pending_alert_count": state.pending_alert_count,
            "last_window_rms": state.last_window_rms,
            "telegram_status": get_telegram_notifier().last_status,
            "telegram_subscribers": get_telegram_notifier().subscriber_count,
        }


# ── Render helpers ─────────────────────────────────────────────────────────────

def _render_status(snap: dict) -> None:
    is_alert = snap["is_alert"]
    label = snap["alert_label"] if is_alert else snap["label"]
    confidence = snap["alert_confidence"] if is_alert else snap["confidence"]
    display_label = LABEL_DISPLAY.get(label, label)
    color = "#ff3333" if is_alert else "#22cc66"

    if is_alert:
        html = f"""
        <div style="
            background:{color}22;border:3px solid {color};border-radius:12px;
            padding:20px 32px;text-align:center;
            animation:pulse 1s infinite;
        ">
            <span style="font-size:2.2rem;font-weight:900;color:{color};">
                🚨 ALERT: {display_label.upper()}
            </span><br>
            <span style="font-size:1.1rem;color:{color}80;">
                Confidence {confidence:.0%}
            </span>
        </div>
        <style>
        @keyframes pulse {{
            0%   {{ box-shadow: 0 0 0 0   {color}88; }}
            70%  {{ box-shadow: 0 0 0 14px {color}00; }}
            100% {{ box-shadow: 0 0 0 0   {color}00; }}
        }}
        </style>
        """
    else:
        conf_str = f"  ·  {confidence:.0%}" if label != "—" else ""
        html = f"""
        <div style="
            background:{color}18;border:2px solid {color};border-radius:12px;
            padding:14px 32px;text-align:center;
        ">
            <span style="font-size:1.5rem;font-weight:700;color:{color};">● LISTENING</span>
            <span style="font-size:1.5rem;color:#aaa;margin-left:16px;">
                {display_label}{conf_str}
            </span>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


def _render_waveform(snap: dict) -> None:
    waveform = snap["waveform"]
    if len(waveform) < 10:
        st.caption("Waiting for audio…")
        return

    arr = np.array(waveform, dtype=np.float32)
    step = max(1, len(arr) // DISPLAY_POINTS)
    downsampled = arr[::step]
    t = np.linspace(0, len(arr) / SAMPLE_RATE, len(downsampled))

    df = pd.DataFrame({"t": t, "amp": downsampled})
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.2, color="#4da6ff")
        .encode(
            x=alt.X("t:Q", title="seconds", axis=alt.Axis(labelColor="#888", titleColor="#888")),
            y=alt.Y(
                "amp:Q",
                title="amplitude",
                scale=alt.Scale(domain=[-1.0, 1.0]),
                axis=alt.Axis(labelColor="#888", titleColor="#888", tickCount=3),
            ),
        )
        .properties(height=130)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_detection(snap: dict) -> None:
    label = snap["label"]
    conf = snap["confidence"]
    display = LABEL_DISPLAY.get(label, label)
    color = LABEL_COLOR.get(label, "#cccccc")
    st.markdown(
        f"<div style='text-align:center;padding:18px 0;'>"
        f"<div style='font-size:1.9rem;font-weight:900;color:{color};'>{display}</div>"
        f"<div style='font-size:1.3rem;color:#888;margin-top:6px;'>{conf:.0%} confidence</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_confidence_bars(snap: dict) -> None:
    proba = snap["proba"]
    if not proba:
        st.caption("No prediction yet.")
        return

    color_domain = [LABEL_DISPLAY.get(k, k) for k in LABEL_COLOR]
    color_range = list(LABEL_COLOR.values())

    rows = [
        {"Class": LABEL_DISPLAY.get(k, k), "Confidence": v}
        for k, v in proba.items()
    ]
    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "Confidence:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="",
                axis=alt.Axis(format="%", labelColor="#888"),
            ),
            y=alt.Y(
                "Class:N",
                sort="-x",
                title="",
                axis=alt.Axis(labelColor="#ccc", labelFontSize=12),
            ),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=None,
            ),
        )
        .properties(height=165)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_event_log(snap: dict) -> None:
    log: list[LogEntry] = snap["log"]
    if not log:
        st.caption("No events yet.")
        return

    rows = []
    for entry in log:
        rows.append({
            "Time": entry.time_str,
            "Class": LABEL_DISPLAY.get(entry.label, entry.label),
            "Confidence": f"{entry.confidence:.0%}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(320, 44 + 35 * len(rows)))


def _render_diagnostics(snap: dict) -> None:
    st.caption("Runtime")
    st.write(f"Status: {snap['status_message']}")
    st.write(f"Audio chunks: {snap['audio_chunks']}")
    st.write(f"Predictions: {snap['inference_count']}")
    st.write(f"Last prediction: {snap['last_prediction_time']}")
    st.write(f"Window RMS: {snap['last_window_rms']:.4f}")
    st.write(f"Alert streak: {snap['pending_alert_label'] or '—'} {snap['pending_alert_count']}")
    st.write(f"Telegram subscribers: {snap['telegram_subscribers']}")
    st.write(f"Telegram: {snap['telegram_status']}")
    if snap["error_message"]:
        st.error("Background worker failed.")
        st.code(snap["error_message"], language="text")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Emergency Sound Detection",
        page_icon="🚨",
        layout="wide",
    )

    with st.sidebar:
        st.title("⚙️ Settings")
        threshold = st.slider(
            "Alert threshold", 0.0, 1.0, DEFAULT_ALERT_THRESHOLD, 0.05,
            help="Minimum confidence to trigger an alert.",
        )
        cooldown = st.slider(
            "Cooldown (s)", 0.0, 10.0, DEFAULT_ALERT_COOLDOWN_SECONDS, 0.5,
            help="Minimum seconds between repeated alerts.",
        )
        consecutive = st.slider(
            "Consecutive windows", 1, 6, DEFAULT_ALERT_CONSECUTIVE_WINDOWS, 1,
            help="How many matching emergency windows are required before showing an alert.",
        )
        min_signal_rms = st.slider(
            "Min signal RMS", 0.0, 0.10, DEFAULT_MIN_SIGNAL_RMS, 0.005,
            help="Optional quiet-window filter. Keep at 0.0 to disable.",
        )
        st.divider()
        st.caption("Target classes")
        for key, display in LABEL_DISPLAY.items():
            color = LABEL_COLOR[key]
            st.markdown(
                f"<span style='color:{color};font-size:1.1rem;'>■</span> {display}",
                unsafe_allow_html=True,
            )
        st.divider()
        diagnostics_ph = st.empty()

    st.title("🚨 Emergency Sound Detection")
    st.caption("Real-time microphone classification via YAMNet + logistic regression.")

    start_detection(threshold, cooldown, consecutive, min_signal_rms)

    # Persistent placeholder containers
    status_ph = st.empty()
    st.markdown("##### Live Waveform")
    waveform_ph = st.empty()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### Current Detection")
        detection_ph = st.empty()
    with col_r:
        st.markdown("##### Class Confidence")
        bars_ph = st.empty()
    st.markdown("##### Recent Alarms")
    log_ph = st.empty()

    # Render loop — updates placeholders every REFRESH_INTERVAL_S
    while True:
        snap = _snapshot(get_state())

        with status_ph.container():
            _render_status(snap)
        with waveform_ph.container():
            _render_waveform(snap)
        with detection_ph.container():
            _render_detection(snap)
        with bars_ph.container():
            _render_confidence_bars(snap)
        with log_ph.container():
            _render_event_log(snap)
        with diagnostics_ph.container():
            _render_diagnostics(snap)

        time.sleep(REFRESH_INTERVAL_S)


if __name__ == "__main__":
    main()
