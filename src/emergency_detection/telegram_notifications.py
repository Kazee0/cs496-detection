"""Telegram alarm notification helpers."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
SUBSCRIBERS_PATH = PROJECT_ROOT / "data" / "telegram_subscribers.json"


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE lines into os.environ without overriding existing values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def telegram_token() -> str | None:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    return token or None


def _api_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _post_json(token: str, method: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _api_url(token, method),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(token: str, method: str, params: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{_api_url(token, method)}?{query}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class TelegramSubscriber:
    chat_id: int
    name: str
    registered_at: float


class TelegramNotifier:
    def __init__(self, token: str | None = None, subscribers_path: Path = SUBSCRIBERS_PATH) -> None:
        self.token = token or telegram_token()
        self.subscribers_path = subscribers_path
        self.offset = 0
        self.last_status = "Telegram disabled. Set TELEGRAM_BOT_TOKEN in .env."
        self._subscribers = self._load_subscribers()

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def _load_subscribers(self) -> dict[int, TelegramSubscriber]:
        if not self.subscribers_path.exists():
            return {}
        try:
            raw = json.loads(self.subscribers_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        subscribers: dict[int, TelegramSubscriber] = {}
        for item in raw.get("subscribers", []):
            chat_id = int(item["chat_id"])
            subscribers[chat_id] = TelegramSubscriber(
                chat_id=chat_id,
                name=str(item.get("name", "")),
                registered_at=float(item.get("registered_at", 0.0)),
            )
        return subscribers

    def _save_subscribers(self) -> None:
        self.subscribers_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "subscribers": [
                {
                    "chat_id": subscriber.chat_id,
                    "name": subscriber.name,
                    "registered_at": subscriber.registered_at,
                }
                for subscriber in sorted(self._subscribers.values(), key=lambda item: item.chat_id)
            ]
        }
        self.subscribers_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _send_message(self, chat_id: int, text: str, *, disable_notification: bool = False) -> None:
        if not self.token:
            return
        _post_json(
            self.token,
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "disable_notification": disable_notification,
            },
        )

    def poll_register_commands(self) -> None:
        """Register chats that send /register to the bot."""
        if not self.token:
            return

        try:
            response = _get_json(
                self.token,
                "getUpdates",
                {
                    "offset": self.offset,
                    "timeout": 0,
                    "allowed_updates": json.dumps(["message"]),
                },
            )
        except Exception as exc:
            self.last_status = f"Telegram registration check failed: {exc}"
            return

        for update in response.get("result", []):
            self.offset = max(self.offset, int(update.get("update_id", 0)) + 1)
            message = update.get("message") or {}
            text = str(message.get("text") or "").strip()
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if text.split(maxsplit=1)[0].lower() != "/register" or chat_id is None:
                continue

            name = " ".join(
                part
                for part in [
                    str(chat.get("first_name") or "").strip(),
                    str(chat.get("last_name") or "").strip(),
                ]
                if part
            ) or str(chat.get("username") or chat_id)

            subscriber = TelegramSubscriber(
                chat_id=int(chat_id),
                name=name,
                registered_at=time.time(),
            )
            self._subscribers[subscriber.chat_id] = subscriber
            self._save_subscribers()
            self._send_message(
                subscriber.chat_id,
                "Registered. You will receive emergency sound alarm notifications from this dashboard.",
            )
            self.last_status = f"Registered Telegram chat: {subscriber.name}"

        if self.subscriber_count:
            self.last_status = f"Telegram ready. Subscribers: {self.subscriber_count}"

    def send_alarm(self, label: str, confidence: float) -> None:
        if not self.token:
            return
        if not self._subscribers:
            self.last_status = "Telegram ready, but no subscribers. Send /register to the bot."
            return

        text = f"Emergency sound alarm: {label.replace('_', ' ')} ({confidence:.0%} confidence)"
        sent = 0
        for subscriber in list(self._subscribers.values()):
            try:
                self._send_message(subscriber.chat_id, text)
                sent += 1
            except Exception as exc:
                self.last_status = f"Telegram send failed for {subscriber.chat_id}: {exc}"
        if sent:
            self.last_status = f"Telegram alarm sent to {sent} subscriber(s)."
