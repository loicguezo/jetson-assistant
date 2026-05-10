"""Conversation history — load/save rolling turn window to disk."""

import json
from pathlib import Path

HISTORY_FILE = Path.home() / ".jetson-assistant" / "history.json"


def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def save_history(history: list[dict]) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    except Exception:
        pass


def clear_history() -> None:
    try:
        HISTORY_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def has_history() -> bool:
    return HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 2
