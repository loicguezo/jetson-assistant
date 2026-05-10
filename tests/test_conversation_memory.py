"""
Conversation memory integration test — requires llama-server to be running.

Verifies that the model uses previous turns as context, and that clearing
history makes it forget.

Run with:
    pytest tests/test_conversation_memory.py -v -s
or:
    python tests/test_conversation_memory.py
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.history import HISTORY_FILE, clear_history
from app.manager import get_llama_server_port

SYSTEM_PROMPT = "You are a helpful voice assistant. Answer in one sentence. Be direct."


def _base_url() -> str:
    return f"http://127.0.0.1:{get_llama_server_port()}"


def _server_running() -> bool:
    try:
        with httpx.Client(timeout=3.0) as c:
            return c.get(f"{_base_url()}/v1/models").status_code == 200
    except Exception:
        return False


def ask(messages: list[dict]) -> str:
    """Send a messages list and return the full response text."""
    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{_base_url()}/v1/chat/completions",
            json={
                "messages": messages,
                "stream": True,
                "max_tokens": 64,
                "temperature": 0.1,  # low temperature for deterministic answers
            },
        ) as r:
            r.raise_for_status()
            text = ""
            for line in r.iter_lines():
                if not line or not line.strip().startswith("data:"):
                    continue
                raw = line.strip()[5:]
                if raw == "[DONE]":
                    break
                try:
                    delta = (
                        (json.loads(raw).get("choices") or [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    text += delta or ""
                except json.JSONDecodeError:
                    continue
            return text.strip()


@pytest.fixture(autouse=True)
def cleanup_history():
    """Remove history file before and after each test."""
    clear_history()
    yield
    clear_history()


@pytest.fixture(scope="session")
def base_url():
    if not _server_running():
        pytest.skip("llama-server not running — start with: ./jetson-assistant start")
    return _base_url()


# ── Tests ─────────────────────────────────────────────────────────


def test_model_remembers_name_from_history(base_url):
    """Model should answer 'Jetson' when the name was given in a prior turn."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Your name is Jetson. Remember that."},
        {"role": "assistant", "content": "Understood, my name is Jetson."},
        {"role": "user", "content": "What is your name?"},
    ]
    response = ask(messages)
    print(f"\n  Response with history: {response!r}")
    assert "jetson" in response.lower(), f"Expected 'Jetson' in response, got: {response!r}"


def test_model_forgets_name_without_history(base_url):
    """Without history, the model should NOT know the name 'Jetson'."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is your name?"},
    ]
    response = ask(messages)
    print(f"\n  Response without history: {response!r}")
    assert "jetson" not in response.lower(), (
        f"Model answered 'Jetson' without history — history leak? Got: {response!r}"
    )


def test_clear_history_removes_file(base_url):
    """save_history + clear_history should create then remove the file."""
    from app.history import has_history, save_history

    save_history(
        [
            {"role": "user", "content": "Your name is Jetson."},
            {"role": "assistant", "content": "My name is Jetson."},
        ]
    )
    assert has_history(), "History file should exist after save"

    clear_history()
    assert not HISTORY_FILE.exists(), "History file should be gone after clear"


# ── Standalone runner ─────────────────────────────────────────────


def main():
    if not _server_running():
        print("ERROR: llama-server not running. Start with: ./jetson-assistant start")
        sys.exit(1)

    url = _base_url()
    print(f"llama-server at {url}\n")
    passed = 0
    failed = 0

    # Step 1: ask with history
    print("Step 1: Tell model its name is Jetson (via history), then ask...")
    messages_with = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Your name is Jetson. Remember that."},
        {"role": "assistant", "content": "Understood, my name is Jetson."},
        {"role": "user", "content": "What is your name?"},
    ]
    resp1 = ask(messages_with)
    ok1 = "jetson" in resp1.lower()
    print(f"  Response: {resp1!r}")
    print(f"  {'OK' if ok1 else 'FAIL'} — expected 'Jetson' in response")
    passed += ok1
    failed += not ok1

    # Step 2: ask without history (simulates clear)
    print("\nStep 2: Ask again WITHOUT history (simulates clear)...")
    messages_without = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is your name?"},
    ]
    resp2 = ask(messages_without)
    ok2 = "jetson" not in resp2.lower()
    print(f"  Response: {resp2!r}")
    print(f"  {'OK' if ok2 else 'FAIL'} — expected 'Jetson' NOT in response")
    passed += ok2
    failed += not ok2

    print(f"\n{'─' * 50}")
    print(f"{'All passed' if not failed else f'{failed} FAILED'} ({passed}/{passed + failed})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
