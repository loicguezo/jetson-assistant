"""
LLM latency integration test — requires llama-server to be running.

Run with:
    pytest tests/test_llm_latency.py -v -s
or:
    python tests/test_llm_latency.py

Each planet query must return a first token within TTFT_THRESHOLD seconds.
This catches thinking-model regressions and thermal throttling.
"""

import json
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.manager import get_llama_server_port

TTFT_THRESHOLD = 5.0  # seconds — fail if first token takes longer

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Answer in one sentence. Be direct."
)

PLANETS = [
    ("Mercury", "How far away is Mercury from Earth?"),
    ("Venus",   "How far away is Venus from Earth?"),
    ("Moon",    "How far away is the Moon from Earth?"),
    ("Mars",    "How far away is Mars from Earth?"),
    ("Jupiter", "How far away is Jupiter from Earth?"),
    ("Saturn",  "How far away is Saturn from Earth?"),
    ("Uranus",  "How far away is Uranus from Earth?"),
    ("Neptune", "How far away is Neptune from Earth?"),
]


def _base_url() -> str:
    port = get_llama_server_port()
    return f"http://127.0.0.1:{port or 8080}"


def _server_running() -> bool:
    try:
        with httpx.Client(timeout=3.0) as c:
            return c.get(f"{_base_url()}/v1/models").status_code == 200
    except Exception:
        return False


def measure_ttft(question: str, base_url: str) -> tuple[float, str]:
    """Return (ttft_seconds, first_token_text). Raises on connection error."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    start = time.perf_counter()
    first_token = ""
    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", f"{base_url}/v1/chat/completions", json={
            "messages": messages,
            "stream": True,
            "max_tokens": 64,
            "temperature": 0.7,
        }) as r:
            r.raise_for_status()
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
                    if delta:
                        first_token = delta
                        return time.perf_counter() - start, first_token
                except json.JSONDecodeError:
                    continue
    return time.perf_counter() - start, first_token


# ── pytest parametrized test ─────────────────────────────────────

@pytest.fixture(scope="session")
def base_url():
    if not _server_running():
        pytest.skip("llama-server not running — start with: ./jetson-assistant start")
    url = _base_url()
    # Warm up the KV-cache with one dummy request so that the system-prompt
    # prefix is cached before the timed tests run.
    try:
        measure_ttft("Say hello.", url)
    except Exception:
        pass
    return url


@pytest.mark.parametrize("name,question", PLANETS)
def test_ttft_planet(base_url, name, question):
    ttft, first = measure_ttft(question, base_url)
    print(f"\n  {name:10s}  TTFT {ttft:.2f}s  first='{first.strip()}'")
    assert ttft < TTFT_THRESHOLD, (
        f"{name}: TTFT {ttft:.1f}s exceeds {TTFT_THRESHOLD}s threshold — "
        "possible thinking-model regression or thermal throttling"
    )


# ── standalone runner ─────────────────────────────────────────────

def main():
    if not _server_running():
        print("ERROR: llama-server not running. Start with: ./jetson-assistant start")
        sys.exit(1)

    url = _base_url()
    print(f"llama-server at {url}")
    print("Warming up KV-cache...", end=" ", flush=True)
    try:
        measure_ttft("Say hello.", url)
        print("done\n")
    except Exception as e:
        print(f"failed ({e})\n")

    print(f"{'Planet':<12} {'TTFT':>7}  {'Pass?':<6}  First token")
    print("─" * 60)

    results = []
    for name, question in PLANETS:
        try:
            ttft, first = measure_ttft(question, url)
        except Exception as e:
            print(f"{name:<12} {'ERROR':>7}  FAIL    {e}")
            results.append((name, None))
            continue
        ok = ttft < TTFT_THRESHOLD
        mark = "OK" if ok else "FAIL"
        print(f"{name:<12} {ttft:>6.2f}s  {mark:<6}  {first.strip()!r}")
        results.append((name, ttft))

    failed = [n for n, t in results if t is None or t >= TTFT_THRESHOLD]
    print("─" * 60)
    if failed:
        times = [t for _, t in results if t is not None]
        print(f"FAILED ({len(failed)}/{len(PLANETS)}): {', '.join(failed)}")
        print(f"avg={sum(times)/len(times):.2f}s  max={max(times):.2f}s")
        sys.exit(1)
    else:
        times = [t for _, t in results if t is not None]
        print(f"All {len(PLANETS)} passed — avg {sum(times)/len(times):.2f}s  max {max(times):.2f}s")


if __name__ == "__main__":
    main()
