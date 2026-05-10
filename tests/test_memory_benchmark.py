"""
Conversation memory benchmark — TTFT vs. history depth.

Answers: how many turns of history can we keep before TTFT degrades
or RAM headroom becomes unsafe?

Requires llama-server to be running:
    ./jetson-assistant start

Run:
    python tests/test_memory_benchmark.py        # standalone, prints full table
    pytest tests/test_memory_benchmark.py -v -s  # pytest mode, asserts safe default
"""

import json
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.manager import get_llama_server_port
from app.monitor import get_system_stats, ram_used_gb

# ── Configuration ─────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful voice assistant running on an NVIDIA Jetson. "
    "Answer in one sentence. Be direct."
)

# Realistic synthetic turn: ~15 words user, ~20 words assistant (~45 tokens/turn)
_USER_TURN = "Tell me something interesting about space exploration."
_ASST_TURN = "Space exploration has revealed that Mars once had liquid water on its surface billions of years ago."

TURN_COUNTS = [0, 3, 5, 10, 15, 20]
BENCHMARK_QUESTION = "What is the distance from Earth to the Moon?"

TTFT_BASELINE_THRESHOLD = 2.0  # seconds — 0-turn baseline must be under this
TTFT_DEGRADATION_FACTOR = 2.5  # flag if TTFT grows > 2.5× the 0-turn baseline
RAM_SAFETY_MARGIN_GB = 0.3  # minimum free RAM we want to keep


# ── Helpers ────────────────────────────────────────────────────────


def _base_url() -> str:
    port = get_llama_server_port()
    return f"http://127.0.0.1:{port or 8080}"


def _server_running() -> bool:
    try:
        with httpx.Client(timeout=3.0) as c:
            return c.get(f"{_base_url()}/v1/models").status_code == 200
    except Exception:
        return False


def _build_history(n_turns: int) -> list[dict]:
    history = []
    for _ in range(n_turns):
        history.append({"role": "user", "content": _USER_TURN})
        history.append({"role": "assistant", "content": _ASST_TURN})
    return history


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~1.3 tokens per word."""
    total_words = sum(len(m["content"].split()) for m in messages)
    return int(total_words * 1.3)


def measure_ttft(question: str, history: list[dict], base_url: str) -> tuple[float, float]:
    """Return (ttft_seconds, total_seconds)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    t0 = time.perf_counter()
    ttft = None
    with httpx.Client(timeout=90.0) as client:
        with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "stream": True,
                "max_tokens": 64,
                "temperature": 0.0,
            },
        ) as r:
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
                    if delta and ttft is None:
                        ttft = time.perf_counter() - t0
                except json.JSONDecodeError:
                    continue

    total = time.perf_counter() - t0
    return (ttft or total), total


def run_benchmark(base_url: str) -> list[dict]:
    """Run full benchmark, return list of result dicts."""
    stats = get_system_stats()
    ram_total = stats.ram_total_gb
    ram_used_at_start = stats.ram_used_gb
    ram_free_at_start = ram_total - ram_used_at_start

    print(f"\n{'─' * 65}")
    print(f"  RAM at benchmark start: {ram_used_at_start:.1f} / {ram_total:.1f} GB used")
    print(f"  Free headroom:          {ram_free_at_start:.1f} GB")
    if stats.gpu_temp_c:
        print(
            f"  GPU:                    {stats.gpu_temp_c:.0f}°C  {stats.gpu_freq_mhz or '?':.0f} MHz"
        )
    print(f"{'─' * 65}")

    # Warmup — prime the KV-cache with the system prompt
    print("  Warming up KV-cache...", end=" ", flush=True)
    try:
        measure_ttft("Say hello.", [], base_url)
        print("done")
    except Exception as e:
        print(f"failed ({e})")

    print(f"\n  {'Turns':>5}  {'~Tokens':>8}  {'TTFT':>7}  {'Total':>7}  {'RAM':>8}")
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 8}")

    results = []
    baseline_ttft = None

    for n_turns in TURN_COUNTS:
        history = _build_history(n_turns)
        messages_full = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": BENCHMARK_QUESTION}]
        )
        token_est = _estimate_tokens(messages_full)

        try:
            ttft, total = measure_ttft(BENCHMARK_QUESTION, history, base_url)
            ram_after = ram_used_gb()
        except Exception as e:
            print(f"  {n_turns:>5}  {'—':>8}  {'ERROR':>7}  {'—':>7}  {e}")
            results.append({"turns": n_turns, "ttft": None, "error": str(e)})
            continue

        if baseline_ttft is None:
            baseline_ttft = ttft

        degraded = ttft > baseline_ttft * TTFT_DEGRADATION_FACTOR
        flag = " ⚠" if degraded else ""

        print(
            f"  {n_turns:>5}  {token_est:>7}t  "
            f"{ttft:>6.2f}s  {total:>6.2f}s  "
            f"{ram_after:.1f}/{ram_total:.1f}GB{flag}"
        )

        results.append(
            {
                "turns": n_turns,
                "token_est": token_est,
                "ttft": ttft,
                "total": total,
                "ram_used_gb": ram_after,
                "ram_free_gb": ram_total - ram_after,
                "degraded": degraded,
            }
        )

    return results


def make_recommendation(results: list[dict], ram_total: float) -> dict:
    """Pick the highest safe memory_turns value."""
    valid = [r for r in results if r.get("ttft") is not None and not r.get("degraded")]
    safe_by_ram = [r for r in valid if r.get("ram_free_gb", 0) >= RAM_SAFETY_MARGIN_GB]
    if not safe_by_ram:
        safe_by_ram = valid[:1]  # at least 0 turns

    best = safe_by_ram[-1]
    return best


# ── pytest integration ────────────────────────────────────────────


@pytest.fixture(scope="module")
def benchmark_results():
    if not _server_running():
        pytest.skip("llama-server not running — start with: ./jetson-assistant start")
    return run_benchmark(_base_url())


def test_baseline_ttft_is_acceptable(benchmark_results):
    """0-turn TTFT must be under threshold — server health check."""
    baseline = next((r for r in benchmark_results if r["turns"] == 0), None)
    assert baseline is not None
    assert baseline["ttft"] is not None, "0-turn benchmark failed"
    assert baseline["ttft"] < TTFT_BASELINE_THRESHOLD, (
        f"Baseline TTFT {baseline['ttft']:.2f}s exceeds {TTFT_BASELINE_THRESHOLD}s "
        "— server may be overloaded or using wrong flags"
    )


def test_default_memory_turns_does_not_degrade(benchmark_results):
    """Current default (5 turns) must not cause TTFT degradation."""
    row = next((r for r in benchmark_results if r["turns"] == 5), None)
    if row is None:
        pytest.skip("5-turn result not available")
    assert not row.get("degraded"), (
        f"5-turn TTFT {row['ttft']:.2f}s is {row['ttft'] / benchmark_results[0]['ttft']:.1f}× "
        "baseline — default memory_turns=5 causes degradation"
    )


def test_print_recommendation(benchmark_results):
    """Print the recommendation (always passes, informational only)."""
    stats = get_system_stats()
    rec = make_recommendation(benchmark_results, stats.ram_total_gb)
    baseline = next((r for r in benchmark_results if r["turns"] == 0), {})
    baseline_ttft = baseline.get("ttft", 0)

    print(f"\n{'─' * 65}")
    print(f"  RECOMMENDATION for this device ({stats.ram_total_gb:.1f} GB RAM):")
    print(f"    memory_turns: {rec['turns']}")
    print(f"    TTFT at this setting: {rec['ttft']:.2f}s  (baseline: {baseline_ttft:.2f}s)")
    print(f"    Free RAM:  {rec['ram_free_gb']:.2f} GB")
    print(f"    Tokens in context: ~{rec.get('token_est', '?')}")
    print(f"  Update config/settings.yaml: memory_turns: {rec['turns']}")
    print(f"{'─' * 65}")
    assert True  # always passes


# ── standalone runner ─────────────────────────────────────────────


def main():
    if not _server_running():
        print("ERROR: llama-server not running. Start with: ./jetson-assistant start")
        sys.exit(1)

    base_url = _base_url()
    print(f"llama-server at {base_url}")

    results = run_benchmark(base_url)

    stats = get_system_stats()
    rec = make_recommendation(results, stats.ram_total_gb)
    baseline = next((r for r in results if r["turns"] == 0), {})
    baseline_ttft = baseline.get("ttft", 0)

    print(f"\n{'═' * 65}")
    print(f"  RECOMMENDATION for this device ({stats.ram_total_gb:.1f} GB RAM):")
    print(f"    memory_turns: {rec['turns']}")
    if baseline_ttft:
        ratio = rec["ttft"] / baseline_ttft
        print(
            f"    TTFT at this setting: {rec['ttft']:.2f}s  ({ratio:.1f}× baseline {baseline_ttft:.2f}s)"
        )
    print(f"    Free RAM:  {rec['ram_free_gb']:.2f} GB  (safety margin: {RAM_SAFETY_MARGIN_GB} GB)")
    print(f"    ~{rec.get('token_est', '?')} tokens sent per request")
    print(f"\n  → Edit config/settings.yaml: memory_turns: {rec['turns']}")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
