"""
TTS latency tests — verify synthesis is fast enough for gap-free streaming.

Gap-free condition: synthesis_time(chunk_N+1) < play_time(chunk_N)
With Kokoro CUDA RTF ~0.14x and speech ~2.5 w/s:
  synthesis_time ≈ (words/2.5) × 0.14 ≈ words × 0.056 s
  play_time      ≈  words/2.5          ≈ words × 0.40 s
So synthesis is ~7x faster than playback — plenty of overlap margin.
"""

import json
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


VOICES_DIR = Path(__file__).parent.parent / "voices"
WORKER = Path(__file__).parent.parent / "app" / "tts_worker.py"

CHUNKS = [
    "Hello, how can I help you today.",
    "I am a voice assistant running on an NVIDIA Jetson.",
    "The weather today is sunny and warm.",
    "Would you like me to tell you more?",
]


@pytest.fixture(scope="module")
def tts_proc():
    if not (VOICES_DIR / "kokoro-v1.0.onnx").exists():
        pytest.skip("Kokoro model not downloaded")
    proc = subprocess.Popen(
        [sys.executable, str(WORKER), "--model-dir", str(VOICES_DIR)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ready = select.select([proc.stdout], [], [], 30.0)[0]
    assert ready, "TTS worker timed out"
    resp = json.loads(proc.stdout.readline())
    assert resp.get("status") == "ready", f"Worker not ready: {resp}"
    yield proc, resp.get("provider", "unknown")
    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=5)


def _synthesize(proc, text: str) -> float:
    proc.stdin.write(json.dumps({"cmd": "synthesize", "text": text}) + "\n")
    proc.stdin.flush()
    t0 = time.perf_counter()
    resp = json.loads(proc.stdout.readline())
    return time.perf_counter() - t0, resp


def test_synthesis_faster_than_realtime(tts_proc):
    """Each chunk must synthesize in less than its own play duration (RTF < 1.0)."""
    proc, provider = tts_proc
    for text in CHUNKS:
        dt, resp = _synthesize(proc, text)
        assert "audio_b64" in resp, f"Synthesis failed: {resp}"
        import base64

        import numpy as np

        audio = np.frombuffer(base64.b64decode(resp["audio_b64"]), dtype=np.int16)
        sr = resp["sample_rate"]
        play_time = len(audio) / sr
        rtf = dt / play_time
        assert rtf < 1.0, (
            f"RTF {rtf:.2f}x ≥ 1.0 for '{text[:30]}...' — TTS cannot keep up with playback"
        )


def test_synthesis_overlap_margin(tts_proc):
    """Synthesis must be fast enough to finish while the previous chunk plays.

    Condition: synth_time(chunk_N) < play_time(chunk_N-1)
    i.e. RTF of chunk_N < 1.0 (guaranteed by test above, but we also verify
    the absolute values are within the gap-free window).
    """
    proc, provider = tts_proc
    prev_play_time = None
    for text in CHUNKS:
        dt, resp = _synthesize(proc, text)
        if "audio_b64" not in resp:
            continue
        import base64

        import numpy as np

        audio = np.frombuffer(base64.b64decode(resp["audio_b64"]), dtype=np.int16)
        play_time = len(audio) / resp["sample_rate"]
        if prev_play_time is not None:
            assert dt < prev_play_time, (
                f"Synthesis ({dt:.2f}s) took longer than previous chunk play time "
                f"({prev_play_time:.2f}s) — audio gap would occur. Provider: {provider}"
            )
        prev_play_time = play_time


def test_first_chunk_latency(tts_proc):
    """First 3-word chunk must synthesize in under 0.5 s (time-to-first-speech)."""
    proc, _ = tts_proc
    dt, resp = _synthesize(proc, "Hello there friend.")
    assert "audio_b64" in resp
    assert dt < 0.5, f"First chunk latency {dt:.2f}s — user will hear a noticeable pause"
