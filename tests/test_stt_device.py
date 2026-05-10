"""
STT device tests — verify faster-whisper loads on CUDA and the
cpu_fallback flag correctly reflects what actually happened.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.stt import STT


def _cuda_available() -> bool:
    try:
        from app.stt import _preload_ctranslate2_lib

        _preload_ctranslate2_lib()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def test_stt_loads_on_cuda():
    """STT must load on CUDA — no silent CPU fallback."""
    if not _cuda_available():
        pytest.skip("CTranslate2 CUDA not available on this system")

    stt = STT(model="tiny.en", device="cuda", compute_type="int8")
    ok = stt.load()
    assert ok, "STT failed to load entirely"
    assert not stt.cpu_fallback, (
        "STT fell back to CPU despite CUDA being available. Check CTranslate2 CUDA installation."
    )
    assert stt.device == "cuda", f"Expected device='cuda', got '{stt.device}'"
    stt.unload()


def test_stt_cpu_fallback_flag_set_on_forced_cpu():
    """cpu_fallback flag must be True when CUDA load is forced to fail."""
    from unittest.mock import MagicMock, patch

    call_count = [0]

    def fake_whisper_model(model_name, device="cpu", compute_type="auto"):
        call_count[0] += 1
        if device == "cuda":
            raise RuntimeError("simulated CUDA unavailable")
        return MagicMock()  # CPU load succeeds

    with patch("faster_whisper.WhisperModel", side_effect=fake_whisper_model):
        stt = STT(model="tiny.en", device="cuda", compute_type="int8")
        ok = stt.load()

    assert ok, "CPU fallback should have succeeded after simulated CUDA failure"
    assert stt.cpu_fallback, "cpu_fallback must be True after CUDA load failure"
    assert stt.device == "cpu", f"Expected device='cpu' after fallback, got '{stt.device}'"
    assert call_count[0] == 2, "Expected two WhisperModel calls: CUDA attempt + CPU fallback"


def test_stt_get_info_includes_device_and_fallback():
    """get_info() must expose device and cpu_fallback for callers."""
    stt = STT(model="tiny.en", device="cuda", compute_type="int8")
    stt.load()
    info = stt.get_info()
    assert "device" in info
    assert "cpu_fallback" in info
    assert isinstance(info["cpu_fallback"], bool)
    stt.unload()
