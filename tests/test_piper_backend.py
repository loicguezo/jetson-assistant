"""Unit tests for the Piper TTS backend and dual-backend dispatch (issue #28)."""

import os
from unittest.mock import patch

import numpy as np

from app.config import Config
from app.tts import (  # noqa: F401
    PiperTTS,
    _parse_piper_model,
    create_tts,
    download_piper_model_if_missing,
)

# ── _parse_piper_model ────────────────────────────────────────────


def test_parse_german_model():
    lang, locale, speaker, quality = _parse_piper_model("de_DE-thorsten-medium")
    assert lang == "de"
    assert locale == "de_DE"
    assert speaker == "thorsten"
    assert quality == "medium"


def test_parse_english_model():
    lang, locale, speaker, quality = _parse_piper_model("en_US-lessac-medium")
    assert lang == "en"
    assert locale == "en_US"
    assert speaker == "lessac"
    assert quality == "medium"


def test_parse_hyphenated_speaker():
    # Some piper models have hyphenated speaker names
    lang, locale, speaker, quality = _parse_piper_model("de_DE-kerstin-low")
    assert speaker == "kerstin"
    assert quality == "low"


# ── create_tts dispatch ───────────────────────────────────────────


def test_create_tts_default_is_kokoro():
    from app.tts import KokoroTTS

    tts = create_tts()
    assert isinstance(tts, KokoroTTS)
    assert tts.backend_name == "Kokoro"


def test_create_tts_piper_backend():
    tts = create_tts(backend="piper", piper_model="de_DE-thorsten-medium")
    assert isinstance(tts, PiperTTS)
    assert tts.backend_name == "Piper"
    assert tts.model == "de_DE-thorsten-medium"


def test_create_tts_piper_default_model():
    tts = create_tts(backend="piper")
    assert isinstance(tts, PiperTTS)
    assert tts.model == "en_US-lessac-medium"


def test_create_tts_kokoro_explicit():
    from app.tts import KokoroTTS

    tts = create_tts(backend="kokoro", voice="af_bella")
    assert isinstance(tts, KokoroTTS)
    assert tts.voice == "af_bella"


# ── PiperTTS.synthesize — binary not found ───────────────────────


def test_piper_synthesize_not_loaded_returns_error():
    tts = PiperTTS(model="en_US-lessac-medium")
    # _piper_voice is None — not loaded
    result = tts.synthesize("hello world")
    assert result["audio"] is None
    assert "error" in result


def test_piper_synthesize_handles_empty_text():
    tts = PiperTTS(model="en_US-lessac-medium")
    tts._piper_voice = object()  # simulate loaded
    result = tts.synthesize("  ")
    assert result["audio"] is None
    assert result.get("error") == "Empty"


def test_piper_synthesize_returns_audio_on_success():
    """When piper voice returns audio chunks, synthesize returns int16 numpy array."""
    from unittest.mock import MagicMock

    tts = PiperTTS(model="en_US-lessac-medium")
    tts._sample_rate = 22050

    # Mock AudioChunk-like objects
    chunk = MagicMock()
    chunk.audio_float_array = np.zeros(22050, dtype=np.float32)

    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = [chunk]
    tts._piper_voice = mock_voice

    with patch("piper.config.SynthesisConfig"):
        result = tts.synthesize("hello world")

    assert result["audio"] is not None
    assert result["sample_rate"] == 22050
    assert result["audio"].dtype == np.int16


def test_piper_synthesize_empty_chunks_returns_error():
    """When piper voice returns no chunks, synthesize returns error."""
    from unittest.mock import MagicMock

    tts = PiperTTS(model="en_US-lessac-medium")
    tts._sample_rate = 22050
    mock_voice = MagicMock()
    mock_voice.synthesize.return_value = []
    tts._piper_voice = mock_voice

    with patch("piper.config.SynthesisConfig"):
        result = tts.synthesize("hello")

    assert result["audio"] is None
    assert "error" in result


# ── Config — backend fields ───────────────────────────────────────


def test_config_default_backend_is_kokoro():
    cfg = Config.load(config_path="/nonexistent")
    assert cfg.tts.backend == "kokoro"


def test_config_default_piper_model():
    cfg = Config.load(config_path="/nonexistent")
    assert cfg.tts.piper_model == "en_US-lessac-medium"


def test_config_tts_backend_env_override():
    with patch.dict(os.environ, {"JA_TTS_BACKEND": "piper"}, clear=False):
        cfg = Config.load(config_path="/nonexistent")
    assert cfg.tts.backend == "piper"


def test_config_piper_model_env_override():
    with patch.dict(os.environ, {"JA_PIPER_MODEL": "de_DE-thorsten-medium"}, clear=False):
        cfg = Config.load(config_path="/nonexistent")
    assert cfg.tts.piper_model == "de_DE-thorsten-medium"
