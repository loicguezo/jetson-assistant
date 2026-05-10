"""Unit tests for Config env-var overrides (issue #13 / #17)."""

import os
from unittest.mock import patch

import pytest

from app.config import Config


def _load_with_env(**env_vars):
    """Load a default Config with specific JA_* env vars set."""
    with patch.dict(os.environ, env_vars, clear=False):
        return Config.load(config_path="/nonexistent")


def test_max_tokens_override():
    cfg = _load_with_env(JA_MAX_TOKENS="256")
    assert cfg.llm.max_tokens == 256


def test_temperature_override():
    cfg = _load_with_env(JA_TEMPERATURE="0.3")
    assert cfg.llm.temperature == pytest.approx(0.3)


def test_tts_speed_override():
    cfg = _load_with_env(JA_TTS_SPEED="1.5")
    assert cfg.tts.speed == pytest.approx(1.5)


def test_first_chunk_words_override():
    cfg = _load_with_env(JA_FIRST_CHUNK_WORDS="5")
    assert cfg.tts.first_chunk_words == 5


def test_max_chunk_words_override():
    cfg = _load_with_env(JA_MAX_CHUNK_WORDS="10")
    assert cfg.tts.max_chunk_words == 10


def test_no_override_uses_defaults():
    cfg = _load_with_env()
    assert cfg.llm.max_tokens == 512
    assert cfg.llm.temperature == pytest.approx(0.7)
    assert cfg.tts.speed == pytest.approx(1.0)


def test_all_overrides_together():
    cfg = _load_with_env(
        JA_MAX_TOKENS="128",
        JA_TEMPERATURE="0.5",
        JA_TTS_SPEED="1.2",
        JA_FIRST_CHUNK_WORDS="4",
        JA_MAX_CHUNK_WORDS="8",
    )
    assert cfg.llm.max_tokens == 128
    assert cfg.llm.temperature == pytest.approx(0.5)
    assert cfg.tts.speed == pytest.approx(1.2)
    assert cfg.tts.first_chunk_words == 4
    assert cfg.tts.max_chunk_words == 8
