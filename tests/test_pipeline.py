"""Unit tests for pipeline edge cases (issue #12).

All tests run without hardware or a running llama-server.
"""

import queue
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from app.pipeline import (
    _pa_match,
    chunk_rms,
    stream_and_speak,
    tts_player,
)

# ── chunk_rms ─────────────────────────────────────────────────────


def test_chunk_rms_silence():
    raw = np.zeros(512, dtype=np.int16).tobytes()
    assert chunk_rms(raw) == pytest.approx(0.0)


def test_chunk_rms_full_scale():
    raw = np.full(512, 32767, dtype=np.int16).tobytes()
    assert chunk_rms(raw) > 0.99


def test_chunk_rms_mid_level():
    amplitude = 16384  # ~half scale
    raw = np.full(512, amplitude, dtype=np.int16).tobytes()
    rms = chunk_rms(raw)
    assert 0.4 < rms < 0.6


# ── _pa_match ────────────────────────────────────────────────────


def test_pa_match_exact():
    assert _pa_match("usb_audio", "alsa_input.usb_audio.0")


def test_pa_match_space_as_underscore():
    assert _pa_match("USB Audio", "alsa_input.usb_audio.0")


def test_pa_match_no_match():
    assert not _pa_match("Bluetooth", "alsa_input.usb_audio.0")


def test_pa_match_case_insensitive():
    assert _pa_match("USB AUDIO", "alsa_input.usb_audio.0")


# ── tts_player — TTS returns audio=None ─────────────────────────


def test_tts_player_skips_none_audio():
    """When synthesize returns audio=None, tts_player must not crash or call Popen."""
    tts_obj = Mock()
    tts_obj.synthesize.return_value = {"audio": None}

    tts_q: queue.Queue = queue.Queue()
    tts_q.put("hello world")
    tts_q.put(None)

    with patch("subprocess.Popen") as mock_popen:
        tts_player(tts_obj, tts_q)

    mock_popen.assert_not_called()


def test_tts_player_skips_none_audio_multiple_chunks():
    """Multiple None-audio chunks all skipped; no Popen, no exception."""
    tts_obj = Mock()
    tts_obj.synthesize.return_value = {"audio": None}

    tts_q: queue.Queue = queue.Queue()
    for text in ["chunk one", "chunk two", "chunk three"]:
        tts_q.put(text)
    tts_q.put(None)

    with patch("subprocess.Popen") as mock_popen:
        tts_player(tts_obj, tts_q)

    mock_popen.assert_not_called()
    assert tts_obj.synthesize.call_count == 3


# ── tts_player — paplay → aplay fallback ────────────────────────


def _make_mock_proc():
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.close = MagicMock()
    proc.wait = MagicMock(return_value=0)
    return proc


def test_tts_player_falls_back_to_aplay_when_paplay_missing():
    """paplay raises FileNotFoundError → tts_player retries with aplay."""
    audio = np.zeros(1000, dtype=np.float32)
    tts_obj = Mock()
    tts_obj.synthesize.return_value = {"audio": audio, "sample_rate": 24000}

    tts_q: queue.Queue = queue.Queue()
    tts_q.put("hello")
    tts_q.put(None)

    mock_proc = _make_mock_proc()
    cmds_used = []

    def popen_side_effect(cmd, **kwargs):
        cmds_used.append(cmd[0])
        if cmd[0] == "paplay":
            raise FileNotFoundError
        return mock_proc

    with patch("subprocess.Popen", side_effect=popen_side_effect):
        tts_player(tts_obj, tts_q)

    assert "paplay" in cmds_used
    assert "aplay" in cmds_used
    mock_proc.stdin.write.assert_called_once()


def test_tts_player_uses_paplay_when_available():
    """When paplay is available, aplay must not be called."""
    audio = np.zeros(1000, dtype=np.float32)
    tts_obj = Mock()
    tts_obj.synthesize.return_value = {"audio": audio, "sample_rate": 24000}

    tts_q: queue.Queue = queue.Queue()
    tts_q.put("hello")
    tts_q.put(None)

    mock_proc = _make_mock_proc()
    cmds_used = []

    def popen_side_effect(cmd, **kwargs):
        cmds_used.append(cmd[0])
        return mock_proc

    with patch("subprocess.Popen", side_effect=popen_side_effect):
        tts_player(tts_obj, tts_q)

    assert cmds_used == ["paplay"]
    mock_proc.stdin.write.assert_called_once()


def test_tts_player_reuses_single_paplay_process_for_multiple_chunks():
    """All chunks in one response go through a single paplay process (gapless audio)."""
    audio = np.zeros(1000, dtype=np.float32)
    tts_obj = Mock()
    tts_obj.synthesize.return_value = {"audio": audio, "sample_rate": 24000}

    tts_q: queue.Queue = queue.Queue()
    for text in ["first chunk", "second chunk", "third chunk"]:
        tts_q.put(text)
    tts_q.put(None)

    mock_proc = _make_mock_proc()

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        tts_player(tts_obj, tts_q)

    # Only one Popen call for the whole response
    assert mock_popen.call_count == 1
    # But stdin.write called once per chunk
    assert mock_proc.stdin.write.call_count == 3


# ── stream_and_speak — LLM no response retries once ─────────────


def test_stream_and_speak_empty_llm_retries_once():
    """When LLM returns no tokens (ttft=None), stream_and_speak retries exactly once."""
    call_count = 0

    def fake_stream(prompt, system_prompt, few_shot=None):
        nonlocal call_count
        call_count += 1
        return iter([])  # empty — no tokens generated

    llm = Mock()
    llm.generate_stream = fake_stream

    with patch("time.sleep"):  # skip the 0.3s retry delay
        resp, dt, ttft = stream_and_speak(llm, None, "hi", "be helpful")

    assert ttft is None
    assert resp == ""
    assert call_count == 2  # original attempt + one retry


def test_stream_and_speak_no_retry_on_second_attempt():
    """With _retry=False, an empty response does not trigger a second retry."""
    call_count = 0

    def fake_stream(prompt, system_prompt, few_shot=None):
        nonlocal call_count
        call_count += 1
        return iter([])

    llm = Mock()
    llm.generate_stream = fake_stream

    resp, dt, ttft = stream_and_speak(llm, None, "hi", "sys", _retry=False)

    assert ttft is None
    assert call_count == 1


def test_stream_and_speak_returns_content_on_success():
    """Normal LLM response: content collected, ttft set, no retry."""
    call_count = 0

    def fake_stream(prompt, system_prompt, few_shot=None):
        nonlocal call_count
        call_count += 1
        for word in ["Hello", " world", "."]:
            yield (word, {})

    llm = Mock()
    llm.generate_stream = fake_stream

    resp, dt, ttft = stream_and_speak(llm, None, "hi", "sys")

    assert resp == "Hello world."
    assert ttft is not None
    assert ttft >= 0
    assert call_count == 1


# ── stream_and_speak — TTS chunking ─────────────────────────────


def test_stream_and_speak_first_chunk_word_limit():
    """First TTS chunk sent at first_chunk_words=3, remainder at max_chunk_words=5."""
    received_chunks: list[str] = []

    def fake_stream(prompt, system_prompt, few_shot=None):
        # 10 words, no punctuation — pure word-count gating
        for w in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]:
            yield (w + " ", {})

    llm = Mock()
    llm.generate_stream = fake_stream

    tts_obj = Mock()

    def fake_synthesize(text):
        received_chunks.append(text)
        return {"audio": None}  # skip playback

    tts_obj.synthesize = fake_synthesize

    resp, _, _ = stream_and_speak(llm, tts_obj, "hi", "sys", first_chunk_words=3, max_chunk_words=5)

    assert len(received_chunks) >= 2
    assert len(received_chunks[0].split()) == 3  # first chunk = 3 words
    assert len(received_chunks[1].split()) == 5  # second chunk = 5 words


def test_stream_and_speak_punctuation_flushes_tts_early():
    """A sentence-ending punctuation mark sends TTS even before the word limit."""
    received_chunks: list[str] = []

    def fake_stream(prompt, system_prompt, few_shot=None):
        # "Hello world." — 2 words with a break char; should flush before limit=10
        yield ("Hello", {})
        yield (" world.", {})

    llm = Mock()
    llm.generate_stream = fake_stream

    tts_obj = Mock()

    def fake_synthesize(text):
        received_chunks.append(text)
        return {"audio": None}

    tts_obj.synthesize = fake_synthesize

    stream_and_speak(llm, tts_obj, "hi", "sys", first_chunk_words=10, max_chunk_words=10)

    assert len(received_chunks) == 1
    assert "." in received_chunks[0]
