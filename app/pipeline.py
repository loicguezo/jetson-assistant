# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline — audio I/O, VAD, TTS streaming, and mic recording."""

import sys
import time
import wave
import subprocess
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Iterator

import numpy as np
from pathlib import Path
from rich.console import Console

from app.audio import kill_pulseaudio
from app.config import VADConfig

# Suppress noisy ALSA error messages (underrun warnings etc.)
# The callback reference must be kept alive to avoid segfault from GC.
_ALSA_ERR_T = None
_alsa_handler = None
try:
    import ctypes
    _ALSA_ERR_T = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    _alsa_handler = _ALSA_ERR_T(lambda *_: None)
    ctypes.cdll.LoadLibrary('libasound.so.2').snd_lib_error_set_handler(_alsa_handler)
except Exception:
    pass


# ── Audio constants (fixed by hardware, not user-tunable) ────────

SAMPLE_RATE = 16000
SILERO_CHUNK_SAMPLES = 512  # Silero VAD requires exactly 512 samples (32ms) at 16kHz
CHANNELS = 1

TTS_BREAKS = frozenset('.,;:!?\n')


# ── Audio helpers ─────────────────────────────────────────────────

def chunk_rms(raw: bytes) -> float:
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(pcm ** 2)))


def save_wav(chunks: list[bytes], path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(chunks))


def warmup_stt(stt_obj) -> float:
    """Run a dummy transcription to warm up CUDA. Returns elapsed seconds."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.zeros(SAMPLE_RATE // 2, dtype=np.int16).tobytes())
        t0 = time.perf_counter()
        stt_obj.transcribe(path, sample_rate=SAMPLE_RATE)
        return time.perf_counter() - t0
    finally:
        Path(path).unlink(missing_ok=True)


def _pa_match(needle: str, haystack: str) -> bool:
    """Match a name hint against a PulseAudio device name, ignoring space/underscore differences."""
    n = needle.lower().replace(" ", "_")
    h = haystack.lower().replace(" ", "_")
    return n in h


def find_pa_source(name_hint: str) -> Optional[str]:
    """Find a PulseAudio input source matching name_hint."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sources"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and _pa_match(name_hint, parts[1]) and "monitor" not in parts[1].lower():
                return parts[1]
    except Exception:
        pass
    return None


def find_pa_sink(name_hint: str) -> Optional[str]:
    """Find a PulseAudio output sink matching name_hint."""
    try:
        r = subprocess.run(["pactl", "list", "short", "sinks"],
                           capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and _pa_match(name_hint, parts[1]):
                return parts[1]
    except Exception:
        pass
    return None


def play_audio(audio: np.ndarray, sample_rate: int, sink: Optional[str] = None):
    """Play int16 audio via paplay (PulseAudio) or aplay fallback.

    sink=None  → paplay with PulseAudio default sink (respects BT speaker etc.)
    sink=<name> → paplay with a specific PA sink
    Falls back to aplay if paplay is not available.
    """
    raw = audio.astype(np.int16).tobytes()
    paplay_args = ["--format=s16le", f"--rate={sample_rate}", "--channels=1", "--raw"]
    if sink:
        cmd = ["paplay", f"--device={sink}"] + paplay_args
    else:
        cmd = ["paplay"] + paplay_args
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            p.stdin.write(raw)
        finally:
            p.stdin.close()
        p.wait(timeout=30)
        return
    except FileNotFoundError:
        pass
    except Exception:
        return
    # aplay fallback when paplay is not installed
    try:
        cmd = ["aplay", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw", "-q"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            p.stdin.write(raw)
        finally:
            p.stdin.close()
        p.wait(timeout=30)
    except Exception:
        pass


def tts_player(tts_obj, tts_q: queue.Queue, sink: Optional[str] = None):
    """Synthesize and play TTS chunks with synthesis overlapping playback.

    A synthesis thread pre-fetches the next chunk while the current one plays.
    All chunks are piped to a single paplay process to avoid per-chunk startup
    latency (~100 ms) and produce gapless audio.
    """
    audio_q: queue.Queue = queue.Queue()

    def _synthesize():
        while True:
            text = tts_q.get()
            if text is None:
                audio_q.put(None)
                return
            r = tts_obj.synthesize(text)
            audio_q.put(r)

    threading.Thread(target=_synthesize, daemon=True).start()

    proc: Optional[subprocess.Popen] = None
    use_aplay = False

    while True:
        r = audio_q.get()
        if r is None:
            break
        if r.get("audio") is None:
            continue

        audio_bytes = r["audio"].astype(np.int16).tobytes()
        sr = r["sample_rate"]

        if proc is None:
            pa_args = ["--format=s16le", f"--rate={sr}", "--channels=1", "--raw"]
            cmd = (["paplay", f"--device={sink}"] if sink else ["paplay"]) + pa_args
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                use_aplay = True
                cmd = ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1",
                       "-t", "raw", "-q"]
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    continue

        try:
            proc.stdin.write(audio_bytes)
        except (BrokenPipeError, OSError):
            proc = None

    if proc is not None:
        try:
            proc.stdin.close()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# ── Silero VAD ────────────────────────────────────────────────────

class SileroVAD:
    """Thin wrapper around the Silero VAD ONNX model."""

    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        self._model = load_silero_vad(onnx=True)
        self._torch = torch

    def __call__(self, raw_audio: bytes) -> float:
        """Return speech probability for raw int16 PCM audio at 16 kHz."""
        pcm = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(pcm)
        return self._model(tensor, SAMPLE_RATE).item()

    def reset(self):
        self._model.reset_states()


def load_silero(console: Optional[Console] = None) -> Optional[SileroVAD]:
    """Try to load Silero VAD. Returns wrapper or None on failure."""
    try:
        t0 = time.perf_counter()
        vad = SileroVAD()
        dt = time.perf_counter() - t0
        if console:
            console.print(f"  ✓ Silero VAD (ONNX, loaded in {dt:.1f}s)")
        return vad
    except ImportError:
        if console:
            console.print("  [yellow]⚠ silero-vad not installed (pip install silero-vad), using energy VAD[/yellow]")
        return None
    except Exception as e:
        if console:
            console.print(f"  [yellow]⚠ Silero VAD failed to load: {e}, using energy VAD[/yellow]")
        return None


# ── Speech segment ────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    """A completed speech utterance from the VAD."""
    audio: np.ndarray
    raw_chunks: list
    duration: float
    rms: float
    start_time: float
    end_time: float


# ── Mic recorder ──────────────────────────────────────────────────

class MicRecorder:
    """Manages mic recording via parecord/arecord with a background reader thread."""

    def __init__(self, console: Console, chunk_ms: int = 30):
        self.console = console
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        self.chunk_bytes = self.chunk_samples * CHANNELS * 2
        self.audio_q: queue.Queue[bytes] = queue.Queue()
        self.listening = threading.Event()
        self.listening.set()
        self.alive = True
        self._proc: Optional[subprocess.Popen] = None
        self.pa_source: Optional[str] = None
        self.pa_sink: Optional[str] = None

    def start(self, hw: str, mic_hint: str) -> bool:
        """Start recording. Returns True on success."""
        subprocess.run(["pkill", "-9", "parecord"], capture_output=True)
        subprocess.run(["pkill", "-9", "arecord"], capture_output=True)
        time.sleep(0.3)

        self.pa_source = find_pa_source(mic_hint)
        self.pa_sink = find_pa_sink(mic_hint)

        if self.pa_source:
            self.console.print(f"  PA source: {self.pa_source.split('.')[-2]}")
            rec_cmd = ["parecord", "-d", self.pa_source, "--format=s16le",
                       f"--rate={SAMPLE_RATE}", f"--channels={CHANNELS}", "--raw"]
        else:
            self.console.print("  [yellow]PA source not found, using ALSA direct[/yellow]")
            plughw = hw.replace("hw:", "plughw:")
            rec_cmd = ["arecord", "-D", plughw, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                       "-c", str(CHANNELS), "-t", "raw"]

        for attempt in range(3):
            self._proc = subprocess.Popen(
                rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                # Own process group so Ctrl+C (SIGINT) from the terminal
                # does not reach the recorder — we shut it down explicitly
                # via stop(), which sends SIGTERM then SIGKILL if needed.
                start_new_session=True,
            )
            time.sleep(0.5)
            if self._proc.poll() is None:
                break
            err = self._proc.stderr.read().decode(errors="replace").strip()
            self.console.print(f"  [red]Mic attempt {attempt+1} failed: {err}[/red]")
            # If ALSA says device is busy, release PulseAudio and retry once.
            # Only do this as a last resort — killing PA disconnects BT audio.
            if attempt == 0 and self.pa_source is None and (
                "busy" in err.lower() or "device or resource busy" in err.lower()
            ):
                self.console.print("  [dim]ALSA device busy — releasing PulseAudio...[/dim]")
                kill_pulseaudio()
                time.sleep(0.5)
            else:
                time.sleep(1)

        if self._proc is None or self._proc.poll() is not None:
            return False

        threading.Thread(target=self._reader, daemon=True).start()

        time.sleep(0.5)
        test_chunks = []
        for _ in range(10):
            try:
                test_chunks.append(self.audio_q.get(timeout=0.5))
            except queue.Empty:
                break
        if test_chunks:
            r = chunk_rms(b"".join(test_chunks))
            if r > 0.003:
                self.console.print("  Mic: [green]✓ live[/green]")
            else:
                self.console.print("  Mic: [red]✗ silent — unmute![/red]")
        else:
            self.console.print(
                f"  [red]Mic: no audio data (arecord running: {self._proc.poll() is None})[/red]\n"
                "  [dim]Check 'arecord -l' and set audio.input_device in config/settings.yaml[/dim]"
            )

        return True

    def _reader(self):
        while self.alive:
            raw = self._proc.stdout.read(self.chunk_bytes)
            if not raw:
                # Only report unexpected deaths — if self.alive is False we
                # called stop() intentionally and the process exit is expected.
                if self._proc.poll() is not None and self.alive:
                    err = self._proc.stderr.read().decode(errors="replace").strip()
                    if err:
                        self.console.print(f"\n  [red]arecord died: {err}[/red]")
                break
            if self.listening.is_set():
                self.audio_q.put(raw)

    def flush(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def pause(self):
        """Stop queuing audio and drain the buffer."""
        self.listening.clear()
        self.flush()

    def resume(self):
        """Drain any stale audio and resume queuing."""
        self.flush()
        self.listening.set()

    def stop(self):
        self.alive = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


# ── VAD loop ──────────────────────────────────────────────────────

def vad_loop(
    mic: MicRecorder,
    console: Console,
    vad_cfg: Optional[VADConfig] = None,
    silero: Optional[SileroVAD] = None,
) -> Iterator[SpeechSegment]:
    """Yields SpeechSegment each time a complete utterance is detected.

    When *silero* is provided, speech detection uses the neural model's
    probability (much better at rejecting non-speech sounds like coughs,
    keyboard clicks, and ambient noise).  RMS is still used as a cheap
    pre-filter to skip dead silence without invoking the model.

    The caller is responsible for calling mic.resume() after processing
    each segment (so audio stays paused during STT/LLM/TTS).
    """
    cfg = vad_cfg or VADConfig()
    chunk_ms = cfg.chunk_ms
    silence_chunks = int(cfg.silence_duration_ms / chunk_ms)
    lookback_chunks = int(cfg.lookback_ms / chunk_ms)
    max_chunks = int(cfg.max_speech_secs * 1000 / chunk_ms)

    use_silero = silero is not None
    silero_thresh = cfg.silero_threshold
    rms_silence_floor = 0.002  # below this, skip Silero inference entirely

    lookback: deque[bytes] = deque(maxlen=lookback_chunks)
    speech_raw: list[bytes] = []
    is_speaking = False
    silence_count = 0
    speech_start_t: float = 0.0

    while mic.alive:
        try:
            raw = mic.audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        rms = chunk_rms(raw)

        if use_silero:
            if rms < rms_silence_floor:
                is_speech = False
            else:
                is_speech = silero(raw) > silero_thresh
        else:
            is_speech = rms > cfg.speech_threshold

        if is_speech:
            silence_count = 0
            if not is_speaking:
                is_speaking = True
                speech_start_t = time.monotonic()
                speech_raw = list(lookback)
                sys.stdout.write("  🎤 Listening...\r")
                sys.stdout.flush()
            speech_raw.append(raw)
            if len(speech_raw) < max_chunks:
                continue
        else:
            if is_speaking:
                speech_raw.append(raw)
                silence_count += 1
                if silence_count < silence_chunks:
                    continue
            else:
                lookback.append(raw)
                continue

        is_speaking = False
        captured = speech_raw
        speech_raw = []
        silence_count = 0
        lookback.clear()
        speech_end_t = time.monotonic()

        if use_silero:
            silero.reset()

        dur_s = len(captured) * chunk_ms / 1000
        cap_rms = chunk_rms(b"".join(captured))

        sys.stdout.write("                              \r")
        sys.stdout.flush()
        mic.pause()

        if dur_s < cfg.min_utterance_secs or cap_rms < cfg.min_utterance_rms:
            console.print(f"[dim]  (noise: {dur_s:.1f}s, rms={cap_rms:.4f})[/dim]")
            mic.resume()
            continue

        raw_audio = b"".join(captured)
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        yield SpeechSegment(
            audio=audio_np,
            raw_chunks=captured,
            duration=dur_s,
            rms=cap_rms,
            start_time=speech_start_t,
            end_time=speech_end_t,
        )


# ── LLM streaming with TTS ───────────────────────────────────────

def stream_and_speak(
    llm,
    tts_obj,
    prompt: str,
    system_prompt: str,
    pa_sink: Optional[str] = None,
    few_shot: Optional[list[dict]] = None,
    first_chunk_words: int = 3,
    max_chunk_words: int = 8,
    _retry: bool = True,
) -> tuple[str, float, Optional[float]]:
    """Stream LLM response while chunking text to TTS for real-time playback.

    Returns (full_response, elapsed_seconds, time_to_first_token).
    Retries once when the server returns an empty response (stale KV-cache).
    """
    tts_q = None
    tts_thread = None
    if tts_obj:
        tts_q = queue.Queue()
        tts_thread = threading.Thread(
            target=tts_player, args=(tts_obj, tts_q, pa_sink), daemon=True,
        )
        tts_thread.start()

    full_resp = ""
    tts_buf = ""
    first_tts_sent = False
    t_llm = time.perf_counter()
    ttft = None

    for chunk_data in llm.generate_stream(
        prompt=prompt, system_prompt=system_prompt,
        few_shot=few_shot,
    ):
        content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
        if content:
            if ttft is None:
                ttft = time.perf_counter() - t_llm
            sys.stdout.write(content)
            sys.stdout.flush()
            full_resp += content

            if tts_q is not None:
                tts_buf += content
                words = len(tts_buf.split())
                limit = first_chunk_words if not first_tts_sent else max_chunk_words
                hit_break = any(c in content for c in TTS_BREAKS) and words >= 2
                if hit_break or words >= limit:
                    tts_q.put(tts_buf.strip())
                    tts_buf = ""
                    first_tts_sent = True

    dt_llm = time.perf_counter() - t_llm

    if tts_q is not None:
        if tts_buf.strip():
            tts_q.put(tts_buf.strip())
        tts_q.put(None)
        tts_thread.join(timeout=60)
        if tts_thread.is_alive():
            sys.stderr.write("  [warn] TTS thread did not finish in 60s\n")

    # llama-server occasionally returns [DONE] immediately when its KV-cache
    # is in a bad state after rapid successive requests. Retry once to recover.
    if ttft is None and _retry:
        time.sleep(0.3)
        return stream_and_speak(
            llm, tts_obj, prompt, system_prompt,
            pa_sink=pa_sink, few_shot=few_shot,
            first_chunk_words=first_chunk_words,
            max_chunk_words=max_chunk_words,
            _retry=False,
        )

    return full_resp, dt_llm, ttft


# ── Shared startup helpers ────────────────────────────────────────

def load_llm(config, console: Console):
    """Load and connect the LLM from config. Prints status. Returns LLM instance."""
    from app.llm import LLM
    from app.monitor import ram_used_gb
    ram_before = ram_used_gb()
    llm = LLM(
        model=config.llm.model, base_url=config.llm.base_url,
        backend=config.llm.backend, max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature, timeout=config.llm.timeout,
        system_prompt=config.llm.system_prompt,
    )
    if not llm.load():
        console.print("[red]✗ LLM failed to connect[/red]")
        return None
    delta = ram_used_gb() - ram_before
    total = ram_used_gb()
    console.print(f"  ✓ LLM ({llm.model})"
                  f"[dim]  +{delta:.1f}GB → {total:.1f}GB[/dim]")
    return llm


def load_tts(config, console: Console):
    """Load TTS from config. Prints status. Returns TTS instance or None."""
    from app.tts import create_tts
    from app.monitor import ram_used_gb
    ram_before = ram_used_gb()
    tts = create_tts(voice=config.tts.voice, speed=config.tts.speed, lang=config.tts.lang)
    tts = tts if tts.load() else None
    if tts:
        delta = ram_used_gb() - ram_before
        total = ram_used_gb()
        console.print(f"  ✓ TTS ({tts.backend_name}, {tts.voice})"
                      f"[dim]  +{delta:.1f}GB → {total:.1f}GB[/dim]")
    else:
        console.print("  ⚠ TTS unavailable — responses will be text only")
    return tts


def print_response_timing(console: Console, full_resp: str, dt_llm: float,
                          ttft: Optional[float], prefix: str = "  "):
    """Print TTFT / tok/s timing + live system stats after a response."""
    from app.monitor import get_system_stats, format_stats_inline
    stats = format_stats_inline(get_system_stats())
    if ttft is not None:
        toks = len(full_resp.split())
        console.print(
            f"{prefix}[dim]TTFT {ttft:.1f}s | LLM {dt_llm:.1f}s"
            f" ~{toks / (dt_llm or 1):.0f}w/s | {stats}[/dim]"
        )
    else:
        console.print(f"{prefix}[dim]LLM no response | {stats}[/dim]")
