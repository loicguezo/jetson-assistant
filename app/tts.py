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
#
# TTS — subprocess-isolated Kokoro TTS.
#
# kokoro-onnx depends on phonemizer-fork (GPL-3.0) and espeak-ng (GPL-3.0).
# To avoid loading GPL code into the same process as NVIDIA CUDA libraries,
# synthesis runs in a separate subprocess (app/tts_worker.py) that
# communicates via JSON lines over stdin/stdout.

import base64
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
PIPER_DIR = VOICES_DIR / "piper"

PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


def _download_kokoro_models_if_missing() -> bool:
    """Download Kokoro model and voices to voices/ if not present."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = VOICES_DIR / "kokoro-v1.0.onnx"
    voices_path = VOICES_DIR / "voices-v1.0.bin"
    needed = []
    if not model_path.exists():
        needed.append((KOKORO_MODEL_URL, model_path, "kokoro-v1.0.onnx (~311 MB)"))
    if not voices_path.exists():
        needed.append((KOKORO_VOICES_URL, voices_path, "voices-v1.0.bin (~30 MB)"))
    if not needed:
        return True
    try:
        import httpx
    except ImportError:
        print("Kokoro: install httpx to auto-download models (pip install httpx)")
        return False
    for url, path, label in needed:
        print(f"Downloading {label} to {path} ...")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                done = 0
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=262144):
                        f.write(chunk)
                        done += len(chunk)
                        if total and total > 0:
                            pct = 100 * done / total
                            sys.stdout.write(f"\r  {label}: {pct:.0f}%\r")
                            sys.stdout.flush()
            if total:
                print()
            print(f"  Saved {path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            if path.exists():
                path.unlink()  # remove partial file so next run retries
            return False
    return True


class KokoroTTS:
    """Kokoro TTS client — synthesis runs in a subprocess for GPL isolation."""

    def __init__(self, voice: str = "af_sarah", speed: float = 1.0, lang: str = "en-us"):
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._proc: subprocess.Popen | None = None
        self._sample_rate = 24000
        self.backend_name = "Kokoro"
        self.provider = "unknown"

    def load(self) -> bool:
        model_path = VOICES_DIR / "kokoro-v1.0.onnx"
        voices_path = VOICES_DIR / "voices-v1.0.bin"
        if not model_path.exists() or not voices_path.exists():
            if not _download_kokoro_models_if_missing():
                return False

        worker = Path(__file__).parent / "tts_worker.py"
        try:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    str(worker),
                    "--model-dir",
                    str(VOICES_DIR),
                    "--voice",
                    self.voice,
                    "--speed",
                    str(self.speed),
                    "--lang",
                    self.lang,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # filtered below — see _stderr_filter thread
                text=True,
                bufsize=1,
            )
        except Exception as e:
            print(f"TTS worker spawn failed: {e}")
            return False

        # Forward worker stderr, suppressing a harmless ORT startup warning:
        # ORT probes /sys/class/drm/card0/device/vendor (PCI GPU enumeration);
        # on Jetson that path doesn't exist (SoC GPU uses nvgpu, not PCIe),
        # so ORT logs a benign "GPU device discovery failed" probe error even
        # when CUDAExecutionProvider is active and working correctly.
        import threading

        _SUPPRESS = [
            "DiscoverDevicesForPlatform",
            "device_discovery.cc",
            "GPU device discovery failed",
            # ORT graph-compilation notes: Memcpy nodes inserted for CPU↔GPU
            # data movement, shape ops kept on CPU intentionally, ScatterND
            # index-uniqueness caveat from the Kokoro model itself — all harmless.
            "Memcpy nodes are added",
            "VerifyEachNodeIsAssignedToAnEp",
            "ScatterNDWithAtomicReduction",
            "transformer_memcpy.cc",
            "session_state.cc",
            "scatter_nd.h",
        ]

        stderr_pipe = self._proc.stderr

        def _stderr_filter():
            for line in stderr_pipe:
                if not any(s in line for s in _SUPPRESS):
                    sys.stderr.write(line)

        threading.Thread(target=_stderr_filter, daemon=True).start()

        import select

        ready = select.select([self._proc.stdout], [], [], 30.0)[0]
        if not ready:
            print("TTS worker startup timed out")
            self._proc.kill()
            self._proc.wait()
            return False
        line = self._proc.stdout.readline()
        if not line:
            print("TTS worker exited before signalling ready")
            return False

        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            print(f"TTS worker sent invalid init response: {line!r}")
            return False

        if resp.get("status") == "ready":
            self.provider = resp.get("provider", "unknown")
            return True

        print(f"TTS worker error: {resp.get('error', 'unknown')}")
        return False

    def _send(self, req: dict) -> dict | None:
        if not self._proc or self._proc.poll() is not None:
            return None
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                return None
            return json.loads(line)
        except (BrokenPipeError, json.JSONDecodeError, OSError):
            return None

    def synthesize(self, text: str) -> dict[str, Any]:
        if not text.strip():
            return {"audio": None, "error": "Empty"}

        resp = self._send(
            {
                "cmd": "synthesize",
                "text": text,
                "voice": self.voice,
                "speed": self.speed,
                "lang": self.lang,
            }
        )
        if resp is None:
            return {"audio": None, "error": "Worker not running"}
        if "error" in resp:
            return {"audio": None, "error": resp["error"]}

        audio_bytes = base64.b64decode(resp["audio_b64"])
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        return {"audio": audio, "sample_rate": resp["sample_rate"]}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        resp = self._send({"cmd": "health"})
        return resp is not None and resp.get("healthy", False)

    def unload(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
                self._proc.wait()
            self._proc = None


def _download_file(url: str, path: Path, label: str) -> bool:
    """Download a single file with progress. Returns True on success."""
    try:
        import httpx
    except ImportError:
        print(f"Install httpx to auto-download {label} (pip install httpx)")
        return False
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            done = 0
            with open(path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=262144):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        sys.stdout.write(f"\r  {label}: {100 * done / total:.0f}%")
                        sys.stdout.flush()
        if total:
            print()
        print(f"  Saved {path}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        if path.exists():
            path.unlink()
        return False


def _parse_piper_model(model_name: str) -> tuple[str, str, str, str]:
    """Parse 'de_DE-thorsten-medium' into (lang, locale, speaker, quality)."""
    parts = model_name.split("-")
    locale = parts[0]  # e.g. de_DE
    quality = parts[-1]  # low / medium / high
    speaker = "-".join(parts[1:-1])
    lang = locale.split("_")[0]
    return lang, locale, speaker, quality


def download_piper_model_if_missing(model_name: str = "en_US-lessac-medium") -> bool:
    """Download Piper voice model (.onnx + .onnx.json) to voices/piper/ if not present."""
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    model_onnx = PIPER_DIR / f"{model_name}.onnx"
    model_json = PIPER_DIR / f"{model_name}.onnx.json"
    if model_onnx.exists() and model_json.exists():
        return True
    lang, locale, speaker, quality = _parse_piper_model(model_name)
    base = f"{PIPER_VOICES_BASE}/{lang}/{locale}/{speaker}/{quality}/{model_name}"
    print(f"Downloading Piper voice model {model_name}...")
    if not _download_file(f"{base}.onnx", model_onnx, f"{model_name}.onnx"):
        return False
    return _download_file(f"{base}.onnx.json", model_json, f"{model_name}.onnx.json")


class PiperTTS:
    """Piper TTS via the piper-tts Python package (OHF-Voice/piper1-gpl).

    Uses the Python API directly — no subprocess, no binary download.
    Installed with --no-deps to avoid overwriting onnxruntime-gpu with the
    CPU-only onnxruntime wheel that piper-tts declares as a dependency.
    Speed is controlled via SynthesisConfig.length_scale (inverse of speed).
    """

    def __init__(self, model: str = "en_US-lessac-medium", speed: float = 1.0):
        self.model = model
        self.speed = speed
        self.voice = model
        self.backend_name = "Piper"
        self.provider = "CPU"
        self._piper_voice = None
        self._sample_rate = 22050

    def load(self) -> bool:
        try:
            from piper.voice import PiperVoice
        except ImportError:
            print(
                "piper-tts not installed. Run:\n"
                "  pip install piper-tts --no-deps && pip install pathvalidate"
            )
            return False

        if not download_piper_model_if_missing(self.model):
            return False

        model_path = PIPER_DIR / f"{self.model}.onnx"
        config_path = PIPER_DIR / f"{self.model}.onnx.json"
        try:
            import onnxruntime as ort

            use_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            self._piper_voice = PiperVoice.load(
                model_path, config_path=config_path, use_cuda=use_cuda
            )
            self._sample_rate = self._piper_voice.config.sample_rate
            provider = "CUDA" if use_cuda else "CPU"
            self.provider = provider
            print(f"Piper TTS loaded — model: {self.model}, {self._sample_rate} Hz, {provider}")
        except Exception as e:
            print(f"Piper load failed: {e}")
            return False

        # Quick sanity check
        result = self.synthesize("test")
        if result.get("audio") is None:
            print(f"Piper self-test failed: {result.get('error')}")
            return False

        return True

    def synthesize(self, text: str) -> dict[str, Any]:
        if not text.strip():
            return {"audio": None, "error": "Empty"}
        if self._piper_voice is None:
            return {"audio": None, "error": "Not loaded"}

        try:
            from piper.config import SynthesisConfig

            syn_cfg = SynthesisConfig(length_scale=1.0 / max(self.speed, 0.1))
            chunks = list(self._piper_voice.synthesize(text, syn_config=syn_cfg))
            if not chunks:
                return {"audio": None, "error": "Piper produced no audio"}
            # Concatenate float32 chunks and convert to int16
            audio_f32 = np.concatenate([c.audio_float_array for c in chunks])
            audio_i16 = np.clip(audio_f32 * 32767, -32768, 32767).astype(np.int16)
            return {"audio": audio_i16, "sample_rate": self._sample_rate}
        except Exception as e:
            return {"audio": None, "error": str(e)}

    def synthesize_to_file(self, text: str, path: str) -> bool:
        r = self.synthesize(text)
        if r.get("audio") is None:
            return False
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(r["sample_rate"])
            wf.writeframes(r["audio"].tobytes())
        return True

    def health_check(self) -> bool:
        return self._piper_voice is not None

    def unload(self):
        self._piper_voice = None


def create_tts(
    backend: str = "kokoro",
    voice: str = "",
    speed: float = 1.0,
    lang: str = "en-us",
    piper_model: str = "",
    **_kwargs,
):
    """Create a TTS backend. backend='kokoro' (default) or 'piper'."""
    if backend == "piper":
        return PiperTTS(model=piper_model or "en_US-lessac-medium", speed=speed)
    return KokoroTTS(voice=voice or "af_sarah", speed=speed, lang=lang)
