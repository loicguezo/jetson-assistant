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

"""Configuration — loads settings.yaml into typed dataclasses."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    model: str = ""
    base_url: str = "http://localhost:8080"
    backend: str = "openai"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: float = 120.0
    system_prompt: str = "You are a helpful AI assistant."
    memory_turns: int = 5  # number of past turns to include as context (0 = disabled)


@dataclass
class STTConfig:
    model: str = "base.en"
    device: str = "cuda"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1


@dataclass
class TTSConfig:
    backend: str = "kokoro"  # kokoro or piper
    voice: str = "af_sarah"
    speed: float = 0.8
    lang: str = "en-us"
    first_chunk_words: int = 3
    max_chunk_words: int = 8
    piper_model: str = "en_US-lessac-medium"
    ready_phrase: str = "Ready!"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    input_device: str | None = None


@dataclass
class AppConfig:
    mode: str = "voice"  # voice or text


@dataclass
class VADConfig:
    speech_threshold: float = 0.008
    silence_duration_ms: int = 500
    lookback_ms: int = 250
    max_speech_secs: int = 15
    chunk_ms: int = 30
    min_utterance_secs: float = 0.3
    min_utterance_rms: float = 0.005
    use_silero: bool = False
    silero_threshold: float = 0.5


_SECTIONS = [
    ("app", "app", AppConfig),
    ("llm", "llm", LLMConfig),
    ("stt", "stt", STTConfig),
    ("tts", "tts", TTSConfig),
    ("audio", "audio", AudioConfig),
    ("vad", "vad", VADConfig),
]


def _config_paths(base: Path) -> list[Path]:
    """Return [settings.yaml, settings.local.yaml] — local overrides base."""
    return [base, base.with_name("settings.local.yaml")]


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        config = cls()
        for path in _config_paths(Path(config_path)):
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                for yaml_key, attr_name, _ in _SECTIONS:
                    section_obj = getattr(config, attr_name)
                    for k, v in data.get(yaml_key, {}).items():
                        if hasattr(section_obj, k):
                            setattr(section_obj, k, v)
            except Exception as e:
                print(f"Error loading config {path.name}: {e}")
        config._apply_env_overrides()
        return config

    def _apply_env_overrides(self):
        """Override config values from JA_* environment variables (set by CLI flags)."""
        if v := os.environ.get("JA_MAX_TOKENS"):
            self.llm.max_tokens = int(v)
        if v := os.environ.get("JA_TEMPERATURE"):
            self.llm.temperature = float(v)
        if v := os.environ.get("JA_TTS_SPEED"):
            self.tts.speed = float(v)
        if v := os.environ.get("JA_FIRST_CHUNK_WORDS"):
            self.tts.first_chunk_words = int(v)
        if v := os.environ.get("JA_MAX_CHUNK_WORDS"):
            self.tts.max_chunk_words = int(v)
        if v := os.environ.get("JA_TTS_BACKEND"):
            self.tts.backend = v
        if v := os.environ.get("JA_PIPER_MODEL"):
            self.tts.piper_model = v
