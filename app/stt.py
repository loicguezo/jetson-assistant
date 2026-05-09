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

"""STT — faster-whisper, GPU-accelerated Whisper on Jetson."""

from typing import Dict, Any, Union
import numpy as np


class STT:
    def __init__(
        self,
        model: str = "base.en",
        device: str = "cuda",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model = None

    def load(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
            return True
        except Exception as e:
            print(f"faster-whisper load error: {e}")
            try:
                from faster_whisper import WhisperModel
                print("Falling back to CPU...")
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                self.device = "cpu"
                return True
            except Exception as e2:
                print(f"CPU fallback failed: {e2}")
                return False

    def transcribe(self, audio: Union[np.ndarray, str], sample_rate: int = 16000) -> Dict[str, Any]:
        if self._model is None:
            return {"text": "", "error": "Model not loaded"}
        try:
            if isinstance(audio, np.ndarray):
                if audio.size == 0:
                    return {"text": "", "error": "Empty audio"}
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = audio.flatten().astype(np.float32)
                if np.abs(audio).max() > 1.5:
                    audio = audio / 32768.0

            segments, info = self._model.transcribe(
                audio, language=self.language, beam_size=self.beam_size,
                no_speech_threshold=0.1, log_prob_threshold=-1.0,
            )
            text = " ".join(s.text for s in segments).strip()
            return {"text": text, "language": info.language, "duration": info.duration}
        except Exception as e:
            return {"text": "", "error": str(e)}

    def get_info(self) -> Dict[str, Any]:
        return {"backend": "faster-whisper", "model": self.model_name, "device": self.device}

    def health_check(self) -> bool:
        return self._model is not None

    def unload(self):
        if self._model:
            del self._model
            self._model = None
