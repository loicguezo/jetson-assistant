# Third-Party Software Notices

This project uses the following third-party open source software.

## Direct Dependencies

| Package | License | URL |
|---------|---------|-----|
| PyYAML | MIT | https://github.com/yaml/pyyaml |
| Rich | MIT | https://github.com/Textualize/rich |
| Typer | MIT | https://github.com/fastapi/typer |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| sounddevice | MIT | https://github.com/spatialaudio/python-sounddevice |
| Silero VAD | MIT | https://github.com/snakers4/silero-vad |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| faster-whisper | MIT | https://github.com/SYSTRAN/faster-whisper |
| kokoro-onnx | MIT | https://github.com/thewh1teagle/kokoro-onnx |

## Separately Installed Dependencies

| Package | License | URL |
|---------|---------|-----|
| onnxruntime-gpu | MIT | https://github.com/microsoft/onnxruntime |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| llama.cpp | MIT | https://github.com/ggml-org/llama.cpp |

## Key Transitive Dependencies

| Package | License | URL |
|---------|---------|-----|
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| espeakng-loader | MIT | https://github.com/thewh1teagle/espeakng-loader |

## GPL-Licensed Transitive Dependencies (Subprocess-Isolated)

The following GPL-licensed packages are transitive dependencies of `kokoro-onnx`
(MIT). They run in a **separate subprocess** (`app/tts_worker.py`) that does not
load any NVIDIA proprietary libraries. The main application process never imports
these packages. Communication between processes uses JSON over stdin/stdout pipes
(standard IPC), which does not constitute linking under GPL.

| Package | License | URL |
|---------|---------|-----|
| phonemizer-fork | GPL-3.0 | https://github.com/thewh1teagle/phonemizer |
| espeak-ng | GPL-3.0 | https://github.com/espeak-ng/espeak-ng |

## Model Licenses

| Model | License | URL |
|-------|---------|-----|
| faster-whisper (small.en) | MIT | https://huggingface.co/Systran/faster-whisper-small.en |
| Kokoro v1.0 | Apache-2.0 | https://huggingface.co/hexgrad/Kokoro-82M |
