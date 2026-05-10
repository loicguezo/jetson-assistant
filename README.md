# Jetson Voice Assistant

<p align="center">
  <a href="https://developer.nvidia.com/embedded/jetson-orin-nano"><img src="docs/images/jetson-family.png" alt="NVIDIA Jetson" height="180"/></a>
</p>

A low-latency, fully on-device voice assistant for NVIDIA Jetson. Everything runs locally with GPU acceleration — no cloud, no API keys, no internet required at runtime.

> **Current target:** Jetson Orin Nano 8GB (JetPack 6.x, Python 3.10)

## What It Does

Speak into a microphone and the assistant responds using a local LLM. Speech is detected automatically via VAD, transcribed via Whisper, answered by the LLM, and spoken back via TTS.

```
[Mic] → [Silero VAD] → [faster-whisper STT] → [LLM stream] → [TTS stream] → [Speaker]
```

## Stack

| Component | Library | Acceleration |
|-----------|---------|:---:|
| **LLM** | llama.cpp (native, no Docker) | GPU (CUDA) |
| **STT** | faster-whisper | GPU (CUDA) |
| **TTS** | Kokoro ONNX | GPU (CUDA) |
| **VAD** | Silero VAD | CPU |

llama.cpp is compiled directly on the Jetson — no Docker, no Python wrapper overhead. This keeps the memory footprint as small as possible on the shared 8 GB unified memory.

Kokoro TTS runs in a **separate subprocess** to isolate its GPL-licensed dependencies (phonemizer, espeak-ng) from the CUDA process. ONNX Runtime GPU (`onnxruntime-gpu` from the Jetson AI Lab index) enables `CUDAExecutionProvider`, reducing TTS RTF from ~1.17x to ~0.14x (8x faster).

**Default model:** [Gemma 4 E4B Q4_K_M](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) (~4.6 GB) — Google's Gemma 4 Efficient 4B, quantized by unsloth. Any GGUF model placed in `~/models/` is picked up automatically by `./jetson-assistant start`.

## Prerequisites

- **NVIDIA Jetson Orin Nano** (8GB) with JetPack 6.x, Python 3.10
- **USB microphone** and **speaker** (or Bluetooth audio device)
- **NVMe SSD** recommended for swap and model storage
- **HuggingFace account** — required to download models (`hf auth login`)

## Setup

See **[SETUP.md](SETUP.md)** for the full installation guide — dependencies, building llama.cpp, Python packages, model downloads, and troubleshooting.

## Usage

```bash
./jetson-assistant setup            # first-time setup: build llama.cpp, download model, create venv
./jetson-assistant start            # model picker → llama-server → voice chat
./jetson-assistant stop             # stop everything
./jetson-assistant status           # show what's running + memory usage
./jetson-assistant optimize         # apply memory optimizations (reversible, asks per item)
./jetson-assistant optimize --all       # apply all without prompting
./jetson-assistant optimize --restore   # undo optimizations
./jetson-assistant optimize --status    # show what is applied
./jetson-assistant test --llm       # test LLM (auto-starts llama-server)
./jetson-assistant test --stt       # test speech-to-text (records 3 seconds)
./jetson-assistant test --tts       # test text-to-speech (plays a sentence)
./jetson-assistant test --vad       # test VAD (shows mic activity for 5 seconds)
./jetson-assistant test --mic       # test microphone (lists devices, records 3s, plays back)
./jetson-assistant test --all       # run all component tests
./jetson-assistant benchmark        # TTS → STT → LLM chain with fixed inputs, reports timing
```

Add the project directory to `PATH` to use `jetson-assistant` from anywhere:
```bash
echo 'export PATH="$HOME/workspace/jetson-assistant:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Configuration

All settings live in `config/settings.yaml`:

| Section | What It Controls |
|---------|-----------------|
| `llm` | Server URL, model, temperature, max tokens, system prompt |
| `stt` | Whisper model size, CUDA device, beam size, language |
| `tts` | Voice, speed, language code, chunking |
| `audio` | Sample rate, input device name hint |
| `vad` | Silero threshold, silence duration, utterance filters |

**Selecting your microphone:** set `audio.input_device` in `settings.yaml` to a substring of your device name (check `arecord -l` or run `./jetson-assistant test --mic`), or leave it `null` to auto-detect.

**Language:** set `stt.language` (e.g. `"de"`, `"fr"`) and use a matching Kokoro voice + `tts.lang` code (e.g. `"de"`, `"fr-fr"`). Switch the STT model from `small.en` to `small` for multilingual transcription.

**Bluetooth speaker:** use `scripts/connect-bt-speaker.sh` to pair and set a Bluetooth speaker as the default PulseAudio sink.

## Project Structure

```
jetson-assistant/
├── app/
│   ├── pipeline.py       # Audio I/O, VAD loop, TTS streaming
│   ├── config.py         # Configuration dataclasses + YAML loader
│   ├── llm.py            # LLM client (OpenAI-compatible, streams to llama-server)
│   ├── stt.py            # faster-whisper speech-to-text
│   ├── tts.py            # TTS client (spawns subprocess worker)
│   ├── tts_worker.py     # TTS subprocess (Kokoro + GPL deps, isolated)
│   ├── manager.py        # llama-server lifecycle, PID files, GGUF discovery
│   ├── optimize.py       # System optimizations with state persistence
│   ├── setup_wizard.py   # First-time setup logic (build, download, venv)
│   ├── benchmark.py      # Full-pipeline benchmark (TTS → STT → LLM)
│   ├── test_components.py # Component tests (LLM, STT, TTS, VAD, mic)
│   ├── monitor.py        # CPU/GPU/RAM stats
│   └── audio.py          # PulseAudio / ALSA device helpers
├── config/
│   └── settings.yaml     # All runtime configuration
├── scripts/
│   └── connect-bt-speaker.sh  # Bluetooth speaker setup helper
├── voices/               # TTS voice files (gitignored)
├── manage.py             # ./jetson-assistant CLI entry point
└── run_voice_chat.py     # Voice pipeline entry point
```

## Performance (Orin Nano 8GB)

Measured with `./jetson-assistant benchmark` — fixed inputs, reproducible across runs.

| Component | Time | Model |
|-----------|-----:|-------|
| TTS (synthesis) | 3.66 s | Kokoro `af_sarah`, CPU |
| STT (transcribe) | 1.19 s | faster-whisper `small.en`, CUDA |
| LLM (time to first token) | 3.55 s | Gemma 4 E4B Q4_K_M |
| LLM (full response) | 4.15 s | Gemma 4 E4B Q4_K_M |
| **Total** | **9.01 s** | TTS + STT + LLM |

## Roadmap

- [x] Orin Nano 8GB — full pipeline validated
- [x] Silero VAD for robust speech detection
- [x] Native llama.cpp (no Docker overhead)
- [x] Management CLI (`setup`, `start`, `stop`, `status`, `optimize`, `test`, `benchmark`)
- [x] Memory optimizations (`optimize` command — reversible, per-item dialog)
- [x] Bluetooth speaker support (`scripts/connect-bt-speaker.sh`)
- [x] Language configurable via `settings.yaml` (`stt.language`, `tts.lang` + voice)
- [x] TTS GPU acceleration (Kokoro via `onnxruntime-gpu`, RTF ~0.14x)
- [x] Multi-turn conversation memory (rolling window, persistent across sessions)
- [ ] Auto-start as systemd service (boot without manual `./jetson-assistant start`)
- [ ] Wake word detection (hands-free activation)
- [ ] Wake word detection (hands-free activation)

## Further Reading

- [AI models that run on Jetson Orin Nano Super 8GB — a practical guide](https://forums.developer.nvidia.com/t/ai-models-that-run-on-jetson-orin-nano-super-8gb-a-practical-guide/365412) — model selection, benchmarks, and what fits in 8 GB
- [Maximizing Memory Efficiency on NVIDIA Jetson](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/) — the techniques behind `./jetson-assistant optimize`

## Troubleshooting

See [SETUP.md](SETUP.md#troubleshooting) for common issues and fixes.

## License Notes

This project uses [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) for text-to-speech. Kokoro ONNX itself is MIT-licensed, but it depends on:

- **phonemizer-fork** — GPL-3.0
- **espeak-ng** — GPL-3.0

To avoid loading GPL-licensed code into the same process as NVIDIA CUDA libraries, TTS runs in a **separate subprocess** (`app/tts_worker.py`). The main process never imports `kokoro-onnx` directly — it communicates with the worker via JSON over stdin/stdout.

All other dependencies use permissive licenses (MIT, BSD-3, Apache-2.0). See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full list.

## Contributing

We welcome community contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, including the Developer Certificate of Origin (DCO) sign-off requirement.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Attribution

Forked from [NVIDIA-AI-IOT/reachy-mini-jetson-assistant](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant). Original work copyright © 2023–2025 NVIDIA CORPORATION & AFFILIATES.
