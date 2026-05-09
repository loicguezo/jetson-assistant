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

**Default model:** [Gemma 4 E4B Q4_K_M](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) (~4.6 GB) — Google's Gemma 4 Efficient 4B, quantized by unsloth. Any GGUF model placed in `~/models/` is picked up automatically by `./jetson-assistant start`.

## Prerequisites

- **NVIDIA Jetson Orin Nano** (8GB) with JetPack 6.x, Python 3.10
- **USB microphone** and **speaker** (or USB audio device)
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
./jetson-assistant optimize         # apply memory optimizations (reversible)
./jetson-assistant optimize --restore   # undo optimizations
./jetson-assistant optimize --status    # show what is applied
./jetson-assistant test --llm       # test individual components
./jetson-assistant test --stt
./jetson-assistant test --tts
./jetson-assistant test --vad
./jetson-assistant test --all
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
| `stt` | Whisper model size, CUDA device, beam size |
| `tts` | Voice, speed, language, chunking |
| `audio` | Sample rate, input device name hint |
| `vad` | Silero threshold, silence duration, utterance filters |

**Selecting your microphone:** by default the assistant searches for a device matching `"USB Audio"`. Set `audio.input_device` in `settings.yaml` to a substring of your device name (check `arecord -l`), or leave it `null` to auto-detect the first available input.

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
│   ├── monitor.py        # CPU/GPU/RAM stats
│   └── audio.py          # PulseAudio / ALSA device helpers
├── config/
│   └── settings.yaml     # All runtime configuration
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
- [x] Kokoro TTS GPU acceleration
- [x] Silero VAD for robust speech detection
- [x] Native llama.cpp (no Docker overhead)
- [ ] Wake word detection (hands-free activation)
- [ ] Multi-turn conversation memory
- [ ] Multi-language support
- [ ] Auto-start llama-server as systemd service

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
