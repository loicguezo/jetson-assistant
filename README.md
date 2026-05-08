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

## Prerequisites

- **NVIDIA Jetson Orin Nano** (8GB) with JetPack 6.x, Python 3.10
- **USB microphone** and **speaker** (or USB audio device)
- **NVMe SSD** recommended for swap and model storage

## Setup

See **[SETUP.md](SETUP.md)** for the full installation guide — dependencies, building llama.cpp, Python packages, model downloads, and troubleshooting.

## Usage

### Voice Chat

**Terminal 1** — Start the LLM server:

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-3-4b-it-Q4_K_M.gguf \
  --port 8080 --host 127.0.0.1 -ngl 99 -c 4096
```

**Terminal 2** — Start the assistant:

```bash
source venv/bin/activate
python3 run_voice_chat.py
```

Speak anytime — the assistant auto-detects speech. Press **Ctrl+C** to quit.

### CLI (Text Only)

```bash
source venv/bin/activate
python3 main.py chat                            # interactive text chat
python3 main.py ask "What is the Jetson Orin?"  # single question
python3 main.py info                            # system info + dependency check
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
│   ├── pipeline.py      # Audio I/O, VAD, TTS streaming, mic recording
│   ├── config.py        # Configuration dataclasses + YAML loader
│   ├── llm.py           # LLM client (OpenAI-compatible API)
│   ├── stt.py           # faster-whisper speech-to-text
│   ├── tts.py           # TTS client (spawns subprocess worker)
│   ├── tts_worker.py    # TTS subprocess (Kokoro + GPL deps, isolated)
│   ├── monitor.py       # System resource monitoring (CPU/GPU/RAM)
│   ├── audio.py         # PulseAudio / ALSA device helpers
│   └── cli.py           # Typer CLI (chat, ask, info)
├── config/
│   └── settings.yaml    # All runtime configuration
├── voices/              # TTS voice files (gitignored)
├── run_voice_chat.py    # Main voice assistant entry point
└── main.py              # CLI entry point
```

## Performance (Orin Nano 8GB)

| Metric | Value |
|--------|-------|
| STT latency | ~0.7s (small.en, beam=1) |
| LLM TTFT | ~1–2s (Gemma 3 4B Q4_K_M, warm) |
| TTS latency (first chunk) | ~0.3s (Kokoro GPU) |
| End-to-end (speak → response) | ~2–4s |
| Peak RAM | ~5–6 GB (STT + LLM + TTS) |

## Roadmap

- [x] Orin Nano 8GB — full pipeline validated
- [x] Kokoro TTS GPU acceleration
- [x] Silero VAD for robust speech detection
- [x] Native llama.cpp (no Docker overhead)
- [ ] Wake word detection (hands-free activation)
- [ ] Multi-turn conversation memory
- [ ] Multi-language support
- [ ] Auto-start llama-server as systemd service

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
