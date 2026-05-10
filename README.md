# Jetson Voice Assistant

<p align="center">
  <a href="https://developer.nvidia.com/embedded/jetson-orin-nano"><img src="docs/images/jetson-family.png" alt="NVIDIA Jetson" height="180"/></a>
</p>

A low-latency, fully on-device voice and text assistant for NVIDIA Jetson. Everything runs locally with GPU acceleration — no cloud, no API keys, no internet required at runtime.

> **Current target:** Jetson Orin Nano 8GB (JetPack 6.x, Python 3.10)

## What It Does

Speak into a microphone (or type) and the assistant responds using a local LLM. Speech is detected automatically via VAD, transcribed via Whisper, answered by the LLM, and spoken back via TTS.

```
[Mic] → [Silero VAD] → [faster-whisper STT] → [LLM stream] → [TTS stream] → [Speaker]
[Keyboard] ──────────────────────────────────↗
```

## Stack

| Component | Library | Acceleration |
|-----------|---------|:---:|
| **LLM** | llama.cpp (native, no Docker) | GPU (CUDA) |
| **STT** | faster-whisper | GPU (CUDA) |
| **TTS** | Piper *(default)* or Kokoro | GPU (CUDA) |
| **VAD** | Silero VAD | CPU |

**TTS backends:** Piper (default, multilingual, CPU+CUDA) and Kokoro (English, CUDA) are both supported. Switch via `./jetson-assistant config` or CLI flags per session.

llama.cpp is compiled directly on the Jetson — no Docker, no Python wrapper overhead. This keeps the memory footprint as small as possible on the shared 8 GB unified memory.

**Default model:** [Gemma 4 E4B Q4_K_M](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) (~4.6 GB) — Google's Gemma 4 Efficient 4B, quantized by unsloth. Any GGUF model placed in `~/models/` is picked up automatically.

## Prerequisites

- **NVIDIA Jetson Orin Nano** (8GB) with JetPack 6.x, Python 3.10
- **USB microphone** and **speaker** (voice mode only)
- **NVMe SSD** recommended for swap and model storage
- **HuggingFace account** — required to download models (`hf auth login`)

## Setup

See **[SETUP.md](SETUP.md)** for the full installation guide — dependencies, building llama.cpp, Python packages, model downloads, and troubleshooting.

```bash
./jetson-assistant setup   # builds llama.cpp, downloads model + TTS/STT models,
                           # and walks through personal configuration (Step 8/8)
```

## Usage

```bash
./jetson-assistant start            # show settings, launch llama-server, start voice chat
./jetson-assistant start --text     # text mode — type prompts, no microphone required
./jetson-assistant config           # change language, TTS voice, STT model, LLM model
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

Per-session overrides via CLI flags:
```bash
./jetson-assistant start --tts-backend piper --piper-model de_DE-thorsten-high
./jetson-assistant start --max-tokens 256 --temperature 0.5
./jetson-assistant start --tts-speed 1.2 --first-chunk-words 4 --max-chunk-words 10
```

Add the project directory to `PATH` to use `jetson-assistant` from anywhere:
```bash
echo 'export PATH="$HOME/workspace/jetson-assistant:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Text Mode

Text mode lets you chat with the assistant without a microphone — useful for testing, SSH sessions, or when you just prefer typing. Responses are still spoken aloud via TTS.

```bash
./jetson-assistant start --text
```

```
╭──────────────────────────────────────────────────────────────╮
│ Text Assistant                                               │
│ Type your message — response is spoken aloud                 │
│ 'quit' to exit · 'stats' for system info · Ctrl-C to quit   │
╰──────────────────────────────────────────────────────────────╯

Loading...
  ✓ LLM (gemma-4-E4B-it-Q4_K_M.gguf)  +0.0GB → 5.0GB
  ✓ TTS (Piper, en_US-lessac-medium)   +0.6GB → 5.6GB

Ready!

You: What is a NVIDIA Jetson in one sentence?
Assistant: An NVIDIA Jetson is a family of small, power-efficient, single-board computers
specifically designed to enable on-device artificial intelligence and edge computing
applications, allowing complex AI processing to happen locally rather than in the cloud.
  TTFT 0.9s | LLM 4.6s ~7w/s | RAM 6.1/7.4GB GPU 52°C 918MHz

You: What makes this assistant different from cloud-based ones?
Assistant: This assistant runs directly on an NVIDIA Jetson, meaning processing occurs
locally on the device itself, not on remote servers. This edge computing capability
provides significantly lower latency, as data does not need to travel to the cloud
and back. Furthermore, it offers enhanced privacy because sensitive data remains within
the local hardware, making it ideal for real-time, offline deployments.
  TTFT 0.3s | LLM 6.5s ~9w/s | RAM 6.4/7.4GB GPU 53°C 918MHz
```

Special commands in both modes: `stats` (system info), `forget` (clear conversation history).

## Configuration

Personal settings are stored in `config/settings.local.yaml` (gitignored, never committed). Run the interactive wizard to configure everything in one flow:

```bash
./jetson-assistant config
```

The wizard covers: interface mode, language, TTS backend and voice, STT model, and LLM model. On first `./jetson-assistant start`, it runs automatically if no local config exists.

Every `start` shows your active settings before launching, with a 5-second window to jump into the wizard:

```
  Mode      voice
  Language  de  ·  STT: base
  TTS       piper · de_DE-thorsten-high · speed 0.8
  LLM       gemma-4-E4B-it-Q4_K_M.gguf
  Prompt    Du bist ein hilfreicher Sprachassistent auf einem NVI…

  Change settings? [n]  —  auto in 5s
```

For advanced settings (`max_tokens`, `temperature`, VAD thresholds, audio device), edit `config/settings.yaml` directly:

| Section | What It Controls |
|---------|-----------------|
| `llm` | Server URL, model, temperature, max tokens, system prompt, memory turns |
| `stt` | Whisper model size, CUDA device, beam size, language |
| `tts` | Backend, voice, speed, chunking parameters |
| `audio` | Sample rate, input device name hint |
| `vad` | Silero threshold, silence duration, utterance filters |

**Selecting your microphone:** set `audio.input_device` in `settings.yaml` to a substring of your device name (check `arecord -l` or run `./jetson-assistant test --mic`), or leave it `null` to auto-detect.

**Bluetooth speaker:** use `scripts/connect-bt-speaker.sh` to pair and set a Bluetooth speaker as the default PulseAudio sink.

## Project Structure

```
jetson-assistant/
├── app/
│   ├── pipeline.py         # Audio I/O, VAD loop, TTS streaming
│   ├── config.py           # Configuration dataclasses + YAML loader
│   ├── llm.py              # LLM client (OpenAI-compatible, streams to llama-server)
│   ├── stt.py              # faster-whisper speech-to-text
│   ├── tts.py              # TTS client (Piper in-process, Kokoro subprocess)
│   ├── tts_worker.py       # Kokoro TTS subprocess (GPL deps isolated)
│   ├── history.py          # Persistent conversation history (rolling window)
│   ├── manager.py          # llama-server lifecycle, PID files, GGUF discovery
│   ├── optimize.py         # System optimizations with state persistence
│   ├── setup_wizard.py     # First-time setup logic (build, download, venv)
│   ├── benchmark.py        # Full-pipeline benchmark (TTS → STT → LLM)
│   ├── test_components.py  # Component tests (LLM, STT, TTS, VAD, mic)
│   ├── monitor.py          # CPU/GPU/RAM stats
│   └── audio.py            # PulseAudio / ALSA device helpers
├── config/
│   ├── settings.yaml       # Default runtime configuration
│   └── settings.local.yaml # Personal overrides (gitignored)
├── scripts/
│   └── connect-bt-speaker.sh  # Bluetooth speaker setup helper
├── voices/                 # TTS voice files (gitignored)
├── manage.py               # ./jetson-assistant CLI entry point
├── run_voice_chat.py       # Voice pipeline entry point
└── run_text_chat.py        # Text pipeline entry point
```

## Performance (Orin Nano 8GB)

Measured with `./jetson-assistant benchmark` — fixed inputs, reproducible across runs.

| Component | Time | Model |
|-----------|-----:|-------|
| TTS (synthesis) | ~0.47 s | Piper `de_DE-thorsten-high`, CUDA (RTF 0.13×) |
| STT (transcribe) | ~1.19 s | faster-whisper `small.en`, CUDA |
| LLM (time to first token) | 0.3–0.9 s | Gemma 4 E4B Q4_K_M |
| LLM (full response) | 4–7 s | Gemma 4 E4B Q4_K_M (~7–9 w/s) |

### TTS Backend Comparison

Both backends use CUDA via `onnxruntime-gpu`. Measured on Orin Nano 8GB with a ~3.6 s sentence (3 warm runs averaged).

| Backend | Voice | Language | Avg synthesis | RTF (warm) |
|---------|-------|----------|:-------------:|:----------:|
| **Piper** *(default)* | `de_DE-thorsten-high` | German | ~0.47 s | 0.13× |
| **Kokoro** | `af_sarah` | English | ~0.56 s | 0.16× |

Piper synthesises in-process (no subprocess IPC overhead). Kokoro runs in a subprocess for GPL isolation — that adds one JSON round-trip per utterance.

Run your own comparison:
```bash
./jetson-assistant benchmark --tts-backend piper --piper-model de_DE-thorsten-high
./jetson-assistant benchmark --tts-backend kokoro
```

> RTF = synthesis time ÷ audio duration. RTF < 1.0 means faster than real-time.

## Roadmap

- [x] Orin Nano 8GB — full pipeline validated
- [x] Silero VAD for robust speech detection
- [x] Native llama.cpp (no Docker overhead)
- [x] Management CLI (`setup`, `start`, `stop`, `status`, `optimize`, `test`, `benchmark`)
- [x] Memory optimizations (`optimize` command — reversible, per-item dialog, autostart service)
- [x] Bluetooth speaker support (`scripts/connect-bt-speaker.sh`)
- [x] Dual TTS backends — Piper (multilingual) + Kokoro (English), both CUDA-accelerated
- [x] Multi-turn conversation memory (rolling window, persistent across sessions)
- [x] Text mode (`--text` flag — keyboard input, no microphone required)
- [x] Interactive config wizard (`config` command — language, voice, STT, LLM in one flow)
- [x] Language localization — system prompt and ready phrase in 12 languages
- [x] Personal config overrides (`settings.local.yaml`, gitignored)
- [ ] Auto-start as systemd service (boot without manual `./jetson-assistant start`)
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

To avoid loading GPL-licensed code into the same process as NVIDIA CUDA libraries, Kokoro TTS runs in a **separate subprocess** (`app/tts_worker.py`). The main process never imports `kokoro-onnx` directly — it communicates with the worker via JSON over stdin/stdout.

All other dependencies use permissive licenses (MIT, BSD-3, Apache-2.0). See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full list.

## Contributing

We welcome community contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, including the Developer Certificate of Origin (DCO) sign-off requirement.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Attribution

Forked from [NVIDIA-AI-IOT/reachy-mini-jetson-assistant](https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant). Original work copyright © 2023–2025 NVIDIA CORPORATION & AFFILIATES.
