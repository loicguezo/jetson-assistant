# AGENTS.md

Guidelines for AI agents working on this repository.

## Project

A low-latency, fully on-device voice assistant for NVIDIA Jetson Orin Nano 8GB.
Pipeline: mic → Silero VAD → faster-whisper STT → llama.cpp LLM → Kokoro TTS → speaker.
Everything runs locally — no cloud, no Docker, no internet required at runtime.

## Key constraints

- **8 GB unified memory** (CPU + GPU share the same pool). Every MB counts.
  Never add dependencies that pull in large runtimes (no PyTorch, no Ollama, no Docker).
- **llama.cpp is compiled natively** on the device (`~/llama.cpp/build/bin/llama-server`).
  Do not suggest Docker or Ollama as alternatives.
- **GPL isolation**: `app/tts_worker.py` runs in a subprocess specifically to keep
  GPL-licensed code (phonemizer, espeak-ng) out of the main CUDA process. Do not
  import `kokoro` or `phonemizer` from any file other than `tts_worker.py`.
- **Target platform**: JetPack 6.x, Python 3.10, CUDA 12.6, CUDA arch sm_87.

## Commands

```bash
./jetson-assistant setup            # first-time setup wizard
./jetson-assistant start            # pick model → llama-server → voice chat
./jetson-assistant stop
./jetson-assistant status
./jetson-assistant optimize         # memory optimizations (reversible)
./jetson-assistant optimize --restore
./jetson-assistant optimize --status
```

Running the assistant requires `llama-server` to be running and a `.gguf` model
in `~/models/`. The setup wizard handles both.

## File map

| File | Purpose |
|------|---------|
| `manage.py` | Typer CLI — all `./jetson-assistant` commands |
| `run_voice_chat.py` | Voice pipeline entry point |
| `app/pipeline.py` | Audio I/O, VAD loop, TTS streaming |
| `app/llm.py` | OpenAI-compatible LLM client (streams to llama-server) |
| `app/stt.py` | faster-whisper wrapper |
| `app/tts.py` | TTS client — spawns `tts_worker.py` as subprocess |
| `app/tts_worker.py` | TTS subprocess (GPL-isolated) |
| `app/manager.py` | llama-server lifecycle, PID files, GGUF discovery |
| `app/optimize.py` | System optimizations with state persistence |
| `app/setup_wizard.py` | First-time setup logic (build, download, venv) |
| `app/config.py` | Config dataclasses + YAML loader |
| `app/monitor.py` | CPU/GPU/RAM stats |
| `config/settings.yaml` | All runtime configuration |

## Development

- No SPDX headers on new files. Headers are kept only on files derived from the
  NVIDIA upstream (`SPDX-License-Identifier: Apache-2.0`).
- State files live in `~/.jetson-assistant/` (PID files, optimization state, logs).
- Model files live in `~/models/` (gitignored). TTS voice files live in `voices/`.
- HuggingFace login is required to download models: `hf auth login`.
- **Comments**: keep them short and to the point — explain *why*, not *what*.
