# Setup Guide

Full installation instructions for the Jetson Voice Assistant.

## Prerequisites

### Hardware

- **NVIDIA Jetson Orin Nano** (8GB) — other Jetson modules may work but are untested
- **USB microphone** — any USB audio device works; check `arecord -l` after connecting
- **Speaker** — USB or analog (3.5mm) output
- **NVMe SSD** recommended — for swap space and model storage

### Software

- **JetPack 6.x** (L4T r36.x, Ubuntu 22.04, CUDA 12.6)
- **Python 3.10** (ships with JetPack 6 Ubuntu 22.04)
- **PulseAudio** (for mic/speaker multiplexing)

> **Important:** This project requires **Python 3.10** specifically. The Jetson ONNX Runtime GPU wheels and CTranslate2 builds are built against Python 3.10 on JetPack 6. Using a different Python version will cause compatibility issues.

## Hardware Setup

### NVMe Swap (Recommended for 8GB Jetson)

Running STT + LLM + TTS simultaneously can be memory-intensive. Setting up swap on NVMe prevents OOM kills:

```bash
sudo fallocate -l 8G /mnt/nvme/swapfile   # adjust path to your NVMe mount
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

# Persist across reboots:
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Installation

### Step 1: System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  portaudio19-dev \
  libasound2-dev \
  pulseaudio-utils \
  libcudnn9-dev-cuda-12
```

### Step 2: Clone and Create Virtual Environment

```bash
git clone https://github.com/segutwein/jetson-assistant
cd jetson-assistant
python3.10 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Packages

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### Step 4: Install ONNX Runtime GPU (Jetson-Specific)

The default `onnxruntime` from pip is CPU-only. For GPU inference (Kokoro TTS, Silero VAD) on Jetson:

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> If `CUDAExecutionProvider` isn't listed after install, uninstall the CPU version first:
> `pip uninstall onnxruntime && pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126`

### Step 5: Pin NumPy (Compatibility Fix)

The Jetson `onnxruntime-gpu` wheel requires NumPy 1.x:

```bash
pip install "numpy==1.26.4"
```

### Step 6: Build CTranslate2 with CUDA (GPU-Accelerated STT)

The pip `ctranslate2` package is CPU-only. For GPU-accelerated speech-to-text on Jetson, build from source:

```bash
pip install pybind11

cd ~
git clone --depth 1 https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
git submodule update --init --recursive

mkdir build && cd build
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH_LIST="8.7" -DOPENMP_RUNTIME=NONE -DWITH_MKL=OFF

make -j$(nproc)
cmake --install . --prefix ~/.local

export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH
cd ../python
pip install .
```

Persist the library path in your venv activation script:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/jetson-assistant/venv/bin/activate
```

### Verify Installation

```bash
source venv/bin/activate
python3 -c "
import ctranslate2; print('CTranslate2 CUDA devices:', ctranslate2.get_cuda_device_count())
import onnxruntime; print('ONNX providers:', onnxruntime.get_available_providers())
import faster_whisper; print('faster-whisper: OK')
import kokoro_onnx; print('kokoro-onnx: OK')
"
```

Expected output:
```
CTranslate2 CUDA devices: 1
ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
faster-whisper: OK
kokoro-onnx: OK
```

## LLM Setup

The assistant connects to a locally running LLM server. Two backends are supported:

### Option A: llama.cpp (Docker)

```bash
docker run --rm --runtime nvidia --gpus all \
  --name assistant-llm \
  -p 8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf ggml-org/gemma-3-1b-it-GGUF:Q8_0 \
  --port 8080 --host 0.0.0.0 -ngl 99
```

Wait until you see `llama server listening at http://0.0.0.0:8080`.

In `config/settings.yaml`:
```yaml
llm:
  backend: "openai"
  base_url: "http://localhost:8080"
```

### Option B: Ollama

```bash
ollama serve
ollama pull gemma3:1b
```

In `config/settings.yaml`:
```yaml
llm:
  backend: "ollama"
  base_url: "http://localhost:11434"
  model: "gemma3:1b"
```

## TTS Voices

Kokoro TTS downloads automatically on first run (~340 MB). No manual step needed.

To pre-download for offline use:

```bash
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P voices/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Available voices: `af_sarah`, `af_bella`, `am_adam`, `bf_emma`, `bm_george` (configure in `settings.yaml`).

## Microphone Setup

Connect your USB microphone and check it is detected:

```bash
arecord -l
```

Example output:
```
card 2: Device [USB Audio Device], device 0: USB Audio [USB Audio]
```

Set the device hint in `config/settings.yaml` to a substring of your device name:

```yaml
audio:
  input_device: "USB Audio"   # matches any device containing this string
```

Leave `input_device: null` to auto-detect the first available input device.

## Troubleshooting

**`CUDAExecutionProvider` not available:**
Uninstall CPU onnxruntime and reinstall the GPU version:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**CTranslate2 not finding CUDA:**
Make sure the library path is set: `export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH`

**LLM server not responding:**
Check the Docker container is running: `docker ps`. View logs: `docker logs assistant-llm`

**Mic not found / no audio:**
Run `arecord -l` to list devices. Set `audio.input_device` in `settings.yaml` to a substring of your device name. If the mic is silent, check `alsamixer` and ensure the capture channel is unmuted.

**`arecord` / `parecord` errors:**
Try killing stale processes: `pkill -9 parecord; pkill -9 arecord`
