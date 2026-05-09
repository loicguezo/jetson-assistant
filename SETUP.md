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

> **Important:** This project requires **Python 3.10** specifically. The Jetson ONNX Runtime GPU wheels and CTranslate2 builds target Python 3.10 on JetPack 6. Using a different Python version will cause compatibility issues.

---

## Part 1: System Setup

### NVMe Swap (Recommended for 8GB Jetson)

STT + LLM + TTS together can push close to the 8 GB limit. NVMe swap prevents OOM kills:

```bash
sudo fallocate -l 8G /mnt/nvme/swapfile   # adjust path to your NVMe mount
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

# Persist across reboots:
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  portaudio19-dev \
  libasound2-dev \
  pulseaudio-utils \
  libcudnn9-dev-cuda-12 \
  cmake \
  build-essential \
  git
```

---

## Part 2: Build llama.cpp (Native, No Docker)

llama.cpp is compiled directly on the Jetson. This avoids container overhead and keeps memory usage minimal on the shared 8 GB unified memory.

### Build

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

> `DCMAKE_CUDA_ARCHITECTURES=87` targets the Ampere GPU in the Jetson Orin Nano. The build takes ~10–15 minutes.

Verify the build:

```bash
~/llama.cpp/build/bin/llama-server --version
```

### Download a Model

Model downloads require a free [HuggingFace](https://huggingface.co) account. Create one, then log in:

```bash
hf auth login
# Paste your token from https://huggingface.co/settings/tokens (read access is enough)
```

Download the recommended model:

```bash
hf download unsloth/gemma-4-E4B-it-GGUF \
  --include "gemma-4-E4B-it-Q4_K_M.gguf" \
  --local-dir ~/models
```

**Recommended model for Orin Nano 8GB:**

| Model | File | Size |
|-------|------|------|
| Gemma 4 E4B (unsloth) | `gemma-4-E4B-it-Q4_K_M.gguf` | ~4.6 GB |

Any `.gguf` file placed in `~/models/` is detected automatically by `./jetson-assistant start`.

### Start the LLM Server

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma-3-4b-it-Q4_K_M.gguf \
  --port 8080 --host 127.0.0.1 \
  -ngl 99 \
  -c 4096
```

- `-ngl 99` — offload all layers to GPU
- `-c 4096` — context window size

Wait until you see: `llama server listening at http://127.0.0.1:8080`

**Optional: run as background service**

To start llama-server automatically at boot, create a systemd service:

```bash
sudo tee /etc/systemd/system/llama-server.service > /dev/null <<EOF
[Unit]
Description=llama.cpp server
After=network.target

[Service]
ExecStart=/home/jetson/llama.cpp/build/bin/llama-server \
  -m /home/jetson/models/gemma-3-4b-it-Q4_K_M.gguf \
  --port 8080 --host 127.0.0.1 -ngl 99 -c 4096
Restart=on-failure
User=jetson

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server
```

---

## Part 3: Python Environment

### Clone and Create Virtual Environment

```bash
git clone https://github.com/segutwein/jetson-assistant
cd jetson-assistant
python3.10 -m venv venv
source venv/bin/activate
```

### Install Python Packages

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### Install ONNX Runtime GPU (Jetson-Specific)

The default `onnxruntime` from pip is CPU-only. For GPU inference (Kokoro TTS, Silero VAD):

```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

> If `CUDAExecutionProvider` isn't listed after install, uninstall the CPU version first:
> `pip uninstall onnxruntime && pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126`

### Pin NumPy (Compatibility Fix)

```bash
pip install "numpy==1.26.4"
```

### Build CTranslate2 with CUDA (GPU-Accelerated STT)

The pip `ctranslate2` package is CPU-only. Build from source for GPU-accelerated Whisper:

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

Persist the library path:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/jetson-assistant/venv/bin/activate
```

### Verify Installation

```bash
source ~/jetson-assistant/venv/bin/activate
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

---

## Part 4: TTS Voices

Kokoro TTS downloads automatically on first run (~340 MB). To pre-download for offline use:

```bash
mkdir -p ~/jetson-assistant/voices
wget -P ~/jetson-assistant/voices/ \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P ~/jetson-assistant/voices/ \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Available voices: `af_sarah`, `af_bella`, `am_adam`, `bf_emma`, `bm_george` (configure in `settings.yaml`).

---

## Part 5: Microphone Setup

Connect your USB microphone and check it is detected:

```bash
arecord -l
```

Example output:
```
card 2: Device [USB Audio Device], device 0: USB Audio [USB Audio]
```

Set the device hint in `config/settings.yaml`:

```yaml
audio:
  input_device: "USB Audio"   # substring match against arecord -l output
```

Leave `input_device: null` to auto-detect the first available input.

---

## Troubleshooting

**llama.cpp CUDA build fails:**
Make sure CUDA is on the PATH: `export PATH=/usr/local/cuda/bin:$PATH`. Check `nvcc --version` returns 12.x.

**`CUDAExecutionProvider` not available (ONNX):**
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**CTranslate2 not finding CUDA at runtime:**
`export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH`

**LLM server not responding:**
Check it's running: `curl http://127.0.0.1:8080/v1/models`. If not, check for OOM: `dmesg | grep -i kill`.

**Out of memory (OOM):**
- Use a more aggressively quantized model (e.g. Q4_K_M instead of Q8_0)
- Make sure NVMe swap is active: `swapon --show`
- Reduce context size: `-c 2048`

**Mic not found / no audio:**
Run `arecord -l` to list devices. Set `audio.input_device` to a substring of your device name. Check `alsamixer` to ensure the capture channel is not muted.

**`arecord` / `parecord` errors:**
Kill stale processes: `pkill -9 parecord; pkill -9 arecord`
