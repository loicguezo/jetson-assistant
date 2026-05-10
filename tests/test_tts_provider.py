"""
TTS provider tests — verify CUDA is used when onnxruntime-gpu is installed.
No TTS worker spawn needed; we query ORT directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cuda_provider_available():
    """CUDAExecutionProvider must be present after installing onnxruntime-gpu."""
    try:
        import onnxruntime as ort
    except ImportError:
        import pytest
        pytest.skip("onnxruntime not installed")

    providers = ort.get_available_providers()
    assert "CUDAExecutionProvider" in providers, (
        f"CUDAExecutionProvider not available — got: {providers}\n"
        "Fix: pip uninstall -y onnxruntime onnxruntime-gpu && "
        "pip install --force-reinstall numpy==1.26.4 onnxruntime-gpu==1.23.0 "
        "--extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126"
    )


def test_cpu_provider_always_present():
    try:
        import onnxruntime as ort
    except ImportError:
        import pytest
        pytest.skip("onnxruntime not installed")

    assert "CPUExecutionProvider" in ort.get_available_providers()


def test_tts_worker_uses_cuda(tmp_path):
    """Spawn the TTS worker and verify it reports CUDAExecutionProvider."""
    import json
    import subprocess
    import select
    import pytest

    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDAExecutionProvider not available on this system")
    except ImportError:
        pytest.skip("onnxruntime not installed")

    voices_dir = Path(__file__).parent.parent / "voices"
    model_path = voices_dir / "kokoro-v1.0.onnx"
    if not model_path.exists():
        pytest.skip("Kokoro model not downloaded — run the assistant once to fetch it")

    worker = Path(__file__).parent.parent / "app" / "tts_worker.py"
    proc = subprocess.Popen(
        [sys.executable, str(worker), "--model-dir", str(voices_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        ready = select.select([proc.stdout], [], [], 30.0)[0]
        assert ready, "TTS worker startup timed out"
        line = proc.stdout.readline()
        resp = json.loads(line)
        assert resp.get("status") == "ready", f"Worker not ready: {resp}"
        assert resp.get("provider") == "CUDAExecutionProvider", (
            f"Expected CUDAExecutionProvider, got: {resp.get('provider')}"
        )
    finally:
        proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=5)
