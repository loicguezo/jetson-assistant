"""Process management — llama-server lifecycle and PID tracking."""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import httpx

STATE_DIR = Path.home() / ".jetson-assistant"
LLAMA_PID_FILE = STATE_DIR / "llama-server.pid"
LLAMA_PORT_FILE = STATE_DIR / "llama-server.port"
LLAMA_LOG_FILE = STATE_DIR / "llama-server.log"

_LLAMA_SEARCH_PATHS = [
    Path.home() / "llama.cpp/build/bin/llama-server",
    Path("/usr/local/bin/llama-server"),
    Path("/usr/bin/llama-server"),
]


def find_llama_server() -> Path | None:
    for p in _LLAMA_SEARCH_PATHS:
        if p.exists() and os.access(p, os.X_OK):
            return p
    found = shutil.which("llama-server")
    return Path(found) if found else None


def find_gguf_models() -> list[Path]:
    search_dirs = [
        Path.home() / "models",
        Path.home() / ".cache/huggingface",
        Path.cwd(),
    ]
    models = []
    for d in search_dirs:
        if d.exists():
            models.extend(sorted(d.rglob("*.gguf")))
    seen = set()
    return [m for m in models if not (m in seen or seen.add(m))]


def _llama_port() -> int:
    """Return the port llama-server was last started on (default 8080)."""
    if LLAMA_PORT_FILE.exists():
        try:
            return int(LLAMA_PORT_FILE.read_text().strip())
        except Exception:
            pass
    return 8080


def get_llama_server_port() -> int:
    """Public accessor for the llama-server port (used by tests)."""
    return _llama_port()


def _save_pid(pid_file: Path, pid: int):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but is owned by another user


def is_llama_server_running() -> bool:
    pid = read_pid(LLAMA_PID_FILE)
    if pid and is_process_running(pid):
        return True
    # also check via HTTP in case PID file is stale
    try:
        r = httpx.get(f"http://127.0.0.1:{_llama_port()}/v1/models", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def wait_for_llama_server(timeout: int = 120) -> bool:
    port = _llama_port()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_llama_server(model_path: Path, port: int = 8080, ctx: int = 8192) -> int | None:
    llama = find_llama_server()
    if not llama:
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(llama),
        "-m",
        str(model_path),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-ngl",
        "99",
        "-c",
        str(ctx),
        "-np",
        "1",  # single slot = full KV-cache reuse across turns
        "--reasoning",
        "off",  # disable thinking tokens (saves 15-28s TTFT on thinking models)
    ]
    try:
        with open(LLAMA_LOG_FILE, "w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
    except Exception as e:
        print(f"llama-server spawn failed: {e}")
        return None
    _save_pid(LLAMA_PID_FILE, proc.pid)
    LLAMA_PORT_FILE.write_text(str(port))
    return proc.pid


def _kill_pid(pid: int, timeout_steps: int = 20) -> None:
    """SIGTERM a PID, escalate to SIGKILL if it doesn't die."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(timeout_steps):
        if not is_process_running(pid):
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def find_llama_server_pids() -> list[int]:
    """Return PIDs of all running llama-server processes (tracked or orphaned)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    except Exception:
        return []


def stop_llama_server():
    tracked_pid = read_pid(LLAMA_PID_FILE)

    # Kill tracked PID first
    if tracked_pid and is_process_running(tracked_pid):
        _kill_pid(tracked_pid)

    # Kill any orphaned llama-server processes not in the PID file
    for pid in find_llama_server_pids():
        if pid != tracked_pid and is_process_running(pid):
            _kill_pid(pid)

    if LLAMA_PID_FILE.exists():
        LLAMA_PID_FILE.unlink()
    if LLAMA_PORT_FILE.exists():
        LLAMA_PORT_FILE.unlink()


def get_llama_model_name() -> str | None:
    try:
        r = httpx.get(f"http://127.0.0.1:{_llama_port()}/v1/models", timeout=2.0)
        models = r.json().get("data", [])
        return models[0].get("id") if models else None
    except Exception:
        return None
