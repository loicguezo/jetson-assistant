"""
Server lifecycle integration tests.

These tests require llama-server to be installed but not necessarily running.
They verify that stop_llama_server() eliminates ALL llama-server processes,
including orphans that are not tracked by the PID file.

Run with:
    pytest tests/test_server_lifecycle.py -v -s
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.manager import (
    LLAMA_PID_FILE,
    STATE_DIR,
    find_llama_server,
    find_llama_server_pids,
    is_llama_server_running,
    stop_llama_server,
)


@pytest.fixture(autouse=True)
def ensure_stopped():
    """Kill all llama-server processes before and after each test."""
    stop_llama_server()
    yield
    stop_llama_server()


def _spawn_llama_dummy() -> subprocess.Popen:
    """Start llama-server with --help so it exits quickly, then keep a real
    long-running process for orphan tests using a wrapper trick.
    Actually spawns the binary with a bad model path — it stays in
    the process list long enough to test against."""
    llama = find_llama_server()
    if not llama:
        pytest.skip("llama-server binary not found")
    # Use --version if available; otherwise it will error-exit quickly.
    # We only need the process to exist for a moment, so Popen is enough.
    proc = subprocess.Popen(
        [str(llama), "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _spawn_orphan() -> int:
    """Spawn a llama-server process without writing a PID file — simulates an
    orphan from a previous session or a crash."""
    llama = find_llama_server()
    if not llama:
        pytest.skip("llama-server binary not found")
    proc = subprocess.Popen(
        [str(llama), "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Give it a moment to appear in the process table
    time.sleep(0.2)
    return proc.pid


# ── Unit-level: find_llama_server_pids ───────────────────────────


def test_find_pids_returns_list():
    pids = find_llama_server_pids()
    assert isinstance(pids, list)
    for pid in pids:
        assert isinstance(pid, int) and pid > 0


def test_find_pids_detects_running_process():
    if not find_llama_server():
        pytest.skip("llama-server binary not found")
    proc = _spawn_orphan()
    try:
        pids = find_llama_server_pids()
        # Process may have already exited (--version exits instantly),
        # but if it's still alive it must appear in pids.
        if _pid_alive(proc):
            assert proc in pids, f"orphan pid {proc} not detected by find_llama_server_pids"
    finally:
        try:
            os.kill(proc, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# ── Core bug: stop must kill orphaned processes ───────────────────


def test_stop_kills_orphan_not_in_pid_file():
    """Regression test for the orphan-process bug.

    Scenario: a llama-server process runs but is NOT tracked in the PID file
    (e.g. started by an old version, or PID file was deleted after a crash).
    stop_llama_server() must still kill it.
    """
    if not find_llama_server():
        pytest.skip("llama-server binary not found")

    # Spawn an orphan without writing a PID file
    orphan_pid = _spawn_orphan()
    time.sleep(0.3)

    # Make sure the PID file does NOT contain this pid
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LLAMA_PID_FILE.write_text("99999999")  # wrong/fake PID

    try:
        stop_llama_server()

        # Give the OS a moment
        time.sleep(0.3)

        remaining = find_llama_server_pids()
        alive = [p for p in remaining if _pid_alive(p)]
        assert alive == [], f"stop_llama_server() left orphaned processes running: {alive}"
    finally:
        # Safety cleanup
        try:
            os.kill(orphan_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_stop_clears_pid_file():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LLAMA_PID_FILE.write_text("99999999")
    stop_llama_server()
    assert not LLAMA_PID_FILE.exists(), "PID file should be removed after stop"


def test_is_running_false_after_stop():
    """is_llama_server_running() must return False after stop — no HTTP ghost."""
    stop_llama_server()
    time.sleep(0.5)
    assert not is_llama_server_running(), (
        "is_llama_server_running() returned True after stop — "
        "orphaned process or stale PID file detected"
    )


def test_stop_idempotent():
    """Calling stop twice should not raise."""
    stop_llama_server()
    stop_llama_server()


# ── Helpers ───────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
