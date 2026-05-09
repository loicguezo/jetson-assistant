"""First-time setup wizard — build llama.cpp and download a model."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

LLAMA_DIR = Path.home() / "llama.cpp"
MODELS_DIR = Path.home() / "models"

RECOMMENDED_MODELS = [
    {
        "name": "Gemma 4 E4B Q4_K_M",
        "repo": "unsloth/gemma-4-E4B-it-GGUF",
        "filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "size": "~4.6 GB",
        "description": "Recommended — Google Gemma 4, modern and fast",
        "license_url": None,
    },
]


# ── Prerequisite checks ────────────────────────────────────────────

def check_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def check_cuda() -> Optional[str]:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    # JetPack may install nvcc in versioned or unversioned paths
    candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None



# required: build fails without these
# optional: warning only, build may still succeed
_REQUIRED = ["git", "cmake", "make"]
_OPTIONAL = ["nvcc"]  # cmake can sometimes find CUDA without nvcc in PATH


def check_venv_module() -> Optional[str]:
    """Check that python3.10 -m venv works (requires python3.10-venv on Debian/Ubuntu)."""
    r = subprocess.run(
        ["python3.10", "-m", "venv", "--help"],
        capture_output=True,
    )
    return "python3.10" if r.returncode == 0 else None


def check_portaudio() -> Optional[str]:
    """Check that PortAudio is installed (needed by sounddevice for mic/speaker)."""
    import ctypes.util
    lib = ctypes.util.find_library("portaudio")
    return lib if lib else None


def check_prerequisites() -> dict:
    results = {
        "git":          (check_tool("git"),   True),
        "cmake":        (check_tool("cmake"), True),
        "make":         (check_tool("make"),  True),
        "nvcc":         (check_cuda(),        False),  # optional
        "pip3":         (check_tool("pip3") or check_tool("pip"), True),
        "python3-venv": (check_venv_module(), True),
        "portaudio":    (check_portaudio(),   True),
    }
    return results


# ── llama.cpp ──────────────────────────────────────────────────────

def llama_server_path() -> Optional[Path]:
    p = LLAMA_DIR / "build/bin/llama-server"
    return p if p.exists() else None


def clone_llama_cpp() -> bool:
    if LLAMA_DIR.exists():
        return True
    rc = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ggml-org/llama.cpp", str(LLAMA_DIR)],
    ).returncode
    return rc == 0


def build_llama_cpp() -> bool:
    env_path = "/usr/local/cuda/bin"
    import os
    env = os.environ.copy()
    env["PATH"] = env_path + ":" + env.get("PATH", "")
    env["CUDA_HOME"] = "/usr/local/cuda"

    cmake_configure = subprocess.run(
        [
            "cmake", "-B", "build",
            "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=87",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=LLAMA_DIR,
        env=env,
    )
    if cmake_configure.returncode != 0:
        return False

    nproc = subprocess.run(
        ["nproc"], capture_output=True, text=True
    ).stdout.strip() or "4"

    cmake_build = subprocess.run(
        ["cmake", "--build", "build", "--config", "Release", "-j", nproc],
        cwd=LLAMA_DIR,
        env=env,
    )
    return cmake_build.returncode == 0


# ── Model download ─────────────────────────────────────────────────

def _ensure_huggingface_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"],
        ).returncode
        return rc == 0


def _hf_cmd() -> Optional[str]:
    """Return the HuggingFace CLI binary: `hf` (new) or `huggingface-cli` (old)."""
    return shutil.which("hf") or shutil.which("huggingface-cli")


def check_hf_login() -> bool:
    """Return True if a HuggingFace token is saved locally."""
    cmd = _hf_cmd()
    if cmd:
        name = Path(cmd).name
        check_args = [cmd, "auth", "token"] if name == "hf" else [cmd, "whoami"]
        r = subprocess.run(check_args, capture_output=True, text=True)
        return r.returncode == 0 and "Not logged in" not in r.stdout
    # Python fallback
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token() is not None
    except Exception:
        return False


def hf_login() -> bool:
    """Run HuggingFace login interactively. Return True on success."""
    cmd = _hf_cmd()
    if cmd:
        name = Path(cmd).name
        login_args = [cmd, "auth", "login"] if name == "hf" else [cmd, "login"]
        return subprocess.run(login_args).returncode == 0
    # Python fallback
    return subprocess.run(
        [sys.executable, "-c", "from huggingface_hub import login; login()"]
    ).returncode == 0


class DownloadAuthError(Exception):
    """Raised when a model download fails due to missing authentication (HTTP 401/403)."""


def download_model(repo: str, filename: str) -> Optional[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename
    if dest.exists():
        return dest

    cmd = _hf_cmd()
    if cmd:
        name = Path(cmd).name
        if name == "hf":
            args = [cmd, "download", repo, "--include", filename, "--local-dir", str(MODELS_DIR)]
        else:
            args = [cmd, "download", repo, filename, "--local-dir", str(MODELS_DIR)]
        # Let stdout/stderr through so the hf CLI progress bar is visible.
        # Capture only stderr separately to detect auth errors on failure.
        r = subprocess.run(args, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0 and dest.exists():
            return dest
        err = r.stderr or ""
        if "401" in err or "403" in err or "not logged in" in err.lower() or "token" in err.lower():
            raise DownloadAuthError(repo)
        if err.strip():
            print(err.strip())
        return None

    # Python fallback via huggingface_hub
    if not _ensure_huggingface_hub():
        return None
    from huggingface_hub import hf_hub_download, HfFolder
    token = HfFolder.get_token()
    try:
        path = hf_hub_download(
            repo_id=repo, filename=filename,
            local_dir=str(MODELS_DIR), token=token,
        )
        return Path(path)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            raise DownloadAuthError(repo) from e
        print(f"Download failed: {e}")
        return None


# ── CTranslate2 CUDA build ────────────────────────────────────────

CT2_DIR = Path.home() / "CTranslate2"
CT2_CMAKE_FLAGS = [
    "-DWITH_CUDA=ON",
    "-DWITH_CUDNN=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=87",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda",
    "-DOPENMP_RUNTIME=COMP",
    "-DWITH_MKL=OFF",
    "-DCMAKE_INSTALL_PREFIX=/usr/local",
]
CT2_VERSION = "v4.7.1"

# Pre-built CUDA wheel for Jetson (JetPack 6, CUDA 12.6)
CT2_JETSON_INDEX = "https://pypi.jetson-ai-lab.dev/jp6/cu126"


def ctranslate2_has_cuda(venv_dir: Optional[Path] = None) -> bool:
    """Return True if the installed CTranslate2 was built with CUDA support.

    Runs a subprocess so the check is never confused by module-import caching
    or a missing LD_LIBRARY_PATH in the parent process.
    """
    import os
    python = str(venv_dir / "bin/python3") if venv_dir else sys.executable

    # Build LD_LIBRARY_PATH: include the ctranslate2 package dir (where the
    # .so is bundled when installed from source) and ~/.local/lib.
    ct2_pkg = Path(python).parent.parent / "lib/python3.10/site-packages/ctranslate2"
    ld_paths = []
    if ct2_pkg.exists():
        ld_paths.append(str(ct2_pkg))
    local_lib = Path.home() / ".local/lib"
    if local_lib.exists():
        ld_paths.append(str(local_lib))
    env = os.environ.copy()
    if ld_paths:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(ld_paths + ([existing] if existing else []))

    r = subprocess.run(
        [python, "-c",
         "import ctranslate2; print(ctranslate2.get_cuda_device_count())"],
        capture_output=True, text=True, env=env,
    )
    try:
        return int(r.stdout.strip()) > 0
    except ValueError:
        return False


def install_ctranslate2_cuda(venv_dir: Path) -> bool:
    """Install CTranslate2 with CUDA support.

    Strategy:
      1. Try the pre-built wheel from jetson-ai-lab.dev (fast, no build required).
      2. Fall back to building from source (~20 min) and installing to /usr/local.

    Returns True if CTranslate2 ends up with CUDA support after this call.
    """
    pip = venv_dir / "bin/pip"

    # ── Attempt 1: pre-built wheel ─────────────────────────────
    print("  Trying pre-built wheel from jetson-ai-lab.dev...", flush=True)
    rc = subprocess.run([
        str(pip), "install", "ctranslate2",
        "--extra-index-url", CT2_JETSON_INDEX,
    ]).returncode
    if rc == 0 and ctranslate2_has_cuda(venv_dir):
        return True

    # ── Attempt 2: build from source ──────────────────────────
    print("  Pre-built wheel unavailable — falling back to source build.", flush=True)
    return _build_ctranslate2_from_source(venv_dir)


def clone_ctranslate2() -> bool:
    if CT2_DIR.exists():
        return True
    rc = subprocess.run([
        "git", "clone", "--recursive", "--depth", "1",
        "--branch", CT2_VERSION,
        "https://github.com/OpenNMT/CTranslate2.git",
        str(CT2_DIR),
    ]).returncode
    return rc == 0


def _build_ctranslate2_from_source(venv_dir: Path) -> bool:
    """Build CTranslate2 v4.7.1 from source with CUDA, install to /usr/local."""
    import os
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")
    env["CUDA_HOME"] = "/usr/local/cuda"

    if not clone_ctranslate2():
        return False

    build_dir = CT2_DIR / "build"
    build_dir.mkdir(exist_ok=True)

    # Configure
    rc = subprocess.run(
        ["cmake", ".."] + CT2_CMAKE_FLAGS,
        cwd=build_dir, env=env,
    ).returncode
    if rc != 0:
        return False

    # Build
    nproc = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip() or "4"
    rc = subprocess.run(
        ["cmake", "--build", ".", "--config", "Release", "-j", nproc],
        cwd=build_dir, env=env,
    ).returncode
    if rc != 0:
        return False

    # Install C++ library to /usr/local so the dynamic linker can find it
    rc = subprocess.run(
        ["sudo", "cmake", "--install", "."],
        cwd=build_dir, env=env,
    ).returncode
    if rc != 0:
        return False

    # Refresh the linker cache so libctranslate2.so.4 is found at runtime
    subprocess.run(["sudo", "ldconfig"])

    # Install Python bindings into the venv (bundles libctranslate2.so into the package)
    pip = venv_dir / "bin/pip"
    rc = subprocess.run(
        [str(pip), "install", "--force-reinstall", str(CT2_DIR / "python")],
        env=env,
    ).returncode
    return rc == 0 and ctranslate2_has_cuda(venv_dir)


# ── Whisper model download ─────────────────────────────────────────

def whisper_model_cached(model_name: str) -> bool:
    """Return True if the faster-whisper model is already in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        repo_id = f"Systran/faster-whisper-{model_name}"
        # try_to_load_from_cache returns a path string when found,
        # None when the repo is unknown, or the _CACHED_NO_EXIST sentinel
        # (not a string) when the file was negatively cached. isinstance(str)
        # is the correct check per the huggingface_hub docs.
        result = try_to_load_from_cache(repo_id, "config.json")
        return isinstance(result, str)
    except Exception:
        return False


def download_whisper_model(model_name: str) -> bool:
    """Pre-download the faster-whisper model into the HF cache."""
    if not _ensure_huggingface_hub():
        return False
    try:
        from huggingface_hub import snapshot_download
        repo_id = f"Systran/faster-whisper-{model_name}"
        snapshot_download(repo_id=repo_id)
        return True
    except Exception as e:
        print(f"Whisper model download failed: {e}")
        return False


# ── venv ───────────────────────────────────────────────────────────

def setup_venv(project_dir: Path) -> bool:
    venv_dir = project_dir / "venv"
    if venv_dir.exists():
        return True
    rc = subprocess.run(
        ["python3.10", "-m", "venv", str(venv_dir)]
    ).returncode
    if rc != 0:
        return False
    pip = venv_dir / "bin/pip"
    rc = subprocess.run(
        [str(pip), "install", "--upgrade", "pip", "wheel"]
    ).returncode
    if rc != 0:
        return False
    req = project_dir / "requirements.txt"
    rc = subprocess.run(
        [str(pip), "install", "-r", str(req)]
    ).returncode
    return rc == 0
