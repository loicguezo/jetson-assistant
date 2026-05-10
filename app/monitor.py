# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""System monitor — CPU, RAM, GPU stats for Jetson."""

import glob
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class SystemStats:
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_percent: float | None = None
    gpu_temp_c: float | None = None
    gpu_freq_mhz: float | None = None

    # Legacy MB fields kept for backward compat with any callers
    @property
    def ram_used_mb(self) -> float:
        return self.ram_used_gb * 1024

    @property
    def ram_total_mb(self) -> float:
        return self.ram_total_gb * 1024


def get_system_stats() -> SystemStats:
    used, total, pct = _ram()
    return SystemStats(
        cpu_percent=_cpu(),
        ram_used_gb=used,
        ram_total_gb=total,
        ram_percent=pct,
        gpu_percent=_gpu_util(),
        gpu_temp_c=_gpu_temp(),
        gpu_freq_mhz=_gpu_freq_mhz(),
    )


def ram_used_gb() -> float:
    """Quick RAM snapshot in GB — used for memory-delta tracking."""
    used, _, _ = _ram()
    return used


def format_stats(s: SystemStats) -> str:
    """Long format for the 'stats' command."""
    parts = [
        f"CPU {s.cpu_percent:.0f}%",
        f"RAM {s.ram_used_gb:.1f}/{s.ram_total_gb:.1f}GB ({s.ram_percent:.0f}%)",
    ]
    gpu_parts = []
    if s.gpu_temp_c is not None:
        gpu_parts.append(f"{s.gpu_temp_c:.0f}°C")
    if s.gpu_freq_mhz is not None:
        gpu_parts.append(f"{s.gpu_freq_mhz:.0f}MHz")
    if s.gpu_percent is not None:
        gpu_parts.append(f"{s.gpu_percent:.0f}%")
    if gpu_parts:
        parts.append("GPU " + " ".join(gpu_parts))
    return " | ".join(parts)


def format_stats_inline(s: SystemStats) -> str:
    """Compact one-liner for the per-response timing line."""
    ram = f"RAM {s.ram_used_gb:.1f}/{s.ram_total_gb:.1f}GB"
    gpu = ""
    if s.gpu_temp_c is not None:
        gpu = f" GPU {s.gpu_temp_c:.0f}°C"
        if s.gpu_freq_mhz is not None:
            gpu += f" {s.gpu_freq_mhz:.0f}MHz"
    return ram + gpu


# ── Internal helpers ──────────────────────────────────────────────


def _cpu() -> float:
    try:
        import psutil

        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        try:
            with open("/proc/stat") as f:
                p = f.readline().split()
            idle, total = int(p[4]), sum(int(x) for x in p[1:])
            return 100.0 * (1 - idle / total) if total else 0.0
        except Exception:
            return 0.0


def _ram() -> tuple[float, float, float]:
    """Returns (used_gb, total_gb, percent)."""
    try:
        import psutil

        m = psutil.virtual_memory()
        total = m.total / 1073741824
        used = m.used / 1073741824
        return used, total, m.percent
    except ImportError:
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split()[:2]
                    info[k.rstrip(":")] = int(v)
            total = info.get("MemTotal", 0) / 1048576
            used = total - info.get("MemAvailable", 0) / 1048576
            return used, total, (used / total * 100) if total else 0.0
        except Exception:
            return 0.0, 0.0, 0.0


def _gpu_util() -> float | None:
    for path in [
        "/sys/devices/platform/gpu.0/load",
        "/sys/devices/platform/17000000.gpu/load",
        "/sys/devices/gpu.0/load",
    ]:
        try:
            return int(Path(path).read_text().strip()) / 10.0
        except Exception:
            continue
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _gpu_thermal_zone_path() -> str | None:
    """Find the sysfs path for the GPU thermal zone (cached — doesn't change)."""
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*/")):
        try:
            zone_type = Path(zone + "type").read_text().strip().lower()
            if "gpu" in zone_type:
                return zone + "temp"
        except Exception:
            continue
    return None


def _gpu_temp() -> float | None:
    path = _gpu_thermal_zone_path()
    if not path:
        return None
    try:
        return int(Path(path).read_text().strip()) / 1000.0
    except Exception:
        return None


@lru_cache(maxsize=1)
def _gpu_freq_path() -> str | None:
    """Find the GPU devfreq cur_freq path (cached)."""
    for path in glob.glob("/sys/devices/platform/*/devfreq/*/cur_freq"):
        p = path.lower()
        # Match the Orin GPU (ga10b) or anything labelled 'gpu'
        if "ga10b" in p or ("/17000000.gpu/" in p):
            return path
    # Broader fallback: any devfreq path under a bus-gpu device
    for path in glob.glob("/sys/devices/platform/bus@0/*.gpu/devfreq/*/cur_freq"):
        return path
    return None


def _gpu_freq_mhz() -> float | None:
    path = _gpu_freq_path()
    if not path:
        return None
    try:
        return int(Path(path).read_text().strip()) / 1e6
    except Exception:
        return None


def get_power_mode() -> str | None:
    """Return the current NVPModel power mode name, e.g. '25W'."""
    try:
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=3)
        for line in r.stdout.splitlines():
            if "Power Mode" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def get_jetson_model() -> str:
    try:
        with open("/proc/device-tree/model") as f:
            raw = f.read().strip().rstrip("\x00")
        return raw.replace("NVIDIA ", "").replace(" Engineering Reference Developer Kit", "")
    except Exception:
        return "Jetson"
