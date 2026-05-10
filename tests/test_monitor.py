"""
Monitor unit tests — no server required, just reads sysfs/proc.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.monitor import (
    format_stats,
    format_stats_inline,
    get_power_mode,
    get_system_stats,
    ram_used_gb,
)


def test_ram_values_are_sane():
    s = get_system_stats()
    assert s.ram_total_gb > 0, "total RAM should be > 0"
    assert 0 < s.ram_used_gb <= s.ram_total_gb, "used RAM out of range"
    assert 0 <= s.ram_percent <= 100


def test_gpu_temp_is_plausible():
    s = get_system_stats()
    if s.gpu_temp_c is None:
        return  # not a Jetson or sysfs unavailable — skip silently
    assert 10 < s.gpu_temp_c < 110, f"GPU temp {s.gpu_temp_c}°C looks wrong"


def test_gpu_freq_is_plausible():
    s = get_system_stats()
    if s.gpu_freq_mhz is None:
        return
    assert 50 < s.gpu_freq_mhz < 2000, f"GPU freq {s.gpu_freq_mhz}MHz looks wrong"


def test_format_stats_inline_contains_ram():
    s = get_system_stats()
    line = format_stats_inline(s)
    assert "RAM" in line
    assert "GB" in line


def test_format_stats_full_contains_cpu_and_ram():
    s = get_system_stats()
    line = format_stats(s)
    assert "CPU" in line
    assert "RAM" in line


def test_power_mode_is_string_or_none():
    mode = get_power_mode()
    assert mode is None or isinstance(mode, str)
    if mode:
        assert len(mode) > 0


def test_ram_snapshot_matches_stats():
    snap = ram_used_gb()
    s = get_system_stats()
    # Allow 200 MB drift between two consecutive reads
    assert abs(snap - s.ram_used_gb) < 0.2, "ram_used_gb() diverges from SystemStats"
