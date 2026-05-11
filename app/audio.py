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

"""Audio utilities — PulseAudio management and ALSA device discovery."""

import subprocess
import time


def kill_pulseaudio() -> bool:
    """Kill PulseAudio so ALSA can claim an exclusive device.

    Does NOT write autospawn=no — PulseAudio will restart on next login/session,
    which is the right behaviour (e.g. for Bluetooth audio on next boot).
    """
    subprocess.run(["pulseaudio", "--kill"], capture_output=True)
    subprocess.run(["pkill", "-9", "pulseaudio"], capture_output=True)
    time.sleep(0.5)
    return subprocess.run(["pgrep", "-x", "pulseaudio"], capture_output=True).returncode != 0


def unmute_alsa_capture(card: int) -> None:
    """Set capture volume to maximum and enable the capture switch for a given card.

    USB mics default to 0 capture volume after boot on some systems (e.g. Jetson).
    This is best-effort: silently ignored if the card has no such controls.
    """
    import re

    try:
        r = subprocess.run(
            ["amixer", "-c", str(card), "contents"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Parse multi-line blocks: each control spans several lines.
        # We collect lines per control and extract numid + max from them.
        lines = r.stdout.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            numid_m = re.search(r"numid=(\d+)", line)
            if numid_m and ("Capture Volume" in line or "Capture Switch" in line):
                numid = numid_m.group(1)
                # Collect the next few lines of this block to find max= or type
                block = line
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].startswith("numid="):
                        break
                    block += "\n" + lines[j]

                if "Capture Switch" in line:
                    subprocess.run(
                        ["amixer", "-c", str(card), "cset", f"numid={numid}", "on"],
                        capture_output=True,
                        timeout=5,
                    )
                elif "Capture Volume" in line:
                    max_m = re.search(r"max=(\d+)", block)
                    if max_m:
                        subprocess.run(
                            ["amixer", "-c", str(card), "cset", f"numid={numid}", max_m.group(1)],
                            capture_output=True,
                            timeout=5,
                        )
            i += 1
    except Exception:
        pass


def find_alsa_device(
    name_hint: str = "USB Audio",
    direction: str = "input",
) -> tuple[int, int, str] | None:
    """Find ALSA device by name substring. Returns (card, device, name) or None."""
    cmd = "arecord" if direction == "input" else "aplay"
    try:
        r = subprocess.run([cmd, "-l"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    needle = name_hint.lower()
    for line in r.stdout.splitlines():
        if "card" in line.lower() and needle in line.lower():
            try:
                parts = line.split()
                card = int(parts[parts.index("card") + 1].rstrip(":,"))
                dev = int(parts[parts.index("device") + 1].rstrip(":,"))
                name = line.split("[")[1].split("]")[0] if "[" in line else name_hint
                return card, dev, name
            except (ValueError, IndexError):
                continue
    return None
