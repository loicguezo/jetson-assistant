#!/usr/bin/env bash
# Connect a Bluetooth speaker and set it as the default PulseAudio sink.
#
# Usage:
#   ./scripts/connect-bt-speaker.sh                  # prompts for MAC
#   ./scripts/connect-bt-speaker.sh 08:EB:ED:FF:87:78

set -euo pipefail

MAC="${1:-}"

# ── Resolve MAC ────────────────────────────────────────────────────
if [ -z "$MAC" ]; then
    echo "Paired devices:"
    bluetoothctl devices Paired 2>/dev/null || bluetoothctl devices
    echo ""
    read -rp "Enter MAC address: " MAC
fi

if ! [[ "$MAC" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
    echo "Error: invalid MAC address: $MAC" >&2
    exit 1
fi

echo "[1/4] Ensuring pulseaudio-module-bluetooth is installed..."
if ! dpkg -s pulseaudio-module-bluetooth &>/dev/null; then
    sudo apt-get install -y pulseaudio-module-bluetooth
fi

echo "[2/4] Restarting PulseAudio (ensures Bluetooth module loads after bluetoothd)..."
killall -9 pulseaudio 2>/dev/null || true
sleep 1
pulseaudio --start
sleep 1

echo "[3/4] Connecting to $MAC..."
bluetoothctl connect "$MAC"
sleep 2

echo "[4/4] Setting as default audio sink..."
SINK=$(pactl list sinks short | grep -i "bluez\|$(echo "$MAC" | tr ':' '_')" | awk '{print $2}' | head -1)

if [ -z "$SINK" ]; then
    echo "⚠ Bluetooth sink not found yet — try running again in a few seconds."
    echo "  Available sinks:"
    pactl list sinks short
    exit 1
fi

pactl set-default-sink "$SINK"
echo "✓ Default sink set to: $SINK"
