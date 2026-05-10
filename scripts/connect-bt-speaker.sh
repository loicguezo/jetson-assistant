#!/usr/bin/env bash
# Connect a Bluetooth speaker, set it as the default PulseAudio sink,
# and install a systemd service so it reconnects automatically after
# reboots or when the speaker is powered back on.
#
# Usage:
#   ./scripts/connect-bt-speaker.sh                  # prompts for MAC
#   ./scripts/connect-bt-speaker.sh 08:EB:ED:FF:87:78

set -euo pipefail

MAC="${1:-}"
CONFIG_DIR="$HOME/.config/jetson-assistant"
MAC_FILE="$CONFIG_DIR/bt-speaker.mac"
SERVICE_NAME="bt-speaker-reconnect"
SERVICE_FILE="$HOME/.config/systemd/user/${SERVICE_NAME}.service"

# ── Resolve MAC ────────────────────────────────────────────────────
if [ -z "$MAC" ] && [ -f "$MAC_FILE" ]; then
    MAC=$(cat "$MAC_FILE")
    echo "Using saved MAC: $MAC"
fi

if [ -z "$MAC" ]; then
    echo "Paired devices:"
    bluetoothctl devices 2>/dev/null
    echo ""
    read -rp "Enter MAC address: " MAC
fi

if ! [[ "$MAC" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
    echo "Error: invalid MAC address: $MAC" >&2
    exit 1
fi

# ── Save MAC persistently ──────────────────────────────────────────
mkdir -p "$CONFIG_DIR"
echo "$MAC" > "$MAC_FILE"
echo "Saved MAC to $MAC_FILE"

# ── Ensure pulseaudio-module-bluetooth is installed ────────────────
echo "[1/5] Ensuring pulseaudio-module-bluetooth is installed..."
if ! dpkg -s pulseaudio-module-bluetooth &>/dev/null; then
    sudo apt-get install -y pulseaudio-module-bluetooth
fi

# ── Enable BlueZ auto-reconnect in /etc/bluetooth/main.conf ───────
echo "[2/5] Enabling BlueZ auto-reconnect..."
BT_CONF="/etc/bluetooth/main.conf"
if [ -f "$BT_CONF" ]; then
    # Uncomment ReconnectAttempts and ReconnectIntervals if commented out
    if grep -q "^#ReconnectAttempts" "$BT_CONF"; then
        sudo sed -i 's/^#ReconnectAttempts=.*/ReconnectAttempts=7/' "$BT_CONF"
        echo "  Enabled ReconnectAttempts=7"
    fi
    if grep -q "^#ReconnectIntervals" "$BT_CONF"; then
        sudo sed -i 's/^#ReconnectIntervals=.*/ReconnectIntervals=1, 2, 4, 8, 16, 32, 64/' "$BT_CONF"
        echo "  Enabled ReconnectIntervals"
    fi
    # Restart bluetoothd to pick up the new config
    sudo systemctl restart bluetooth
    sleep 1
fi

# ── Restart PulseAudio ─────────────────────────────────────────────
echo "[3/5] Restarting PulseAudio..."
killall -9 pulseaudio 2>/dev/null || true
sleep 1
pulseaudio --start
sleep 1

# ── Connect and trust ──────────────────────────────────────────────
echo "[4/5] Connecting to $MAC and marking as trusted..."
bluetoothctl trust "$MAC" 2>/dev/null || true
bluetoothctl connect "$MAC"
sleep 2

SINK=$(pactl list sinks short | grep -i "bluez\|$(echo "$MAC" | tr ':' '_')" | awk '{print $2}' | head -1)
if [ -z "$SINK" ]; then
    echo "⚠ Bluetooth sink not found yet — try running again in a few seconds."
    echo "  Available sinks:"
    pactl list sinks short
    exit 1
fi
pactl set-default-sink "$SINK"
echo "✓ Connected. Default sink: $SINK"

# ── Install auto-reconnect systemd service ─────────────────────────
echo "[5/5] Installing auto-reconnect service..."
mkdir -p "$(dirname "$SERVICE_FILE")"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Auto-reconnect Bluetooth speaker ($MAC)
After=bluetooth.target pulseaudio.service
Wants=pulseaudio.service

[Service]
Type=simple
# Poll every 30s: reconnect if not connected, then set as default PA sink.
ExecStart=/bin/bash -c ' \\
  MAC=\$(cat "$MAC_FILE" 2>/dev/null) || exit 1; \\
  while true; do \\
    if bluetoothctl info "\$MAC" 2>/dev/null | grep -q "Connected: yes"; then \\
      sleep 30; continue; \\
    fi; \\
    bluetoothctl connect "\$MAC" 2>/dev/null; \\
    sleep 5; \\
    SINK=\$(pactl list sinks short 2>/dev/null \\
      | grep -i "bluez\\\|\\$(echo "\$MAC" | tr ":" "_")" \\
      | awk "{print \\\$2}" | head -1); \\
    [ -n "\$SINK" ] && pactl set-default-sink "\$SINK"; \\
    sleep 30; \\
  done'
Restart=always
RestartSec=15

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user restart "$SERVICE_NAME"
echo "✓ Service installed and started: $SERVICE_NAME"
echo ""
echo "The speaker will now reconnect automatically after reboots and power cycles."
echo "Check status: systemctl --user status $SERVICE_NAME"
