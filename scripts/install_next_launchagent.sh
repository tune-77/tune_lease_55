#!/bin/bash
# Install and start the persistent Next/FastAPI/Cloudflare launcher.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PLIST="$ROOT_DIR/scripts/launchd/com.tunelease.next.plist"
TARGET_PLIST="$HOME/Library/LaunchAgents/com.tunelease.next.plist"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.tunelease.next"

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT_DIR/logs/next"
cp "$SOURCE_PLIST" "$TARGET_PLIST"
plutil -lint "$TARGET_PLIST" >/dev/null

launchctl bootout "$SERVICE" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$TARGET_PLIST"
launchctl enable "$SERVICE"
launchctl kickstart -k "$SERVICE"

echo "Installed and started: $SERVICE"
echo "Plist: $TARGET_PLIST"
echo "Logs: $ROOT_DIR/logs/next/launchd.out.log"
