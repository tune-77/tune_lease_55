#!/bin/bash
# Install and start the persistent Next/FastAPI/Cloudflare launcher.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PLIST="$ROOT_DIR/scripts/launchd/com.tunelease.next.plist"
TARGET_PLIST="$HOME/Library/LaunchAgents/com.tunelease.next.plist"
AUTH_DIR="${TUNELEASE_CONFIG_DIR:-$HOME/Library/Application Support/tune_lease_55}"
AUTH_FILE="$AUTH_DIR/public_tunnel_auth"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.tunelease.next"

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT_DIR/logs/next" "$AUTH_DIR"
chmod 700 "$AUTH_DIR"

if [ -n "${PUBLIC_TUNNEL_AUTH:-}" ]; then
  tunnel_password="$PUBLIC_TUNNEL_AUTH"
elif [ -L "$AUTH_FILE" ]; then
  echo "Refusing to read a symlinked public tunnel credential: $AUTH_FILE" >&2
  exit 1
elif [ -s "$AUTH_FILE" ]; then
  tunnel_password=""
  chmod 600 "$AUTH_FILE"
  echo "Reusing the existing public tunnel credential."
elif [ -t 0 ]; then
  read -r -s -p "Public tunnel password (12+ characters): " tunnel_password
  echo
else
  echo "No public tunnel credential is provisioned." >&2
  echo "Run this installer interactively or set PUBLIC_TUNNEL_AUTH for this command." >&2
  exit 1
fi

if [ -n "$tunnel_password" ]; then
  if [[ "$tunnel_password" == *$'\n'* || "$tunnel_password" == *$'\r'* ]]; then
    echo "Public tunnel password must not contain line breaks." >&2
    exit 1
  fi
  if [ "${#tunnel_password}" -lt 12 ]; then
    echo "Public tunnel password must be at least 12 characters." >&2
    exit 1
  fi
  umask 077
  temp_auth_file="$(mktemp "$AUTH_DIR/.public_tunnel_auth.XXXXXX")"
  trap 'rm -f "$temp_auth_file"' EXIT
  printf '%s' "$tunnel_password" > "$temp_auth_file"
  chmod 600 "$temp_auth_file"
  mv "$temp_auth_file" "$AUTH_FILE"
  trap - EXIT
  unset tunnel_password
  echo "Stored the public tunnel credential with owner-only permissions."
fi

cp "$SOURCE_PLIST" "$TARGET_PLIST"
plutil -insert EnvironmentVariables.PUBLIC_TUNNEL_AUTH_FILE -string "$AUTH_FILE" "$TARGET_PLIST"
plutil -lint "$TARGET_PLIST" >/dev/null

launchctl bootout "$SERVICE" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$TARGET_PLIST"
launchctl enable "$SERVICE"
launchctl kickstart -k "$SERVICE"

echo "Installed and started: $SERVICE"
echo "Plist: $TARGET_PLIST"
echo "Tunnel credential: $AUTH_FILE"
echo "Logs: $ROOT_DIR/logs/next/launchd.out.log"
