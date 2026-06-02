#!/bin/bash
# Start the Next/FastAPI/Cloudflare launcher detached from the caller.
#
# Use this from Codex or other interactive shells so an interrupted terminal
# session does not send TERM to run_next_stable.sh and trigger its cleanup trap.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/next

LOG_FILE="logs/next/launcher_detached.log"
PID_FILE="logs/next/launcher_detached.pid"

FORCE_RESTART="${FORCE_RESTART:-1}"
PUBLIC_TUNNEL="${PUBLIC_TUNNEL:-1}"
API_HOST="${API_HOST:-127.0.0.1}"
NEXT_HOST="${NEXT_HOST:-127.0.0.1}"
API_RELOAD="${API_RELOAD:-0}"
SKIP_STALE_LAUNCHER_SWEEP="${SKIP_STALE_LAUNCHER_SWEEP:-1}"

nohup env \
  FORCE_RESTART="$FORCE_RESTART" \
  PUBLIC_TUNNEL="$PUBLIC_TUNNEL" \
  API_HOST="$API_HOST" \
  NEXT_HOST="$NEXT_HOST" \
  API_RELOAD="$API_RELOAD" \
  SKIP_STALE_LAUNCHER_SWEEP="$SKIP_STALE_LAUNCHER_SWEEP" \
  bash run_next_stable.sh >"$LOG_FILE" 2>&1 &

launcher_pid=$!
echo "$launcher_pid" > "$PID_FILE"

echo "Detached launcher started: PID $launcher_pid"
echo "Log: $LOG_FILE"
