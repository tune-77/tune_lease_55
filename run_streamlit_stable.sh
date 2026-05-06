#!/bin/bash
# Streamlit stable launcher with restart loop and crash logs.

set -euo pipefail

cd "$(dirname "$0")"

PORT="${STREAMLIT_PORT:-8505}"
LOG_DIR="logs/streamlit"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/streamlit_${TS}.log"
PID_FILE="$LOG_DIR/streamlit_${PORT}.pid"

stop_port_process() {
  local port="$1"
  local pids
  pids="$(lsof -ti :"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Stopping existing Streamlit on port $port: $pids" | tee -a "$LOG_FILE"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

cleanup() {
  echo "$(date '+%F %T') received stop signal" | tee -a "$LOG_FILE"
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
  exit 0
}
trap cleanup INT TERM

stop_port_process "$PORT"

echo "Streamlit stable launcher"
echo "URL: http://localhost:${PORT}"
echo "Log: $LOG_FILE"
echo "Press Ctrl+C to stop."
echo ""

while true; do
  echo "$(date '+%F %T') starting Streamlit on port ${PORT}" | tee -a "$LOG_FILE"
  python3 -m streamlit run tune_lease_55.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.fileWatcherType none \
    --server.runOnSave false \
    --browser.gatherUsageStats false \
    >>"$LOG_FILE" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_FILE"
  wait "$pid"
  code=$?
  rm -f "$PID_FILE"
  echo "$(date '+%F %T') Streamlit exited with code ${code}; restarting in 3 seconds" | tee -a "$LOG_FILE"
  sleep 3
done
