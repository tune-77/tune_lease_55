#!/bin/bash
# Next.js + FastAPI stable launcher with restart loops.

set -euo pipefail

cd "$(dirname "$0")"

API_PORT="${API_PORT:-8000}"
NEXT_PORT="${NEXT_PORT:-3000}"
API_HOST="${API_HOST:-127.0.0.1}"
NEXT_HOST="${NEXT_HOST:-127.0.0.1}"
PUBLIC_TUNNEL="${PUBLIC_TUNNEL:-0}"
LOG_DIR="logs/next"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
API_LOG="$LOG_DIR/api_${TS}.log"
NEXT_LOG="$LOG_DIR/next_${TS}.log"
BUILD_LOG="$LOG_DIR/build_${TS}.log"
API_PID_FILE="$LOG_DIR/api_${API_PORT}.pid"
NEXT_PID_FILE="$LOG_DIR/next_${NEXT_PORT}.pid"
TUNNEL_PID_FILE="$LOG_DIR/tunnel_${NEXT_PORT}.pid"
TUNNEL_LOG="$LOG_DIR/tunnel_${TS}.log"

stop_port_process() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -ti :"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Stopping existing ${label} on port ${port}: ${pids}"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

cleanup() {
  echo ""
  echo "Stopping services..."
  if [ -f "$API_PID_FILE" ]; then
    kill "$(cat "$API_PID_FILE")" 2>/dev/null || true
    rm -f "$API_PID_FILE"
  fi
  if [ -f "$NEXT_PID_FILE" ]; then
    kill "$(cat "$NEXT_PID_FILE")" 2>/dev/null || true
    rm -f "$NEXT_PID_FILE"
  fi
  if [ -f "$TUNNEL_PID_FILE" ]; then
    kill "$(cat "$TUNNEL_PID_FILE")" 2>/dev/null || true
    rm -f "$TUNNEL_PID_FILE"
  fi
  exit 0
}
trap cleanup INT TERM

stop_port_process "$API_PORT" "FastAPI"
stop_port_process "$NEXT_PORT" "Next.js"

echo "Building frontend..."
(cd frontend && npm run build) >>"$BUILD_LOG" 2>&1
echo "Build log: $BUILD_LOG"
echo ""

echo "Starting FastAPI on http://${API_HOST}:${API_PORT}"
while true; do
  python -m uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" >>"$API_LOG" 2>&1 &
  api_pid=$!
  echo "$api_pid" > "$API_PID_FILE"
  wait "$api_pid" || true
  rm -f "$API_PID_FILE"
  echo "$(date '+%F %T') FastAPI exited; restarting in 3 seconds" | tee -a "$API_LOG"
  sleep 3
done &
API_SUPERVISOR_PID=$!

echo "Starting Next.js on http://${NEXT_HOST}:${NEXT_PORT}"
while true; do
  (cd frontend && npm run start -- --hostname "$NEXT_HOST" --port "$NEXT_PORT") >>"$NEXT_LOG" 2>&1 &
  next_pid=$!
  echo "$next_pid" > "$NEXT_PID_FILE"
  wait "$next_pid" || true
  rm -f "$NEXT_PID_FILE"
  echo "$(date '+%F %T') Next.js exited; restarting in 3 seconds" | tee -a "$NEXT_LOG"
  sleep 3
done &
NEXT_SUPERVISOR_PID=$!

if [ "$PUBLIC_TUNNEL" = "1" ]; then
  echo "Starting Cloudflare Tunnel for http://${NEXT_HOST}:${NEXT_PORT}"
  (cloudflared tunnel --url "http://${NEXT_HOST}:${NEXT_PORT}" | tee "$TUNNEL_LOG") &
  tunnel_pid=$!
  echo "$tunnel_pid" > "$TUNNEL_PID_FILE"
fi

echo ""
echo "==================================="
echo "Stable launcher started"
echo "  API   : http://${API_HOST}:${API_PORT}"
echo "  Next  : http://${NEXT_HOST}:${NEXT_PORT}"
if [ "$PUBLIC_TUNNEL" = "1" ]; then
  echo "  Tunnel: see ${TUNNEL_LOG}"
fi
echo "  Logs  : ${LOG_DIR}"
echo "==================================="
echo ""
echo "Press Ctrl+C to stop."

wait "$API_SUPERVISOR_PID" "$NEXT_SUPERVISOR_PID"
