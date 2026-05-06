#!/bin/bash
# tune_lease_55 + Flask 起動スクリプト

set -euo pipefail

cd "$(dirname "$0")"

STREAMLIT_PORT="${STREAMLIT_PORT:-8505}"
FLASK_PORT="${FLASK_PORT:-5050}"

stop_port_process() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -ti :"$port" 2>/dev/null || true)"

  if [ -z "$pids" ]; then
    echo "$label ($port) は稼働していません"
    return
  fi

  echo "$label ($port) を停止中: $pids"
  kill $pids 2>/dev/null || true
  sleep 1

  local alive
  alive="$(lsof -ti :"$port" 2>/dev/null || true)"
  if [ -n "$alive" ]; then
    echo "$label ($port) が残っているため強制停止します: $alive"
    kill -9 $alive 2>/dev/null || true
  fi
}

echo "=== 既存プロセスの停止 ==="
stop_port_process "$STREAMLIT_PORT" "Streamlit"
stop_port_process "$FLASK_PORT" "Flask"

echo ""
echo "=== Flask 起動（ポート${FLASK_PORT}）==="
python web/app.py &
FLASK_PID=$!
echo "Flask PID: $FLASK_PID"
sleep 2

echo ""
echo "=== Streamlit 安定起動（ポート${STREAMLIT_PORT}）==="
STREAMLIT_PORT="$STREAMLIT_PORT" bash run_streamlit_stable.sh &
STREAMLIT_PID=$!
echo "Streamlit supervisor PID: $STREAMLIT_PID"

echo ""
echo "==================================="
echo "✅ 起動完了"
echo "  審査アプリ  : http://localhost:${STREAMLIT_PORT}"
echo "  簡易審査    : http://localhost:${FLASK_PORT}"
echo "==================================="
echo ""
echo "終了するには Ctrl+C を押してください"

cleanup() {
  kill "$FLASK_PID" "$STREAMLIT_PID" 2>/dev/null || true
  echo "停止しました"
  exit
}

trap cleanup INT TERM
wait
