#!/bin/bash
# Next.js + FastAPI stable launcher with restart loops.

set -euo pipefail

cd "$(dirname "$0")"

# Non-login shells launched by Codex/launchd may not inherit the user's PATH.
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH"

FORCE_RESTART="${FORCE_RESTART:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
REUSE_RUNNING="${REUSE_RUNNING:-1}"
RESTART_SCOPE="${RESTART_SCOPE:-all}"  # all | status | api | next | tunnel
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-1}"
API_RELOAD="${API_RELOAD:-0}"
SKIP_STALE_LAUNCHER_SWEEP="${SKIP_STALE_LAUNCHER_SWEEP:-0}"

API_PORT="${API_PORT:-8000}"
NEXT_PORT="${NEXT_PORT:-3000}"
API_HOST="${API_HOST:-127.0.0.1}"
NEXT_HOST="${NEXT_HOST:-127.0.0.1}"
PUBLIC_TUNNEL="${PUBLIC_TUNNEL:-0}"
# Named Tunnel（固定URL・認証あり）を使う場合はこの2つを設定する。
# 未設定なら従来どおり quick tunnel（認証不要・URL変動・本番非推奨）にフォールバックする。
CLOUDFLARE_TUNNEL_CONFIG="${CLOUDFLARE_TUNNEL_CONFIG:-}"
CLOUDFLARE_TUNNEL_HOSTNAME="${CLOUDFLARE_TUNNEL_HOSTNAME:-}"
LOG_DIR="logs/next"
mkdir -p "$LOG_DIR"

named_tunnel_active() {
  [ -n "$CLOUDFLARE_TUNNEL_CONFIG" ] && [ -f "$CLOUDFLARE_TUNNEL_CONFIG" ]
}

pid_file_alive() {
  local file="$1"
  local pid
  if [ ! -f "$file" ]; then
    return 1
  fi
  pid="$(cat "$file" 2>/dev/null || true)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

latest_tunnel_url() {
  if named_tunnel_active && [ -n "$CLOUDFLARE_TUNNEL_HOSTNAME" ]; then
    echo "https://${CLOUDFLARE_TUNNEL_HOSTNAME}"
    return
  fi
  local latest_log
  latest_log="$(ls -t "$LOG_DIR"/tunnel_*.log 2>/dev/null | head -1 || true)"
  if [ -n "$latest_log" ]; then
    rg -o "https://[a-zA-Z0-9-]+\\.trycloudflare\\.com" "$latest_log" 2>/dev/null | tail -1 || true
  fi
}

print_status() {
  local api_state="DOWN"
  local next_state="DOWN"
  local tunnel_url
  if curl -fsS --max-time 2 "http://${API_HOST}:${API_PORT}/docs" >/dev/null 2>&1; then
    api_state="OK"
  elif lsof -ti :"$API_PORT" >/dev/null 2>&1; then
    api_state="LISTENING"
  fi
  if curl -fsS --max-time 2 "http://${NEXT_HOST}:${NEXT_PORT}/" >/dev/null 2>&1; then
    next_state="OK"
  elif lsof -ti :"$NEXT_PORT" >/dev/null 2>&1; then
    next_state="LISTENING"
  fi
  tunnel_url="$(latest_tunnel_url)"
  echo "Status:"
  echo "  API  : ${api_state} http://${API_HOST}:${API_PORT}"
  echo "  Next : ${next_state} http://${NEXT_HOST}:${NEXT_PORT}"
  if [ -n "$tunnel_url" ]; then
    echo "  Tunnel: ${tunnel_url}"
  else
    echo "  Tunnel: not found"
  fi
}

if [ "$RESTART_SCOPE" = "status" ]; then
  print_status
  exit 0
fi

if [ "$RESTART_SCOPE" = "api" ]; then
  api_pids="$(lsof -ti :"$API_PORT" 2>/dev/null || true)"
  if [ -n "$api_pids" ]; then
    echo "API-only restart requested; nudging FastAPI on port ${API_PORT}: ${api_pids}"
    kill $api_pids 2>/dev/null || true
    if pid_file_alive "$LOG_DIR/api_${API_PORT}.supervisor.pid"; then
      exit 0
    fi
    echo "FastAPI supervisor is not alive; continuing with full launch."
    FORCE_RESTART=1
  else
    echo "API-only restart requested, but FastAPI is not listening; continuing with full launch."
    FORCE_RESTART=1
  fi
fi

if [ "$RESTART_SCOPE" = "next" ]; then
  next_pids="$(lsof -ti :"$NEXT_PORT" 2>/dev/null || true)"
  if [ -n "$next_pids" ]; then
    echo "Next-only restart requested; nudging Next.js on port ${NEXT_PORT}: ${next_pids}"
    kill $next_pids 2>/dev/null || true
    if pid_file_alive "$LOG_DIR/next_${NEXT_PORT}.supervisor.pid"; then
      exit 0
    fi
    echo "Next.js supervisor is not alive; continuing with full launch."
    FORCE_RESTART=1
  else
    echo "Next-only restart requested, but Next.js is not listening; continuing with full launch."
    FORCE_RESTART=1
  fi
fi

if [ "$RESTART_SCOPE" = "tunnel" ]; then
  if named_tunnel_active; then
    tunnel_pids="$(pgrep -f "cloudflared tunnel --config ${CLOUDFLARE_TUNNEL_CONFIG} run" | tr '\n' ' ')"
  else
    tunnel_pids="$(ps -eo pid=,command= 2>/dev/null | awk -v url="http://${NEXT_HOST}:${NEXT_PORT}" '$0 ~ /[c]loudflared tunnel --url/ && index($0, url) {print $1}' | tr '\n' ' ')"
  fi
  if [ -n "$tunnel_pids" ]; then
    echo "Tunnel-only restart requested; nudging Cloudflare Tunnel:${tunnel_pids}"
    kill $tunnel_pids 2>/dev/null || true
    if pid_file_alive "$LOG_DIR/tunnel_${NEXT_PORT}.supervisor.pid"; then
      exit 0
    fi
    echo "Cloudflare Tunnel supervisor is not alive; continuing with full launch."
    FORCE_RESTART=1
  else
    echo "Tunnel-only restart requested, but no matching Cloudflare Tunnel is running."
    if [ "$PUBLIC_TUNNEL" = "1" ]; then
      FORCE_RESTART=1
    else
      exit 0
    fi
  fi
fi

# ── ロックファイル: 二重起動を防ぐ ──────────────────────────────
LOCK_FILE="/tmp/tune_lease_launcher.lock"
if [ -f "$LOCK_FILE" ]; then
  existing_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
    if [ "$FORCE_RESTART" = "1" ]; then
      echo "Stopping existing launcher (PID $existing_pid) for forced restart..."
      kill "$existing_pid" 2>/dev/null || true
      sleep 2
      rm -f "$LOCK_FILE"
    else
      echo "Launcher is already running (PID $existing_pid). Reusing existing services."
      echo "To force a restart: FORCE_RESTART=1 bash run_next_stable.sh"
      exit 0
    fi
  else
    echo "Stale lock file found. Removing..."
    rm -f "$LOCK_FILE"
  fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT
# ───────────────────────────────────────────────────────────────────

TS="$(date +%Y%m%d_%H%M%S)"
API_LOG="$LOG_DIR/api_${TS}.log"
NEXT_LOG="$LOG_DIR/next_${TS}.log"
BUILD_LOG="$LOG_DIR/build_${TS}.log"
API_PID_FILE="$LOG_DIR/api_${API_PORT}.pid"
NEXT_PID_FILE="$LOG_DIR/next_${NEXT_PORT}.pid"
API_SUPERVISOR_PID_FILE="$LOG_DIR/api_${API_PORT}.supervisor.pid"
NEXT_SUPERVISOR_PID_FILE="$LOG_DIR/next_${NEXT_PORT}.supervisor.pid"
TUNNEL_SUPERVISOR_PID_FILE="$LOG_DIR/tunnel_${NEXT_PORT}.supervisor.pid"
TUNNEL_PID_FILE="$LOG_DIR/tunnel_${NEXT_PORT}.pid"
TUNNEL_LOG="$LOG_DIR/tunnel_${TS}.log"
BUILD_STAMP="$LOG_DIR/frontend_build.stamp"

kill_pids() {
  local pids="$1"
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
  fi
}

stop_port_process() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -ti :"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Stopping existing ${label} on port ${port}: ${pids}"
    kill_pids "$pids"
    sleep 1
  fi
  pids="$(lsof -ti :"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Force stopping remaining ${label} on port ${port}: ${pids}"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

stop_cloudflare_tunnels() {
  local pids=""
  local pid
  while read -r pid; do
    if [ -n "$pid" ]; then
      pids="${pids} ${pid}"
    fi
  done < <(ps -eo pid=,command= 2>/dev/null | awk -v url="http://${NEXT_HOST}:${NEXT_PORT}" '$0 ~ /[c]loudflared tunnel --url/ && index($0, url) {print $1}')

  if [ -n "$pids" ]; then
    echo "Stopping stale Cloudflare Tunnel processes:${pids}"
    kill_pids "$pids"
    sleep 1
  fi
}

is_current_launcher_family() {
  local candidate="$1"
  local pid="$$"
  while [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "1" ]; do
    if [ "$candidate" = "$pid" ]; then
      return 0
    fi
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  done
  return 1
}

stop_existing_launchers() {
  local stale_pids=""
  local pid
  while read -r pid; do
    if [ -n "$pid" ] && ! is_current_launcher_family "$pid"; then
      stale_pids="${stale_pids} ${pid}"
    fi
  done < <(ps -eo pid=,command= 2>/dev/null | awk '$0 ~ /[r]un_next_stable\.sh/ {print $1}')

  if [ -n "$stale_pids" ]; then
    echo "Stopping stale launcher processes:${stale_pids}"
    kill_pids "$stale_pids"
    sleep 2
  fi
}

port_is_listening() {
  local port="$1"
  lsof -ti :"$port" >/dev/null 2>&1
}

http_ok() {
  local url="$1"
  curl -fsS --max-time 2 "$url" >/dev/null 2>&1
}

frontend_build_needed() {
  if [ "$SKIP_BUILD" = "1" ]; then
    return 1
  fi
  if [ "$FORCE_BUILD" = "1" ]; then
    return 0
  fi
  if [ ! -f "frontend/.next/BUILD_ID" ]; then
    return 0
  fi
  if [ ! -f "$BUILD_STAMP" ]; then
    touch -r "frontend/.next/BUILD_ID" "$BUILD_STAMP" 2>/dev/null || touch "$BUILD_STAMP"
  fi
  if find \
      frontend/src \
      frontend/package.json \
      frontend/package-lock.json \
      frontend/next.config.ts \
      frontend/tsconfig.json \
      frontend/postcss.config.mjs \
      frontend/eslint.config.mjs \
      -newer "$BUILD_STAMP" -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

cleanup() {
  echo ""
  echo "Stopping services..."
  if [ -f "$API_SUPERVISOR_PID_FILE" ]; then
    kill "$(cat "$API_SUPERVISOR_PID_FILE")" 2>/dev/null || true
    rm -f "$API_SUPERVISOR_PID_FILE"
  fi
  if [ -f "$NEXT_SUPERVISOR_PID_FILE" ]; then
    kill "$(cat "$NEXT_SUPERVISOR_PID_FILE")" 2>/dev/null || true
    rm -f "$NEXT_SUPERVISOR_PID_FILE"
  fi
  if [ -f "$API_PID_FILE" ]; then
    kill "$(cat "$API_PID_FILE")" 2>/dev/null || true
    rm -f "$API_PID_FILE"
  fi
  if [ -f "$NEXT_PID_FILE" ]; then
    kill "$(cat "$NEXT_PID_FILE")" 2>/dev/null || true
    rm -f "$NEXT_PID_FILE"
  fi
  # Cloudflare Tunnel はここでは止めない。再起動のたびに quick tunnel の URL が
  # 変わるのを防ぐため、トンネルの停止・再起動は RESTART_SCOPE=tunnel 限定にする。
  stop_port_process "$API_PORT" "FastAPI"
  stop_port_process "$NEXT_PORT" "Next.js"
  rm -f "$LOCK_FILE"
  exit 0
}
trap cleanup INT TERM

if [ "$FORCE_RESTART" != "1" ] && [ "$REUSE_RUNNING" = "1" ]; then
  if port_is_listening "$API_PORT" && port_is_listening "$NEXT_PORT" \
      && http_ok "http://${API_HOST}:${API_PORT}/docs" \
      && http_ok "http://${NEXT_HOST}:${NEXT_PORT}/"; then
    echo "FastAPI and Next.js are already running. No restart/build needed."
    echo "  API  : http://${API_HOST}:${API_PORT}"
    echo "  Next : http://${NEXT_HOST}:${NEXT_PORT}"
    tunnel_url="$(latest_tunnel_url)"
    if [ -n "$tunnel_url" ]; then
      echo "  Tunnel: ${tunnel_url}"
    fi
    echo "To force a restart: FORCE_RESTART=1 bash run_next_stable.sh"
    exit 0
  fi
fi

if [ "$FORCE_RESTART" = "1" ]; then
  if [ "$SKIP_STALE_LAUNCHER_SWEEP" != "1" ]; then
    stop_existing_launchers
  fi
  # Cloudflare Tunnel は殺さない（URL 維持のため）。起動済みなら後段で再利用される。
  # lsof ベースの stop_port_process だけでは supervisor 終了後に re-parent された
  # uvicorn が残存するレースがある。名前で直接 kill して確実に排除する。
  pkill -f "uvicorn api.main:app" 2>/dev/null || true
  sleep 1
  stop_port_process "$API_PORT" "FastAPI"
  stop_port_process "$NEXT_PORT" "Next.js"
else
  if port_is_listening "$API_PORT" || port_is_listening "$NEXT_PORT"; then
    echo "Target ports are already occupied. Reusing existing processes; no restart/build performed."
    echo "  API port ${API_PORT}:  $(lsof -ti :"$API_PORT" 2>/dev/null || echo free)"
    echo "  Next port ${NEXT_PORT}: $(lsof -ti :"$NEXT_PORT" 2>/dev/null || echo free)"
    echo "To replace them: FORCE_RESTART=1 bash run_next_stable.sh"
    exit 0
  fi
fi

if frontend_build_needed; then
  echo "Building frontend..."
  (cd frontend && npm run build) >>"$BUILD_LOG" 2>&1
  touch "$BUILD_STAMP"
  echo "Build log: $BUILD_LOG"
  # standalone モード用: static と public を standalone/ 以下にコピー
  if [ -d "frontend/.next/standalone" ]; then
    mkdir -p frontend/.next/standalone/.next/static frontend/.next/standalone/public
    rsync -a --delete frontend/.next/static/ frontend/.next/standalone/.next/static/ 2>/dev/null || true
    rsync -a --delete frontend/public/ frontend/.next/standalone/public/ 2>/dev/null || true
    echo "Copied static assets to standalone directory."
  fi
else
  echo "Skipping frontend build; no frontend source changes since last successful build."
  echo "Build log: not created"
fi
echo ""

echo "Starting FastAPI on http://${API_HOST}:${API_PORT}"
# .venv が存在すればそちらを優先（uv run はキャッシュ権限エラーを起こす場合があるため）
if [ -f ".venv/bin/python" ]; then
  API_RUNNER=(".venv/bin/python")
elif command -v uv >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then
  API_RUNNER=(uv run --no-sync python)
else
  API_RUNNER=(python)
fi
while true; do
  api_args=(-m uvicorn api.main:app --host "$API_HOST" --port "$API_PORT")
  if [ "$API_RELOAD" = "1" ]; then
    # 開発時だけ明示的にreloadを有効化する。通常運用ではreloadなしにして、
    # AIモデルロード中のSIGTERM・セマフォリーク・親だけ残る不安定化を避ける。
    api_args+=(--reload)
    api_args+=(--reload-exclude "mobile_app/.vector_index/*")
    api_args+=(--reload-exclude "mobile_app/.embeddings_cache.json")
    api_args+=(--reload-exclude "data/*")
    api_args+=(--reload-exclude "logs/*")
    api_args+=(--reload-exclude "**/__pycache__/*")
    api_args+=(--reload-exclude "**/*.pyc")
    api_args+=(--reload-exclude ".claude/*")
    api_args+=(--reload-exclude "**/.DS_Store")
    api_args+=(--reload-exclude "frontend/*")
    api_args+=(--reload-exclude "models/*")
    api_args+=(--reload-exclude "**/*.log")
    api_args+=(--reload-exclude "**/*.jsonl")
    api_args+=(--reload-exclude "**/*.db")
    api_args+=(--reload-exclude "**/*.sqlite*")
    api_args+=(--reload-exclude "api/chroma_db/*")
  fi
  "${API_RUNNER[@]}" "${api_args[@]}" >>"$API_LOG" 2>&1 &
  api_pid=$!
  echo "$api_pid" > "$API_PID_FILE"
  wait "$api_pid" || true
  rm -f "$API_PID_FILE"
  echo "$(date '+%F %T') FastAPI exited; restarting in ${RESTART_DELAY_SECONDS} seconds" | tee -a "$API_LOG"
  sleep "$RESTART_DELAY_SECONDS"
done &
API_SUPERVISOR_PID=$!
echo "$API_SUPERVISOR_PID" > "$API_SUPERVISOR_PID_FILE"

echo "Starting Next.js on http://${NEXT_HOST}:${NEXT_PORT}"
while true; do
  if [ -f "frontend/.next/standalone/server.js" ]; then
    (cd frontend/.next/standalone && HOSTNAME="$NEXT_HOST" PORT="$NEXT_PORT" node server.js) >>"$NEXT_LOG" 2>&1 &
  else
    (cd frontend && npm run start -- --hostname "$NEXT_HOST" --port "$NEXT_PORT") >>"$NEXT_LOG" 2>&1 &
  fi
  next_pid=$!
  echo "$next_pid" > "$NEXT_PID_FILE"
  wait "$next_pid" || true
  rm -f "$NEXT_PID_FILE"
  echo "$(date '+%F %T') Next.js exited; restarting in ${RESTART_DELAY_SECONDS} seconds" | tee -a "$NEXT_LOG"
  sleep "$RESTART_DELAY_SECONDS"
done &
NEXT_SUPERVISOR_PID=$!
echo "$NEXT_SUPERVISOR_PID" > "$NEXT_SUPERVISOR_PID_FILE"

running_tunnel_pid() {
  if named_tunnel_active; then
    pgrep -f "cloudflared tunnel --config ${CLOUDFLARE_TUNNEL_CONFIG} run" | head -1
  else
    ps -eo pid=,command= 2>/dev/null \
      | awk -v url="http://${NEXT_HOST}:${NEXT_PORT}" '$0 ~ /[c]loudflared tunnel --url/ && index($0, url) {print $1; exit}'
  fi
}

if [ "$PUBLIC_TUNNEL" = "1" ]; then
  if named_tunnel_active; then
    echo "Starting Cloudflare Named Tunnel (config: ${CLOUDFLARE_TUNNEL_CONFIG})"
  else
    echo "Starting Cloudflare Tunnel for http://${NEXT_HOST}:${NEXT_PORT}"
  fi
  while true; do
    existing_tunnel_pid="$(running_tunnel_pid)"
    if [ -n "$existing_tunnel_pid" ]; then
      # 既存トンネルを再利用する（quick tunnel の URL を維持）。死んだら次周で起動。
      while kill -0 "$existing_tunnel_pid" 2>/dev/null; do
        sleep 5
      done
      echo "$(date '+%F %T') Reused Cloudflare Tunnel exited; starting a new one" | tee -a "$TUNNEL_LOG"
      continue
    fi
    if named_tunnel_active; then
      cloudflared tunnel --config "$CLOUDFLARE_TUNNEL_CONFIG" run >>"$TUNNEL_LOG" 2>&1 &
    else
      cloudflared tunnel --url "http://${NEXT_HOST}:${NEXT_PORT}" >>"$TUNNEL_LOG" 2>&1 &
    fi
    tunnel_pid=$!
    echo "$tunnel_pid" > "$TUNNEL_PID_FILE"
    wait "$tunnel_pid" || true
    rm -f "$TUNNEL_PID_FILE"
    echo "$(date '+%F %T') Cloudflare Tunnel exited; restarting in ${RESTART_DELAY_SECONDS} seconds" | tee -a "$TUNNEL_LOG"
    sleep "$RESTART_DELAY_SECONDS"
  done &
  TUNNEL_SUPERVISOR_PID=$!
  echo "$TUNNEL_SUPERVISOR_PID" > "$TUNNEL_SUPERVISOR_PID_FILE"
fi

echo ""
echo "==================================="
echo "Stable launcher started"
echo "  API   : http://${API_HOST}:${API_PORT}"
echo "  Next  : http://${NEXT_HOST}:${NEXT_PORT}"
if [ "$PUBLIC_TUNNEL" = "1" ]; then
  tunnel_url=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    tunnel_url="$(latest_tunnel_url)"
    if [ -n "$tunnel_url" ]; then
      break
    fi
    sleep 0.5
  done
  if [ -n "$tunnel_url" ]; then
    echo "  Tunnel: ${tunnel_url}"
  else
    echo "  Tunnel: see ${TUNNEL_LOG}"
  fi
fi
echo "  Logs  : ${LOG_DIR}"
echo "==================================="
echo ""
echo "Press Ctrl+C to stop."

if [ "$PUBLIC_TUNNEL" = "1" ]; then
  wait "$API_SUPERVISOR_PID" "$NEXT_SUPERVISOR_PID" "$TUNNEL_SUPERVISOR_PID"
else
  wait "$API_SUPERVISOR_PID" "$NEXT_SUPERVISOR_PID"
fi
