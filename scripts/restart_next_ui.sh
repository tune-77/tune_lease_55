#!/bin/bash
# Efficient Next UI refresh:
# build only when needed, sync standalone assets, restart Next only, then verify static assets.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH"

API_HOST="${API_HOST:-127.0.0.1}"
NEXT_HOST="${NEXT_HOST:-127.0.0.1}"
NEXT_PORT="${NEXT_PORT:-3000}"
PUBLIC_TUNNEL="${PUBLIC_TUNNEL:-1}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
VERIFY_PAGES="${VERIFY_PAGES:-/ /home /lease-intelligence /multi-shion-demo}"

LOG_DIR="logs/next"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
BUILD_LOG="$LOG_DIR/ui_build_${TS}.log"
BUILD_STAMP="$LOG_DIR/frontend_build.stamp"
NEXT_PID_FILE="$LOG_DIR/next_${NEXT_PORT}.pid"
NEXT_SUPERVISOR_PID_FILE="$LOG_DIR/next_${NEXT_PORT}.supervisor.pid"
LAUNCH_AGENT_LABEL="${LAUNCH_AGENT_LABEL:-com.tunelease.next}"

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

sync_standalone_assets() {
  if [ ! -d "frontend/.next/standalone" ]; then
    echo "Standalone output not found; skipping standalone asset sync."
    return 0
  fi

  mkdir -p frontend/.next/standalone/.next
  cp -R frontend/.next/static frontend/.next/standalone/.next/static
  cp -R frontend/public frontend/.next/standalone/public
  echo "Synced standalone static/public assets."
}

wait_for_next() {
  local url="http://${NEXT_HOST}:${NEXT_PORT}/"
  local attempts="${1:-90}"
  local i
  for i in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Next did not become healthy: $url" >&2
  return 1
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

run_next_supervisor_alive() {
  ps -eo pid=,command= 2>/dev/null | awk '$0 ~ /[r]un_next_stable\.sh/ {found=1} END {exit !found}'
}

start_local_next_supervisor() {
  local next_log="$LOG_DIR/next_ui_${TS}.log"

  if [ ! -f "frontend/.next/standalone/server.js" ]; then
    echo "Standalone server is missing; run with FORCE_BUILD=1 first." >&2
    return 1
  fi

  echo "Starting lightweight Next supervisor."
  (
    echo "$BASHPID" > "$NEXT_SUPERVISOR_PID_FILE"
    while true; do
      (
        cd frontend/.next/standalone
        PORT="$NEXT_PORT" HOSTNAME="$NEXT_HOST" node server.js
      )
      echo "$(date '+%Y-%m-%d %H:%M:%S') Next.js exited; restarting in 1 seconds"
      sleep 1
    done
  ) >>"$next_log" 2>&1 &

  echo "$!" > "$NEXT_SUPERVISOR_PID_FILE"
  echo "Next log: $next_log"
}

restart_next_only() {
  echo "Restarting Next only..."

  local next_pids
  next_pids="$(lsof -ti :"$NEXT_PORT" 2>/dev/null || true)"
  if [ -n "$next_pids" ]; then
    echo "Nudging Next.js on port ${NEXT_PORT}: ${next_pids}"
    kill $next_pids 2>/dev/null || true
    if wait_for_next 30; then
      return 0
    fi
    echo "Existing supervisor did not restore Next quickly; starting lightweight fallback."
  fi

  start_local_next_supervisor
  wait_for_next
}

extract_static_urls() {
  local page="$1"
  local html
  html="$(curl -fsS --max-time 10 "http://${NEXT_HOST}:${NEXT_PORT}${page}" || true)"
  if [ -z "$html" ]; then
    echo "Could not fetch page for static verification: ${page}" >&2
    return 1
  fi
  printf "%s" "$html" \
    | grep -oE '/_next/static/(css|chunks)/[^"'"'"'<> ]+' \
    | sort -u
}

verify_static_assets() {
  local failed=0
  local page
  local asset
  local status
  local page_assets

  echo "Verifying Next static assets..."
  for page in $VERIFY_PAGES; do
    echo "  Page: ${page}"
    page_assets="$(extract_static_urls "$page")" || {
      failed=1
      continue
    }
    while read -r asset; do
      [ -z "$asset" ] && continue
      status="$(curl -o /dev/null -sS -w '%{http_code}' --max-time 10 "http://${NEXT_HOST}:${NEXT_PORT}${asset}" || true)"
      if [ "$status" != "200" ]; then
        echo "    MISSING ${status}: ${asset}" >&2
        failed=1
      fi
    done <<< "$page_assets"
  done

  if [ "$failed" != "0" ]; then
    echo "Static asset verification failed." >&2
    return 1
  fi
  echo "Static asset verification OK."
}

main() {
  if frontend_build_needed; then
    echo "Building frontend..."
    (cd frontend && npm run build) >>"$BUILD_LOG" 2>&1
    touch "$BUILD_STAMP"
    echo "Build log: $BUILD_LOG"
  else
    echo "Skipping frontend build; no source changes since last successful build."
  fi

  sync_standalone_assets
  restart_next_only
  verify_static_assets

  echo "Next UI refresh complete."
  echo "  Local: http://${NEXT_HOST}:${NEXT_PORT}"
}

main "$@"
