#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${PORT:-8080}"
FASTAPI_HOST="${FASTAPI_HOST:-127.0.0.1}"
FASTAPI_PORT="${FASTAPI_PORT:-8000}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
OBSIDIAN_VAULT_PATH="${OBSIDIAN_VAULT_PATH:-}"
BUNDLE_ROOT="${CLOUDRUN_BUNDLE_DIR:-$(pwd)/.cloudrun_bundle}"
NEXT_SERVER="${NEXT_SERVER:-frontend/server.js}"

if [[ ! -f "$NEXT_SERVER" && -f "frontend/.next/standalone/server.js" ]]; then
  NEXT_SERVER="frontend/.next/standalone/server.js"
fi
if [[ ! -f "$NEXT_SERVER" ]]; then
  echo "Next standalone server was not found: $NEXT_SERVER" >&2
  exit 1
fi

mkdir -p "$DATA_DIR"

seed_dir() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    if [[ -z "$(find "$dst" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      cp -R "$src"/. "$dst"/
    fi
  fi
}

seed_dir "$BUNDLE_ROOT/data" "$DATA_DIR"
if [[ -n "$OBSIDIAN_VAULT_PATH" ]]; then
  seed_dir "$BUNDLE_ROOT/obsidian_vault" "$OBSIDIAN_VAULT_PATH"
fi

python -m uvicorn api.main:app \
  --host "$FASTAPI_HOST" \
  --port "$FASTAPI_PORT" \
  --workers 1 &
api_pid=$!

cleanup() {
  kill -TERM "$api_pid" "${next_pid:-}" 2>/dev/null || true
  wait "$api_pid" "${next_pid:-}" 2>/dev/null || true
}
trap cleanup TERM INT EXIT

python - "$FASTAPI_HOST" "$FASTAPI_PORT" <<'PY'
import sys
import time
import urllib.request

host, port = sys.argv[1], sys.argv[2]
url = f"http://{host}:{port}/docs"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit("FastAPI did not become ready within 120 seconds")
PY

HOSTNAME=0.0.0.0 PORT="$PORT" node "$NEXT_SERVER" &
next_pid=$!

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$next_pid" 2>/dev/null; do
  sleep 1
done

exit 1
