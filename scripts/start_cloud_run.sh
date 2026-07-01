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
    # -n: 既存ファイルは上書きしない（git cloneで配置済みのファイルを優先）
    cp -Rn "$src"/. "$dst"/ 2>/dev/null || true
  fi
}

seed_dir "$BUNDLE_ROOT/data" "$DATA_DIR"
if [[ "${CLOUDRUN_DATA_MODE:-}" == "demo" ]]; then
  for db_name in demo.db lease_data.db; do
    if [[ -f "$BUNDLE_ROOT/data/$db_name" ]]; then
      cp -f "$BUNDLE_ROOT/data/$db_name" "$DATA_DIR/$db_name"
      echo "[start_cloud_run] demo DB restored from bundle: $DATA_DIR/$db_name"
    fi
  done
fi
if [[ -n "$OBSIDIAN_VAULT_PATH" ]]; then
  seed_dir "$BUNDLE_ROOT/obsidian_vault" "$OBSIDIAN_VAULT_PATH"
  # vault はシステムプロンプト更新等で書き込みが発生するため権限を回復する
  chmod -R u+w "$OBSIDIAN_VAULT_PATH" 2>/dev/null || true
fi

python -m uvicorn api.main:app \
  --host "$FASTAPI_HOST" \
  --port "$FASTAPI_PORT" \
  --workers 1 &
api_pid=$!

shutdown_requested=0

cleanup() {
  kill -TERM "$api_pid" "${next_pid:-}" 2>/dev/null || true
  wait "$api_pid" "${next_pid:-}" 2>/dev/null || true
}
on_signal() {
  shutdown_requested=1
  cleanup
}
trap on_signal TERM INT
trap cleanup EXIT

READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-240}"
python - "$FASTAPI_HOST" "$FASTAPI_PORT" "$READY_TIMEOUT_SECONDS" <<'PY'
import sys
import time
import urllib.request

host, port, timeout_seconds = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = f"http://{host}:{port}/docs"
for _ in range(timeout_seconds):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit(f"FastAPI did not become ready within {timeout_seconds} seconds")
PY

HOSTNAME=0.0.0.0 PORT="$PORT" node "$NEXT_SERVER" &
next_pid=$!

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$next_pid" 2>/dev/null; do
  sleep 1
done

if [[ "$shutdown_requested" == "1" ]]; then
  echo "[start_cloud_run] shutdown signal received; exiting cleanly"
  exit 0
fi

echo "[start_cloud_run] a child process exited unexpectedly" >&2
exit 1
