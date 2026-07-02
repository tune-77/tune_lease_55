#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${PORT:-8080}"
FASTAPI_HOST="${FASTAPI_HOST:-0.0.0.0}"
FASTAPI_PORT="${FASTAPI_PORT:-$PORT}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
OBSIDIAN_VAULT_PATH="${OBSIDIAN_VAULT_PATH:-}"
BUNDLE_ROOT="${CLOUDRUN_BUNDLE_DIR:-$(pwd)/.cloudrun_bundle}"

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
      echo "[start_api_cloud_run] demo DB restored from bundle: $DATA_DIR/$db_name"
    fi
  done
fi
if [[ -n "$OBSIDIAN_VAULT_PATH" ]]; then
  seed_dir "$BUNDLE_ROOT/obsidian_vault" "$OBSIDIAN_VAULT_PATH"
  # vault はシステムプロンプト更新等で書き込みが発生するため権限を回復する
  chmod -R u+w "$OBSIDIAN_VAULT_PATH" 2>/dev/null || true
fi

exec python -m uvicorn api.main:app \
  --host "$FASTAPI_HOST" \
  --port "$FASTAPI_PORT" \
  --workers 1
