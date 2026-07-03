#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/.cloudrun_bundle"
DATA_OUT="$BUNDLE_DIR/data"
VAULT_OUT="$BUNDLE_DIR/obsidian_vault"
CLOUDRUN_DATA_MODE="${CLOUDRUN_DATA_MODE:-demo}"

SOURCE_VAULT="${OBSIDIAN_VAULT_PATH:-}"
if [[ -z "$SOURCE_VAULT" ]]; then
  for candidate in \
    "$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"; do
    if [[ -d "$candidate" ]]; then
      SOURCE_VAULT="$candidate"
      break
    fi
  done
fi

mkdir -p "$DATA_OUT" "$VAULT_OUT"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    cp -R "$src" "$dst"
  fi
}

if [[ "$CLOUDRUN_DATA_MODE" == "demo" ]]; then
  if [[ ! -f "$ROOT_DIR/data/demo.db" ]]; then
    echo "Demo mode requires data/demo.db. Run scripts/anonymize_db.py first." >&2
    exit 1
  fi
  cp "$ROOT_DIR/data/demo.db" "$DATA_OUT/demo.db"
  # Some graph/stat modules still read DATA_DIR/lease_data.db directly.
  # In demo mode, make that legacy path point at the anonymized demo DB.
  cp "$ROOT_DIR/data/demo.db" "$DATA_OUT/lease_data.db"
else
  copy_if_exists "$ROOT_DIR/data/lease_data.db" "$DATA_OUT/"
  copy_if_exists "$ROOT_DIR/data/demo.db" "$DATA_OUT/"
fi
copy_if_exists "$ROOT_DIR/data/screening_db.sqlite" "$DATA_OUT/"
copy_if_exists "$ROOT_DIR/data/user_personal_memory.md" "$DATA_OUT/"
copy_if_exists "$ROOT_DIR/reports/obsidian_daily_intelligence_latest.json" "$BUNDLE_DIR/"

# 紫苑の記憶索引を毎回ビルドして同梱する（無いと Cloud Run で想起メモが空になる）。
# 改訂宣言（data/shion_memory_revisions.jsonl）はビルド内で再適用される。
# 鮮度（last_used_at / stale）は使用ログがあれば反映する（無くても致命ではないので || true）。
echo "Building Shion memory index for bundle..."
python3 "$ROOT_DIR/scripts/build_shion_memory_index.py"
python3 "$ROOT_DIR/scripts/update_shion_memory_freshness.py" || true
cp "$ROOT_DIR/data/shion_memory_index.json" "$DATA_OUT/"

validate_past_cases() {
  local db_path="$1"
  if [[ ! -f "$db_path" ]]; then
    echo "DB file not found: $db_path" >&2
    exit 1
  fi
  python3 - "$db_path" <<'PY'
import sqlite3
import sys

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
try:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
finally:
    conn.close()
if "past_cases" not in tables:
    raise SystemExit(f"past_cases table missing in {db_path}")
PY
}

validate_past_cases "$DATA_OUT/lease_data.db"
if [[ -f "$DATA_OUT/demo.db" ]]; then
  validate_past_cases "$DATA_OUT/demo.db"
fi

if [[ -z "$SOURCE_VAULT" ]]; then
  echo "iCloud 上の Obsidian Vault が見つかりません。OBSIDIAN_VAULT_PATH を設定してください。" >&2
  exit 1
fi

for rel in \
  "Projects/tune_lease_55" \
  "05-クリップ_記事/リースニュース" \
  "リース知識"; do
  src="$SOURCE_VAULT/$rel"
  dst="$VAULT_OUT/$rel"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
  fi
done

echo "Cloud Run bundle prepared at $BUNDLE_DIR"
