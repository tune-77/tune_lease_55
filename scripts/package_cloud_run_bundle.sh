#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/.cloudrun_bundle"
DATA_OUT="$BUNDLE_DIR/data"
VAULT_OUT="$BUNDLE_DIR/obsidian_vault"
CLOUDRUN_DATA_MODE="${CLOUDRUN_DATA_MODE:-demo}"

if [[ -d "$BUNDLE_DIR" ]]; then
  chmod -R u+w "$BUNDLE_DIR" 2>/dev/null || true
  find "$BUNDLE_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi

# Vault パスは runtime_paths が唯一の解決窓口。
# env の優先順（OBSIDIAN_VAULT_PATH → OBSIDIAN_VAULT）も既定パスもそちらが決めるので、
# ここに候補を直書きしない（直書きすると Python 側の解決結果とずれ、別 Vault をバンドルしうる）。
source "$ROOT_DIR/scripts/resolve_obsidian_vault.sh"
SOURCE_VAULT="$(resolve_obsidian_vault)"
if [[ -n "$SOURCE_VAULT" && ! -d "$SOURCE_VAULT" ]]; then
  SOURCE_VAULT=""
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

REPORTS_OUT="$BUNDLE_DIR/reports"
mkdir -p "$REPORTS_OUT"
copy_if_exists "$ROOT_DIR/reports/latest.json" "$REPORTS_OUT/"
latest_improvement="$(find "$ROOT_DIR/reports" -maxdepth 1 -name 'improvement_report_*.json' -type f | sort | tail -n 1 || true)"
if [[ -n "$latest_improvement" ]]; then
  cp "$latest_improvement" "$REPORTS_OUT/"
fi
copy_if_exists "$ROOT_DIR/reports/recursive_self_improvement_latest.json" "$REPORTS_OUT/"
latest_recursive="$(find "$ROOT_DIR/reports" -maxdepth 1 -name 'recursive_self_improvement_*.json' -type f | sort | tail -n 1 || true)"
if [[ -n "$latest_recursive" ]]; then
  cp "$latest_recursive" "$REPORTS_OUT/"
fi
copy_if_exists "$ROOT_DIR/reports/shion_pm_quality_latest.json" "$REPORTS_OUT/"

# Memory Review Inbox / Memory Engineering は Cloud Run では reports/ と
# data/*.jsonl が送信対象外になるため、レビューに必要な読み取り専用材料だけ
# bundle data に明示同梱する。start_api_cloud_run.sh が /app/data へseedする。
for review_source in \
  judgment_materials_preview.jsonl \
  autoresearch_judgment_asset_candidates.jsonl \
  reflection_action_candidates.jsonl \
  prediction_error_update_candidates.jsonl \
  obsidian_memory_insight_candidates.jsonl \
  memory_review_inbox_state.json; do
  copy_if_exists "$ROOT_DIR/data/$review_source" "$DATA_OUT/"
done

# 紫苑の記憶索引を毎回ビルドして同梱する（無いと Cloud Run で想起メモが空になる）。
# 改訂宣言（data/shion_memory_revisions.jsonl）はビルド内で再適用される。
# 鮮度（last_used_at / stale）は使用ログがあれば反映する（無くても致命ではないので || true）。
echo "Building Shion memory index for bundle..."
python3 "$ROOT_DIR/scripts/build_shion_memory_index.py"
python3 "$ROOT_DIR/scripts/update_shion_memory_freshness.py" || true
if [[ "$CLOUDRUN_DATA_MODE" == "demo" ]]; then
  # 公開デモには対話・内省・private の記憶を載せない（--demo-safe で除外）
  python3 "$ROOT_DIR/scripts/build_shion_memory_index.py" \
    --output "$DATA_OUT/shion_memory_index.json" --demo-safe
  python3 "$ROOT_DIR/scripts/update_shion_memory_freshness.py" \
    --index "$DATA_OUT/shion_memory_index.json" || true
else
  cp "$ROOT_DIR/data/shion_memory_index.json" "$DATA_OUT/"
fi

# Obsidian 検索の前段ルータを Cloud Run に同梱する。
# start_api_cloud_run.sh が .cloudrun_bundle/data を DATA_DIR へ seed し、
# mobile_app/obsidian_bridge.py は DATA_DIR/obsidian_retrieval_graph.json を読む。
# reports/ は Docker イメージ外なので、レポートは bundle 内だけに出す。
echo "Building Obsidian retrieval graph for bundle..."
python3 "$ROOT_DIR/scripts/build_obsidian_retrieval_graph.py" \
  --output-json "$DATA_OUT/obsidian_retrieval_graph.json" \
  --output-md "$REPORTS_OUT/obsidian_retrieval_graph_latest.md"

echo "Building Memory Engineering report for bundle..."
python3 "$ROOT_DIR/scripts/build_memory_engineering_report.py" \
  --canonical-preview "$ROOT_DIR/data/canonical_judgment_rules_preview.json" \
  --canonical-active "$ROOT_DIR/data/canonical_judgment_rules.json" \
  --memory-index "$DATA_OUT/shion_memory_index.json" \
  --memory-usage "$ROOT_DIR/data/shion_memory_usage_log.jsonl" \
  --contradictions "$ROOT_DIR/reports/shion_memory_contradictions_latest.json" \
  --retrieval-graph "$DATA_OUT/obsidian_retrieval_graph.json" \
  --memory-review-state "$DATA_OUT/memory_review_inbox_state.json" \
  --output-json "$DATA_OUT/memory_engineering_latest.json" \
  --output-md "$REPORTS_OUT/memory_engineering_latest.md"

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
  "05-クリップ_記事/業界リスクニュース" \
  "05-クリップ_記事/リースニュース" \
  "リース知識" \
  "lease-wiki-vault"; do
  src="$SOURCE_VAULT/$rel"
  dst="$VAULT_OUT/$rel"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
    echo "📚 $rel をバンドルに追加"
  fi
done

echo "Cloud Run bundle prepared at $BUNDLE_DIR"
