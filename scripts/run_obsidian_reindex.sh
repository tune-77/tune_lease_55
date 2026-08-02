#!/bin/bash
# Obsidian reindex + ChromaDB GCS sync の統合エントリポイント。
# launchd/com.tunelease.obsidian-reindex.plist から呼び出される。

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# launchd 側の PYTHON_BIN と揃える（未設定なら venv）。
# plist に PYTHON_BIN を出しておくと check_obsidian_ops_consistency.py が実行系を検出できる。
PYTHON="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
SYNC_SCRIPT="$PROJECT_DIR/scripts/sync_chromadb_to_gcs.sh"

# --- Obsidian reindex ---
cd "$PROJECT_DIR"
"$PYTHON" -m mobile_app.rag_daily_maintenance
REINDEX_EXIT=$?

if [ $REINDEX_EXIT -ne 0 ]; then
    echo "[run_obsidian_reindex] reindex が失敗しました (exit=$REINDEX_EXIT)。GCS sync は実行します"
fi

# --- ChromaDB GCS sync (失敗してもreindexの結果を変えない) ---
bash "$SYNC_SCRIPT" || true

exit $REINDEX_EXIT
