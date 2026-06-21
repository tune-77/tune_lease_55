#!/bin/bash
# Obsidian reindex + ChromaDB GCS sync の統合エントリポイント。
# launchd/com.tunelease.obsidian-reindex.plist から呼び出される。

set -uo pipefail

PROJECT_DIR="/Users/kobayashiisaoryou/clawd/tune_lease_55"
PYTHON="$PROJECT_DIR/.venv/bin/python"
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
