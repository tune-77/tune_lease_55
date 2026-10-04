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
REINDEX_MAX_ATTEMPTS="${REINDEX_MAX_ATTEMPTS:-2}"
REINDEX_RETRY_DELAY_SECONDS="${REINDEX_RETRY_DELAY_SECONDS:-60}"
REINDEX_ATTEMPT=1
while true; do
    "$PYTHON" -m mobile_app.rag_daily_maintenance
    REINDEX_EXIT=$?
    if [ $REINDEX_EXIT -ne 75 ] || [ $REINDEX_ATTEMPT -ge "$REINDEX_MAX_ATTEMPTS" ]; then
        break
    fi
    echo "[run_obsidian_reindex] writer競合のため ${REINDEX_RETRY_DELAY_SECONDS} 秒後に再試行します (${REINDEX_ATTEMPT}/${REINDEX_MAX_ATTEMPTS})"
    sleep "$REINDEX_RETRY_DELAY_SECONDS"
    REINDEX_ATTEMPT=$((REINDEX_ATTEMPT + 1))
done

if [ $REINDEX_EXIT -eq 75 ]; then
    echo "[run_obsidian_reindex] ChromaDB writer が使用中のため reindex を延期しました。GCS sync もスキップします"
elif [ $REINDEX_EXIT -ne 0 ]; then
    echo "[run_obsidian_reindex] reindex が失敗しました (exit=$REINDEX_EXIT)。GCS sync は実行します"
fi

# --- ChromaDB GCS sync (失敗してもreindexの結果を変えない) ---
if [ $REINDEX_EXIT -ne 75 ]; then
    bash "$SYNC_SCRIPT" || true
fi

# --- Retrieval graph index (router/index/edges) 再構築 ---
# Cloud Runデプロイ時(package_cloud_run_bundle.sh)のスナップショットのままだと
# 新規ノートがobsidian_bridge.pyの_graph_route_candidates()に反映されないため、
# 夜間reindexのタイミングでdata/obsidian_retrieval_graph.jsonも更新する。
# 失敗してもreindexの結果には影響させない。
"$PYTHON" -m scripts.build_obsidian_retrieval_graph || echo "[run_obsidian_reindex] retrieval graph index の再構築に失敗しました"

exit $REINDEX_EXIT
