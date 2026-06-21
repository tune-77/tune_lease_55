#!/bin/bash
# ChromaDB を GCS へ同期し、Cloud Run に新リビジョンを起動させる。
# reindex ジョブの後に呼び出す。失敗してもログのみで継続する。

set -euo pipefail

LOG_DIR="$HOME/Library/Logs/tunelease"
LOG_FILE="$LOG_DIR/chromadb_gcs_sync.log"
CHROMA_LOCAL="/Users/kobayashiisaoryou/clawd/tune_lease_55/api/chroma_db"
GCS_DEST="gs://tune-lease-55-data/chromadb/"
CLOUD_RUN_SERVICE="tune-lease-55"
CLOUD_RUN_REGION="asia-northeast1"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== ChromaDB GCS sync 開始 ==="

# --- GCS sync ---
if ! command -v gsutil &>/dev/null; then
    log "ERROR: gsutil が見つかりません。Google Cloud SDK をインストールしてください"
    exit 0  # reindex ジョブを止めないよう exit 0
fi

log "gsutil rsync 開始: $CHROMA_LOCAL -> $GCS_DEST"
if gsutil -m rsync -r "$CHROMA_LOCAL/" "$GCS_DEST" 2>&1 | tee -a "$LOG_FILE"; then
    log "gsutil rsync 完了"
else
    log "ERROR: gsutil rsync 失敗（Cloud Run 更新をスキップ）"
    exit 0
fi

# --- Cloud Run 新リビジョン起動 ---
if ! command -v gcloud &>/dev/null; then
    log "ERROR: gcloud が見つかりません。Cloud Run 更新をスキップ"
    exit 0
fi

TIMESTAMP="$(date +%Y%m%d%H%M%S)"
log "Cloud Run 新リビジョン起動: $CLOUD_RUN_SERVICE (CHROMADB_UPDATED_AT=$TIMESTAMP)"
if gcloud run services update "$CLOUD_RUN_SERVICE" \
    --region "$CLOUD_RUN_REGION" \
    --update-env-vars "CHROMADB_UPDATED_AT=$TIMESTAMP" 2>&1 | tee -a "$LOG_FILE"; then
    log "Cloud Run 更新完了"
else
    log "ERROR: Cloud Run 更新失敗（非致命的）"
fi

log "=== ChromaDB GCS sync 完了 ==="
