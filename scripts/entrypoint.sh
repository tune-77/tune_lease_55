#!/usr/bin/env bash
# Cloud Run エントリポイント: データ初期化 → アプリ起動
set -Eeuo pipefail

echo "[entrypoint] Cloud Run 初期化を開始..."

# git clone + GCS ChromaDB 同期（環境変数が設定されている場合のみ）
if [[ -n "${GITHUB_REPO:-}" || -n "${GCS_BUCKET:-}" ]]; then
    python3 /app/scripts/cloud_init.py || echo "[entrypoint] cloud_init.py で非致命的エラー（続行）"
fi

echo "[entrypoint] アプリケーションを起動..."
exec ./scripts/start_cloud_run.sh
