#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-tune-lease-55-api}"
SHORT_SHA="${SHORT_SHA:-$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo manual)}"
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
TIMEOUT="${TIMEOUT:-900}"
CONCURRENCY="${CONCURRENCY:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
# 既定は認証必須（安全側）。無認証公開が必要な場合のみ ALLOW_UNAUTHENTICATED=1 を明示する
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-0}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
DATABASE_SECRET_NAME="${DATABASE_SECRET_NAME:-DATABASE_URL}"
CLOUDSQL_INSTANCE="${CLOUDSQL_INSTANCE:-}"
CLOUDRUN_DATA_MODE="${CLOUDRUN_DATA_MODE:-demo}"
# 公開デモ（CLOUDRUN_DATA_MODE=demo）では既定で削除操作を無効化する（来場者による
# デモデータ破壊を防止）。本番データモードや明示指定で上書き可能。
if [[ "$CLOUDRUN_DATA_MODE" == "demo" ]]; then
  DEMO_READONLY="${DEMO_READONLY:-1}"
else
  DEMO_READONLY="${DEMO_READONLY:-0}"
fi

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

echo "Preparing Cloud Run bundle (CLOUDRUN_DATA_MODE=${CLOUDRUN_DATA_MODE})..."
CLOUDRUN_DATA_MODE="$CLOUDRUN_DATA_MODE" "$ROOT_DIR/scripts/package_cloud_run_bundle.sh"

echo "Running Cloud Run predeploy readiness checks..."
python3 "$ROOT_DIR/scripts/check_cloudrun_demo_readiness.py"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT_ID"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}/${SERVICE_NAME}:${SHORT_SHA}"

gcloud builds submit \
  --project "$PROJECT_ID" \
  --config "$ROOT_DIR/cloudbuild.api.yaml" \
  --substitutions _IMAGE_URI="$IMAGE_URI" \
  "$ROOT_DIR"

deploy_args=(
  run deploy "$SERVICE_NAME"
  --image "$IMAGE_URI"
  --project "$PROJECT_ID"
  --region "$REGION"
  --execution-environment gen2
  --port 8080
  --memory "$MEMORY"
  --cpu "$CPU"
  --timeout "$TIMEOUT"
  --concurrency "$CONCURRENCY"
  --min-instances "$MIN_INSTANCES"
  --max-instances "$MAX_INSTANCES"
  # ENABLE_OBSIDIAN_INDEXING/ENABLE_FEEDBACK_LOADING は30秒遅延のバックグラウンド
  # スレッドで動く（api/main.py起動処理参照）。ワーカーゾンビ化の既知リスクは
  # ローカル --reload 開発時に限られ、Cloud Runは--workers 1・--reloadなしの
  # 単一プロセスなので該当しない。ここを有効化しないとGCS Vault同期後も
  # ChromaDBが空のままになり、/api/chatのセマンティックRAGが常時キーワード
  # フォールバックに落ちる（根幹の知識ベースが機能しない状態が続く）。
  # ENABLE_GUNSHI_RAG は別経路（リクエスト同期でembeddingモデルを読む）で
  # 過去に共有プロセスの不安定化を招いた実績があるため、意図的に false のまま。
  --set-env-vars "DATA_DIR=/app/data,ENABLE_OBSIDIAN_INDEXING=true,ENABLE_FEEDBACK_LOADING=true,ENABLE_GUNSHI_RAG=false,OBSIDIAN_VAULT_PATH=/app/obsidian_vault,CLOUDRUN_BUNDLE_DIR=/app/.cloudrun_bundle,CLOUDRUN_DATA_MODE=${CLOUDRUN_DATA_MODE},DEMO_READONLY=${DEMO_READONLY},DB_PATH=/app/data/demo.db,USE_GCS_VAULT=true,GCS_VAULT_RESYNC_INTERVAL=3600,TZ=Asia/Tokyo"
)

if gcloud secrets describe GEMINI_API_KEY --project "$PROJECT_ID" >/dev/null 2>&1; then
  deploy_args+=(--set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest")
else
  echo "Warning: Secret Manager secret GEMINI_API_KEY was not found." >&2
fi

if gcloud secrets describe ESTAT_APP_ID --project "$PROJECT_ID" >/dev/null 2>&1; then
  deploy_args+=(--set-secrets "ESTAT_APP_ID=ESTAT_APP_ID:latest")
else
  echo "Warning: Secret Manager secret ESTAT_APP_ID was not found." >&2
fi

if [[ "$CLOUDRUN_DATA_MODE" == "demo" ]]; then
  echo "Demo mode: DATABASE_URL/Cloud SQL is intentionally not attached."
else
  if gcloud secrets describe "$DATABASE_SECRET_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
    deploy_args+=(--set-secrets "DATABASE_URL=${DATABASE_SECRET_NAME}:latest")
  else
    echo "Warning: Secret Manager secret ${DATABASE_SECRET_NAME} was not found. Cloud SQL will not be enabled." >&2
  fi

  if [[ -n "$CLOUDSQL_INSTANCE" ]]; then
    deploy_args+=(--add-cloudsql-instances "$CLOUDSQL_INSTANCE")
  fi
fi

if [[ -n "$SERVICE_ACCOUNT" ]]; then
  deploy_args+=(--service-account "$SERVICE_ACCOUNT")
fi

if [[ "$ALLOW_UNAUTHENTICATED" == "1" ]]; then
  deploy_args+=(--allow-unauthenticated)
else
  deploy_args+=(--no-allow-unauthenticated)
fi

gcloud "${deploy_args[@]}"
