#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/require_api_access_key_secret.sh"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-tune-lease-55-web}"
API_SERVICE_NAME="${API_SERVICE_NAME:-tune-lease-55-api}"
SHORT_SHA="${SHORT_SHA:-$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo manual)}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-1}"
TIMEOUT="${TIMEOUT:-900}"
CONCURRENCY="${CONCURRENCY:-80}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

API_URL="$(gcloud run services describe "$API_SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')"
if [[ -z "$API_URL" ]]; then
  echo "API service URL could not be resolved: $API_SERVICE_NAME" >&2
  exit 1
fi

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT_ID"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}/${SERVICE_NAME}:${SHORT_SHA}"

gcloud builds submit \
  --project "$PROJECT_ID" \
  --config "$ROOT_DIR/cloudbuild.web.yaml" \
  --substitutions _IMAGE_URI="$IMAGE_URI",_FASTAPI_URL="$API_URL" \
  --suppress-logs \
  "$ROOT_DIR/frontend"

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
  --set-env-vars "FASTAPI_URL=$API_URL"
)

# API側のApiKeyAuthMiddlewareと同じ値をWeb側にも配線する（frontend/src/proxy.tsが
# process.env.API_ACCESS_KEYを読んでX-API-Keyを自動注入する）。公開Webだけがキーなしで
# デプロイされると全APIが503になるため、設定漏れはfail-closedで止める。
api_access_key_ref="$(require_api_access_key_secret "$PROJECT_ID" "Web")" || exit 1
deploy_args+=(--set-secrets "API_ACCESS_KEY=${api_access_key_ref}")

# Web境界はIAM認証必須（--no-allow-unauthenticated --invoker-iam-check）。
# 2026-09-08、UX上の理由から一時 --allow-unauthenticated に変更したが、
# frontend/src/proxy.ts は Web境界のIAM有無に関係なく全ての未認証 /api/* リクエストへ
# 上で配線した特権的な API_ACCESS_KEY を代理付与するため、Webを公開にすると誰でも
# そのプロキシ経由でAPIの鍵付きエンドポイント（コスト発生するchat・案件削除等）を
# 叩けてしまい、元のコスト急増インシデントと同種の穴を別URLで再現していた
# （Codexレビューで指摘、PR #983 で一度マージされたが直後に巻き戻し）。
deploy_args+=(--no-allow-unauthenticated --invoker-iam-check)

gcloud "${deploy_args[@]}"
