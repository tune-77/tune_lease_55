#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# 公開可否は接続先APIの実際のデータモードから決める。未設定は実データ扱い。
API_DESCRIPTION="$(gcloud run services describe "$API_SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" --format=json)"
API_DATA_MODE="$(python3 -c 'import json,sys; d=json.load(sys.stdin); env=d["spec"]["template"]["spec"]["containers"][0].get("env", []); print(next((e.get("value", "production") for e in env if e.get("name") == "CLOUDRUN_DATA_MODE"), "production"))' <<< "$API_DESCRIPTION")"

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

# Web→APIの共有キーを配線する。接続先が実データ用なら未登録で停止する。
# デモだけはキー未設定でも公開できる。
if gcloud secrets describe API_ACCESS_KEY --project "$PROJECT_ID" >/dev/null 2>&1; then
  deploy_args+=(--set-secrets "API_ACCESS_KEY=API_ACCESS_KEY:latest")
elif [[ "$API_DATA_MODE" != "demo" ]]; then
  echo "ERROR: API_ACCESS_KEY is required for a production Web deployment." >&2
  exit 1
else
  echo "Warning: Secret Manager secret API_ACCESS_KEY was not found. Web will not send X-API-Key." >&2
fi

if [[ "$API_DATA_MODE" == "demo" ]]; then
  deploy_args+=(--allow-unauthenticated)
else
  # APIキーの自動付与より前に、Cloud Run IAMで利用者を認証する。
  deploy_args+=(--no-allow-unauthenticated --invoker-iam-check)
fi

gcloud "${deploy_args[@]}"
