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
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-1}"

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
  --project "$PROJECT_ID"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}/${SERVICE_NAME}:${SHORT_SHA}"

gcloud builds submit \
  --project "$PROJECT_ID" \
  --config "$ROOT_DIR/cloudbuild.web.yaml" \
  --substitutions _IMAGE_URI="$IMAGE_URI",_FASTAPI_URL="$API_URL" \
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
)

if [[ "$ALLOW_UNAUTHENTICATED" == "1" ]]; then
  deploy_args+=(--allow-unauthenticated)
else
  deploy_args+=(--no-allow-unauthenticated)
fi

gcloud "${deploy_args[@]}"
