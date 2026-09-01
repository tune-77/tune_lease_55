#!/usr/bin/env bash
# Combined (single-container) deploy: FastAPI + Next.js in one Cloud Run service.
# Targets the existing tune-lease-55 service.
# For split-service deploys, call deploy_cloud_run_api.sh / deploy_cloud_run_web.sh directly.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-tune-lease-55}"
SHORT_SHA="${SHORT_SHA:-$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo manual)}"
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
TIMEOUT="${TIMEOUT:-900}"
CONCURRENCY="${CONCURRENCY:-1}"
# 既定は0（アイドル課金なし）。発表・審査前は scripts/demo_warm.sh on で温める
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
CLOUDRUN_DATA_MODE="${CLOUDRUN_DATA_MODE:-demo}"

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
  --config "$ROOT_DIR/cloudbuild.yaml" \
  --substitutions _IMAGE_URI="$IMAGE_URI" \
  --suppress-logs \
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
  --set-env-vars "DATA_DIR=/app/data,ENABLE_OBSIDIAN_INDEXING=true,ENABLE_FEEDBACK_LOADING=true,ENABLE_GUNSHI_RAG=false,OBSIDIAN_VAULT_PATH=/app/obsidian_vault,CLOUDRUN_BUNDLE_DIR=/app/.cloudrun_bundle,CLOUDRUN_DATA_MODE=${CLOUDRUN_DATA_MODE},DB_PATH=/app/data/lease_data.db,GCS_BUCKET=tune-lease-55-data,GITHUB_REPO=git@github.com:tune-77/tune_lease_55.git,DATA_GIT_DIR=/app/data-git,USE_GCS_VAULT=true,GCS_VAULT_RESYNC_INTERVAL=3600"
)

has_replacement_secrets=0

if gcloud secrets describe GEMINI_API_KEY --project "$PROJECT_ID" >/dev/null 2>&1; then
  deploy_args+=(--set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest")
  has_replacement_secrets=1
else
  echo "Warning: Secret Manager secret GEMINI_API_KEY was not found." >&2
fi

if gcloud secrets describe ESTAT_APP_ID --project "$PROJECT_ID" >/dev/null 2>&1; then
  deploy_args+=(--set-secrets "ESTAT_APP_ID=ESTAT_APP_ID:latest")
  has_replacement_secrets=1
else
  echo "Warning: Secret Manager secret ESTAT_APP_ID was not found." >&2
fi

if (( has_replacement_secrets == 0 )); then
  deploy_args+=(--clear-secrets)
fi

echo "SQLite/GCS mode: DATABASE_URL/Cloud SQL is intentionally not attached."
deploy_args+=(--clear-cloudsql-instances)

if [[ -n "$SERVICE_ACCOUNT" ]]; then
  deploy_args+=(--service-account "$SERVICE_ACCOUNT")
fi

deploy_args+=(--allow-unauthenticated)

gcloud "${deploy_args[@]}"
