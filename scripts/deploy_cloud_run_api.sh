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
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-1}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
DATABASE_SECRET_NAME="${DATABASE_SECRET_NAME:-DATABASE_URL}"
CLOUDSQL_INSTANCE="${CLOUDSQL_INSTANCE:-}"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

"$ROOT_DIR/scripts/package_cloud_run_bundle.sh"

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
  --set-env-vars "DATA_DIR=/app/data,ENABLE_OBSIDIAN_INDEXING=false,ENABLE_FEEDBACK_LOADING=false,ENABLE_GUNSHI_RAG=false,OBSIDIAN_VAULT_PATH=/app/obsidian_vault,CLOUDRUN_BUNDLE_DIR=/app/.cloudrun_bundle,USE_GCS_VAULT=true"
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

if gcloud secrets describe "$DATABASE_SECRET_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
  deploy_args+=(--set-secrets "DATABASE_URL=${DATABASE_SECRET_NAME}:latest")
else
  echo "Warning: Secret Manager secret ${DATABASE_SECRET_NAME} was not found. Cloud SQL will not be enabled." >&2
fi

if [[ -n "$CLOUDSQL_INSTANCE" ]]; then
  deploy_args+=(--add-cloudsql-instances "$CLOUDSQL_INSTANCE")
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
