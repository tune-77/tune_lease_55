#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/require_api_access_key_secret.sh"
source "$ROOT_DIR/scripts/lib/require_public_tunnel_auth_secret.sh"

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
  --set-env-vars "FASTAPI_URL=$API_URL,PUBLIC_TUNNEL=1"
)

# API側のApiKeyAuthMiddlewareと同じ値をWeb側にも配線する（frontend/src/proxy.tsが
# process.env.API_ACCESS_KEYを読んでX-API-Keyを自動注入する）。公開Webだけがキーなしで
# デプロイされると全APIが503になるため、設定漏れはfail-closedで止める。
api_access_key_ref="$(require_api_access_key_secret "$PROJECT_ID" "Web")" || exit 1
tunnel_auth_ref="$(require_public_tunnel_auth_secret "$PROJECT_ID" "Web")" || exit 1
secrets_value="API_ACCESS_KEY=${api_access_key_ref},PUBLIC_TUNNEL_AUTH=${tunnel_auth_ref}"

# GitHub Actionsの knowledge-sync-health 監視専用シークレット（frontend/src/proxy.ts
# がBasic認証の代わりにX-Sync-Probe-Keyヘッダで要求する）。無くてもデプロイは止めない
# ―― 未設定ならproxy.tsがそのプローブパスを401にするだけで、安全側に倒れるため。
if gcloud secrets describe KNOWLEDGE_SYNC_PROBE_TOKEN --project "$PROJECT_ID" >/dev/null 2>&1; then
  secrets_value+=",KNOWLEDGE_SYNC_PROBE_TOKEN=KNOWLEDGE_SYNC_PROBE_TOKEN:latest"
else
  echo "Warning: Secret Manager secret KNOWLEDGE_SYNC_PROBE_TOKEN was not found; /api/system/knowledge-sync-health will 401 for the external monitor until it is created." >&2
fi
deploy_args+=(--set-secrets "$secrets_value")

# Web境界は公開（--allow-unauthenticated）。2026-09-06にIAM認証必須へ変更し、
# 2026-09-08にPR #983で一時公開へ戻したが、frontend/src/proxy.ts は Web境界の
# IAM有無に関係なく全ての未認証 /api/* リクエストへ上で配線した特権的な
# API_ACCESS_KEY を代理付与するため、単純にIAMを外すだけでは誰でもそのプロキシ
# 経由でAPIの鍵付きエンドポイント（コスト発生するchat・案件削除等）を叩けてしまい、
# 元のコスト急増インシデントと同種の穴を別URLで再現していた（Codexレビューで指摘、
# 一度巻き戻し）。今回は同じ穴を開けないため、上で配線した PUBLIC_TUNNEL=1 と
# PUBLIC_TUNNEL_AUTH をセットで必ず配線する。proxy.ts はこの2つが揃っている時、
# Basic認証（ID: lease）を通らない限り /api/* を含む全パスを401で拒否するため、
# Web境界のIAMを外しても未認証の第三者はAPIへ到達できない。
deploy_args+=(--allow-unauthenticated)

gcloud "${deploy_args[@]}"
