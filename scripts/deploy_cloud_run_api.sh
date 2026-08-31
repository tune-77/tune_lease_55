#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-northeast1}"
SERVICE_NAME="${SERVICE_NAME:-tune-lease-55-api}"
API_IGNORE_FILE="${API_IGNORE_FILE:-$ROOT_DIR/.gcloudignore.api}"
SHORT_SHA="${SHORT_SHA:-$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo manual)}"
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
TIMEOUT="${TIMEOUT:-900}"
CONCURRENCY="${CONCURRENCY:-1}"
# 既定は0（アイドル課金なし）。発表・審査前は scripts/demo_warm.sh on で温める
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
CLOUDRUN_DATA_MODE="${CLOUDRUN_DATA_MODE:-production}"
# 紫苑記憶のハイブリッド想起（キーワード＋埋め込み）。埋め込みモデルは
# ENABLE_OBSIDIAN_INDEXING の経路で既にロードされるため追加コストは小さい。
# コレクションは初回起動時にバックグラウンド自動構築される（api/shion_memory_vector.py）。
SHION_MEMORY_HYBRID="${SHION_MEMORY_HYBRID:-1}"
# 非demoモード（実データ）では API キー検証を既定で必須にする（fail-closed）。
# demoモードのみ従来通り既定で無効（ローカル検証・公開デモ体験を壊さない）。
if [[ "$CLOUDRUN_DATA_MODE" == "demo" ]]; then
  REQUIRE_API_ACCESS_KEY="${REQUIRE_API_ACCESS_KEY:-0}"
else
  REQUIRE_API_ACCESS_KEY="${REQUIRE_API_ACCESS_KEY:-1}"
fi
# lease_data.db をGCSへ定期スナップショットする間隔（秒）。demoモードでは
# api/cloudrun_db_snapshot.py が自動的に無効化するため、この値は非demoモードのみ
# 効果を持つ（REV-310）。
GCS_DB_SNAPSHOT_INTERVAL_SECONDS="${GCS_DB_SNAPSHOT_INTERVAL_SECONDS:-300}"
# 紫苑（軍師AI/ADKエージェント）がVertex AI Searchを能動的にツール呼び出しできるようにする
# opt-inフラグ。課金対象の外部APIのため既定オフだが、本番でも使う判断が出たため有効化する
# （api/shion_vertex_tools.py参照）。
SHION_ENABLE_VERTEX_TOOLS="${SHION_ENABLE_VERTEX_TOOLS:-true}"
# 通常チャットのVertex Answer APIでGoogle検索グラウンディングを使うか。コード側の既定もtrueだが、
# 課金を伴う挙動なので本番の環境変数としても明示しておく（api/vertex_agent_search.py参照）。
VERTEX_GOOGLE_SEARCH_GROUNDING_ENABLED="${VERTEX_GOOGLE_SEARCH_GROUNDING_ENABLED:-true}"
# Cloud Run demo でも登録・削除を通常操作として試せるよう、既定では
# DEMO_READONLY を解除する。公開審査などで削除を塞ぎたい時だけ
# DEMO_READONLY=1 を明示して上書きする。
if [[ "$CLOUDRUN_DATA_MODE" == "demo" ]]; then
  DEMO_READONLY="${DEMO_READONLY:-0}"
else
  DEMO_READONLY="${DEMO_READONLY:-0}"
fi

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi
if [[ ! -f "$API_IGNORE_FILE" ]]; then
  echo "API ignore file is required: $API_IGNORE_FILE" >&2
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
  --ignore-file "$API_IGNORE_FILE" \
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
  # ENABLE_OBSIDIAN_INDEXING/ENABLE_FEEDBACK_LOADING は30秒遅延のバックグラウンド
  # スレッドで動く（api/main.py起動処理参照）。ワーカーゾンビ化の既知リスクは
  # ローカル --reload 開発時に限られ、Cloud Runは--workers 1・--reloadなしの
  # 単一プロセスなので該当しない。ここを有効化しないとGCS Vault同期後も
  # ChromaDBが空のままになり、/api/chatのセマンティックRAGが常時キーワード
  # フォールバックに落ちる（根幹の知識ベースが機能しない状態が続く）。
  # ENABLE_GUNSHI_RAG は別経路（リクエスト同期でembeddingモデルを読む）で
  # 過去に共有プロセスの不安定化を招いた実績があるため、意図的に false のまま。
  --set-env-vars "DATA_DIR=/app/data,ENABLE_OBSIDIAN_INDEXING=true,ENABLE_FEEDBACK_LOADING=true,ENABLE_GUNSHI_RAG=false,OBSIDIAN_VAULT_PATH=/app/obsidian_vault,CLOUDRUN_BUNDLE_DIR=/app/.cloudrun_bundle,CLOUDRUN_DATA_MODE=${CLOUDRUN_DATA_MODE},DEMO_READONLY=${DEMO_READONLY},DB_PATH=/app/data/lease_data.db,USE_GCS_VAULT=true,GCS_VAULT_RESYNC_INTERVAL=3600,SHION_MEMORY_HYBRID=${SHION_MEMORY_HYBRID},REQUIRE_API_ACCESS_KEY=${REQUIRE_API_ACCESS_KEY},GCS_DB_SNAPSHOT_INTERVAL_SECONDS=${GCS_DB_SNAPSHOT_INTERVAL_SECONDS},SHION_ENABLE_VERTEX_TOOLS=${SHION_ENABLE_VERTEX_TOOLS},VERTEX_GOOGLE_SEARCH_GROUNDING_ENABLED=${VERTEX_GOOGLE_SEARCH_GROUNDING_ENABLED},TZ=Asia/Tokyo"
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

if gcloud secrets describe API_ACCESS_KEY --project "$PROJECT_ID" >/dev/null 2>&1; then
  deploy_args+=(--set-secrets "API_ACCESS_KEY=API_ACCESS_KEY:latest")
elif [[ "$CLOUDRUN_DATA_MODE" != "demo" ]]; then
  echo "ERROR: Secret Manager secret API_ACCESS_KEY was not found. Refusing to deploy non-demo (real data) without an access key. Register it first: gcloud secrets create API_ACCESS_KEY --replication-policy=automatic --project ${PROJECT_ID}" >&2
  exit 1
else
  echo "Warning: Secret Manager secret API_ACCESS_KEY was not found. Demo mode stays unauthenticated at the app layer." >&2
fi

echo "SQLite/GCS mode: DATABASE_URL/Cloud SQL is intentionally not attached."
deploy_args+=(--remove-secrets=DATABASE_URL --clear-cloudsql-instances)

if [[ -n "$SERVICE_ACCOUNT" ]]; then
  deploy_args+=(--service-account "$SERVICE_ACCOUNT")
fi

deploy_args+=(--allow-unauthenticated)

gcloud "${deploy_args[@]}"
