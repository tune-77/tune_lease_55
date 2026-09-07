#!/usr/bin/env bash
# 高頻度push（deploy.yml が api/**・frontend/** 変更のたびにビルドする）で
# Artifact Registry (cloud-run-source-deploy) に溜まり続けるコンテナイメージの
# ストレージ課金を止める。scripts/cleanup_cloud_run_revisions.py はCloud Run
# の「リビジョン」を消すだけで、Artifact Registry の「イメージ」自体は消さない
# ため、これは別問題として対処する。
#
# 継続監視ではなく、GCPのArtifact Registryクリーンアップポリシー（サーバー側で
# 自動適用され続ける宣言的な設定）を一度設定するだけの運用（このリポジトリの
# scripts/demo_warm.sh 等と同じく、gcloud認証を持つ運用者が手元で実行する想定）。
#
# 安全策:
#   - KEEP_COUNT（既定15）より新しいバージョンは常に保持（直近デプロイ・
#     ロールバック候補を誤って消さないため）
#   - それより古いバージョンのうち、DELETE_AFTER_DAYS（既定30日）を超えたものだけ削除
#
# 使い方:
#   PROJECT_ID=... ./scripts/setup_artifact_registry_cleanup_policy.sh
#   KEEP_COUNT=20 DELETE_AFTER_DAYS=14 PROJECT_ID=... ./scripts/setup_artifact_registry_cleanup_policy.sh
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-northeast1}"
REPOSITORY="${REPOSITORY:-cloud-run-source-deploy}"
KEEP_COUNT="${KEEP_COUNT:-15}"
DELETE_AFTER_DAYS="${DELETE_AFTER_DAYS:-30}"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

DELETE_AFTER_SECONDS=$(( DELETE_AFTER_DAYS * 86400 ))
POLICY_FILE="$(mktemp)"
trap 'rm -f "$POLICY_FILE"' EXIT

cat > "$POLICY_FILE" <<JSON
[
  {
    "name": "keep-recent-versions",
    "action": {"type": "Keep"},
    "mostRecentVersions": {"keepCount": ${KEEP_COUNT}}
  },
  {
    "name": "delete-old-versions",
    "action": {"type": "Delete"},
    "condition": {"olderThan": "${DELETE_AFTER_SECONDS}s"}
  }
]
JSON

echo "Setting cleanup policy on ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}"
echo "  keep most recent ${KEEP_COUNT} versions per image, delete others older than ${DELETE_AFTER_DAYS} days"

gcloud artifacts repositories set-cleanup-policies "$REPOSITORY" \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --policy "$POLICY_FILE"

echo "Done. Policy is enforced server-side by Artifact Registry going forward (no recurring job needed)."
echo "Verify anytime with: gcloud artifacts repositories describe ${REPOSITORY} --project ${PROJECT_ID} --location ${REGION} --format='value(cleanupPolicies)'"
