#!/usr/bin/env bash
# 公開Cloud Run（API/Web）は Secret Manager の API_ACCESS_KEY が無ければ
# 絶対にデプロイしない、というfail-closed方針の単一の実装。
#
# ローカル手動デプロイ (scripts/deploy_cloud_run.sh / deploy_cloud_run_api.sh /
# deploy_cloud_run_web.sh) と CI自動デプロイ (.github/workflows/deploy.yml) の
# 両方がここを経由することで、片方だけ認証強化して他方が取り残される乖離
# （2026-09、CIが未認証のままAPI/Webを再公開し続けCloud Runコスト急増を招いた
# 事故の原因）を構造的に防ぐ。
#
# 使い方:
#   secret_ref="$(require_api_access_key_secret "$PROJECT_ID" "API")" || exit 1
#   deploy_args+=(--set-secrets "API_ACCESS_KEY=${secret_ref}")
#
# 第2引数はエラーメッセージに出すサービスの呼び名（任意、既定 "service"）。
require_api_access_key_secret() {
  local project_id="$1"
  local label="${2:-service}"
  if gcloud secrets describe API_ACCESS_KEY --project "$project_id" >/dev/null 2>&1; then
    echo "API_ACCESS_KEY:latest"
    return 0
  fi
  echo "ERROR: Secret Manager secret API_ACCESS_KEY was not found. Refusing to deploy a public ${label} without an access key. Register it first: gcloud secrets create API_ACCESS_KEY --replication-policy=automatic --project ${project_id}" >&2
  return 1
}
