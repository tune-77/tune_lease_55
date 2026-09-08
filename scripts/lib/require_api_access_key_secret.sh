#!/usr/bin/env bash
# 公開Cloud Run（API/Web）に Secret Manager の API_ACCESS_KEY を必ず配線する、
# fail-closed方針の単一の実装。
#
# ローカル手動デプロイ (scripts/deploy_cloud_run.sh / deploy_cloud_run_api.sh /
# deploy_cloud_run_web.sh) と CI自動デプロイ (.github/workflows/deploy.yml) の
# 両方がここを経由することで、片方だけ認証強化して他方が取り残される乖離
# （2026-09、CIが未認証のままAPI/Webを再公開し続けCloud Runコスト急増を招いた
# 事故の原因）を構造的に防ぐ。
#
# ── fail-closedの実体はどこにあるか ──────────────────────────────────
# 「キー無しでは公開デプロイさせない」保証の実体は、呼び出し側が必ず付ける
#   gcloud run deploy --set-secrets "API_ACCESS_KEY=API_ACCESS_KEY:latest"
# 側にある。シークレットを解決できなければデプロイ自体が失敗するため、
# この関数が通っただけでキー無しデプロイが成立することはない。
#
# 逆に、この関数の `gcloud secrets describe` は本番操作とは別の権限
# (secretmanager.secrets.get) を要求する。ここをデプロイ可否のゲートにすると
# 「本来デプロイできるのに事前チェックだけが落ちて永久に進めない」状態が
# 起こり得る（2026-09-07、実際にこれでCIが停止した）。
# そのため本関数は「確実に存在しない(NOT_FOUND)」時だけ止め、それ以外の失敗
# （権限不足など）は理由を出して先へ進め、可否は実デプロイに判定させる。
#
# 使い方:
#   secret_ref="$(require_api_access_key_secret "$PROJECT_ID" "API")" || exit 1
#   deploy_args+=(--set-secrets "API_ACCESS_KEY=${secret_ref}")
#
# 第2引数はエラーメッセージに出すサービスの呼び名（任意、既定 "service"）。
require_api_access_key_secret() {
  local project_id="$1"
  local label="${2:-service}"
  local secret_ref="API_ACCESS_KEY:latest"
  local gcloud_error
  local status

  # gcloudの実エラーを捨てない。捨てると 403(権限不足) / 404(存在しない) /
  # API無効 / プロジェクト指定ミス がすべて同じ「not found」に潰れ、原因の
  # 特定が不可能になる（2026-09-07、この潰れたメッセージのせいでIAMロールの
  # 当て推量を繰り返し、1回17分のCIを何度も空費した）。
  # `2>&1 >/dev/null` はstderrだけを捕捉する定型。`local`と代入を分けているのは
  # `local x="$(cmd)"` だと $? が local の終了ステータスになり、cmd の失敗を
  # 取りこぼすため。
  gcloud_error="$(gcloud secrets describe API_ACCESS_KEY --project "$project_id" 2>&1 >/dev/null)" \
    && status=0 || status=$?

  if (( status == 0 )); then
    echo "$secret_ref"
    return 0
  fi

  if printf '%s' "$gcloud_error" | grep -q 'NOT_FOUND'; then
    echo "ERROR: Secret Manager secret API_ACCESS_KEY does not exist in project ${project_id}. Refusing to deploy a public ${label} without an access key. Create it first:" >&2
    echo "  gcloud secrets create API_ACCESS_KEY --replication-policy=automatic --project ${project_id}" >&2
    echo "  printf %s \"\$(openssl rand -hex 32)\" | gcloud secrets versions add API_ACCESS_KEY --data-file=- --project ${project_id}" >&2
    echo "--- gcloud output ---" >&2
    echo "$gcloud_error" >&2
    return 1
  fi

  # 権限不足やその他の理由。事前チェックの権限は実デプロイの権限と別物なので、
  # ここで止めない（止めるとデプロイ可能な状況でもCIが永久に進めなくなる）。
  echo "WARNING: could not verify the API_ACCESS_KEY secret before deploying the ${label} (this check needs secretmanager.secrets.get, which the deploy itself does not). Continuing; gcloud run deploy --set-secrets will fail if the secret is truly unusable." >&2
  echo "--- gcloud output ---" >&2
  echo "$gcloud_error" >&2
  echo "$secret_ref"
  return 0
}
