#!/usr/bin/env bash
# 公開Web(tune-lease-55-web)に Secret Manager の PUBLIC_TUNNEL_AUTH を必ず配線する、
# fail-closed方針の実装。require_api_access_key_secret.sh と同じ理由・同じ形。
#
# ── 背景 ──────────────────────────────────────────────────────────────
# Web境界を --allow-unauthenticated にするだけでは、frontend/src/proxy.ts が
# 未認証の全 /api/* リクエストへ特権的な API_ACCESS_KEY を自動付与してしまい、
# 誰でもコスト発生するAPIを叩けるようになる（PR #983で実際に再発、Codexレビュー
# で指摘）。proxy.ts は PUBLIC_TUNNEL=1 かつ PUBLIC_TUNNEL_AUTH が設定されている
# 時、/api/* を含む全パスをBasic認証（ID: lease）で保護してから通す作りに
# なっているため、この2つを必ずセットで配線することで
# 「ブラウザで直接開ける」と「未認証の第三者はAPIを叩けない」を両立する。
# PUBLIC_TUNNEL_AUTHが無いままPUBLIC_TUNNEL=1だけ設定すると、proxy.ts は
# 各リクエストを401で拒否する（tunnelPasswordが空になるため）ので誰も使えなく
# なるだけで安全側に倒れるが、それでは公開する意味が無いため、ここでも
# API_ACCESS_KEY同様デプロイ前にfail-closedで止める。
#
# 使い方:
#   secret_ref="$(require_public_tunnel_auth_secret "$PROJECT_ID" "Web")" || exit 1
#   deploy_args+=(--set-secrets "...,PUBLIC_TUNNEL_AUTH=${secret_ref}")
#
# 第2引数はエラーメッセージに出すサービスの呼び名（任意、既定 "service"）。
require_public_tunnel_auth_secret() {
  local project_id="$1"
  local label="${2:-service}"
  local secret_ref="PUBLIC_TUNNEL_AUTH:latest"
  local gcloud_error
  local status

  gcloud_error="$(gcloud secrets describe PUBLIC_TUNNEL_AUTH --project "$project_id" 2>&1 >/dev/null)" \
    && status=0 || status=$?

  if (( status == 0 )); then
    echo "$secret_ref"
    return 0
  fi

  if printf '%s' "$gcloud_error" | grep -q 'NOT_FOUND'; then
    echo "ERROR: Secret Manager secret PUBLIC_TUNNEL_AUTH does not exist in project ${project_id}. Refusing to deploy a public ${label} without a tunnel password (proxy.ts would 401 every request). Create it first:" >&2
    echo "  gcloud secrets create PUBLIC_TUNNEL_AUTH --replication-policy=automatic --project ${project_id}" >&2
    echo "  printf %s \"\$(openssl rand -hex 32)\" | gcloud secrets versions add PUBLIC_TUNNEL_AUTH --data-file=- --project ${project_id}" >&2
    echo "--- gcloud output ---" >&2
    echo "$gcloud_error" >&2
    return 1
  fi

  echo "WARNING: could not verify the PUBLIC_TUNNEL_AUTH secret before deploying the ${label} (this check needs secretmanager.secrets.get, which the deploy itself does not). Continuing; gcloud run deploy --set-secrets will fail if the secret is truly unusable." >&2
  echo "--- gcloud output ---" >&2
  echo "$gcloud_error" >&2
  echo "$secret_ref"
  return 0
}
