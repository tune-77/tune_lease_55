#!/usr/bin/env bash
# 本番Cloud Run APIのJev（TypeSafe）設定の単一の実装（REV-425）。
#
# gcloud run deploy --set-env-vars は環境変数を丸ごと置き換えるため、CI自動デプロイ
# (.github/workflows/deploy.yml) とローカル手動デプロイ (scripts/deploy_cloud_run_api.sh)
# の片方にだけ書くと、もう片方でデプロイした瞬間にJev設定が消える。両方ここを経由する。
#
# ── 本番の初期モード（すべて安全側） ─────────────────────────────
# TYPESAFE_ROUTING_MODE=shadow
#   ローカルは enforce だが、ローカルAPIログに [TypeSafeRouting] 行が一度も無く
#   実績ゼロ。shadow では従来のGemini分類が決め、Jevの判定はログに残すだけ。
# TYPESAFE_RAG_MODE=shadow
#   判定してログに残すだけで、検索結果は一切変えない。
# TYPESAFE_RECIPE_CLASSIFY_ENABLED=0
#   shadowが無く、キーがあるだけで判定結果がそのまま採用される（enforce相当）。
#   改善パイプライン（ローカル）用の分類なので本番では止めておく。
# TYPESAFE_RAG_TIMEOUT_SECONDS=5
#   shadowでは1チャットにJev呼び出しが最大2回直列で増える（concurrency=1）。
#   Jev障害時の上乗せを抑える（既定8秒。routingとRAGで共有）。
# TYPESAFE_ALLOW_SCREENING / TYPESAFE_ALLOW_SHARED_CONTEXT / TYPESAFE_CURATION_MODE は
# 意図的に未設定（off）。個人情報・案件情報を外部に出さないゲートをローカルと揃える。
# ニュース・調査検証・重複判定はローカルlaunchd専用でCloud Runでは動かないため設定しない。
JEV_CLOUD_RUN_ENV_VARS="TYPESAFE_ROUTING_MODE=shadow,TYPESAFE_RAG_MODE=shadow,TYPESAFE_RECIPE_CLASSIFY_ENABLED=0,TYPESAFE_RAG_TIMEOUT_SECONDS=5"

# --set-secrets に渡す TYPESAFE_API_KEY の参照を出力する。見つからなければ何も出力しない。
# Jevのキーが無いことでデプロイを止めてはいけない（ガードがoffのままになるだけ）ので
# 常に 0 を返す。
#
# CIのデプロイ用サービスアカウントは secretmanager.secrets.get を持たず describe が
# 失敗する。存在しないシークレットを配線するとデプロイ自体が失敗するため、CIでは
# GitHub Actions変数 TYPESAFE_API_KEY_SECRET による明示的なオプトインだけで配線する。
# 手動デプロイ（本人の権限）では describe で存在を確認して自動で配線する。
#
# 使い方:
#   jev_ref="$(jev_typesafe_secret_ref "$PROJECT_ID")"
#   if [ -n "$jev_ref" ]; then secret_args+=("TYPESAFE_API_KEY=${jev_ref}"); fi
jev_typesafe_secret_ref() {
  local project_id="$1"
  if [ -n "${TYPESAFE_API_KEY_SECRET:-}" ]; then
    echo "${TYPESAFE_API_KEY_SECRET}:latest"
    return 0
  fi
  if gcloud secrets describe TYPESAFE_API_KEY --project "$project_id" >/dev/null 2>&1; then
    echo "TYPESAFE_API_KEY:latest"
    return 0
  fi
  echo "Warning: TYPESAFE_API_KEY is not wired; Jev guards stay off. Set the GitHub Actions variable TYPESAFE_API_KEY_SECRET or create the Secret Manager secret TYPESAFE_API_KEY." >&2
  return 0
}
