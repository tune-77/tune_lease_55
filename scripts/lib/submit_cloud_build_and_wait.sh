#!/usr/bin/env bash
# gcloud builds submit（--async を付けない既定の呼び方）はビルド完了を待つ間、
# Cloud Buildのデフォルトログバケットをストリーミング読みしようとする。
# WIF(Workload Identity Federation)認証のCIランナーからだと、対象サービス
# アカウントに roles/storage.admin を持たせていても
#   "This tool can only stream logs if you are Viewer/Owner of the project"
# で失敗することがある（2026-09、.github/workflows/deploy.yml の自動デプロイが
# 数時間おきに同じ位置で失敗し続けた原因。IAMロール追加では解決しなかった）。
#
# ビルドを --async で投げて、Cloud Build API（gcloud builds describe）の
# status フィールドだけを見て完了を待つことで、ログバケット読み取り自体を
# 回避する。
#
# 使い方:
#   source scripts/lib/submit_cloud_build_and_wait.sh
#   submit_cloud_build_and_wait "$PROJECT_ID" cloudbuild.api.yaml "_IMAGE_URI=${IMAGE_URI}" .
submit_cloud_build_and_wait() {
  local project_id="$1"
  local config="$2"
  local substitutions="$3"
  local context_dir="$4"

  local build_id
  build_id="$(gcloud builds submit \
    --project "$project_id" \
    --config "$config" \
    --substitutions "$substitutions" \
    --async \
    --format='value(id)' \
    "$context_dir")"

  echo "Cloud Build ID: ${build_id}"
  echo "Logs: https://console.cloud.google.com/cloud-build/builds/${build_id}?project=${project_id}"

  local status
  while true; do
    status="$(gcloud builds describe "$build_id" --project "$project_id" --format='value(status)')"
    case "$status" in
      SUCCESS)
        echo "Build succeeded."
        return 0
        ;;
      WORKING|QUEUED|PENDING|"")
        sleep 10
        ;;
      *)
        echo "Cloud Build failed with status: ${status}" >&2
        return 1
        ;;
    esac
  done
}
