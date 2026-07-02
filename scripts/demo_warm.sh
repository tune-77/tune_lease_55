#!/bin/bash
# 発表・審査の直前に API を温める（min-instances=1）／終わったら冷ます（0）。
# 再デプロイせず min-instances 設定だけを更新するので数秒で反映される。
#
# 使い方:
#   ./scripts/demo_warm.sh on      # 温める（常時1インスタンス。コールドスタート無し）
#   ./scripts/demo_warm.sh off     # 冷ます（ゼロスケール。アイドル課金ほぼ0）
#   ./scripts/demo_warm.sh status  # 現在の min-instances と状態を表示
#
# 環境変数で上書き可: SERVICE_NAME / REGION / PROJECT_ID
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-tune-lease-55-api}"
REGION="${REGION:-asia-northeast1}"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID が取得できません。gcloud config set project <id> を実行するか PROJECT_ID= で指定してください。" >&2
  exit 1
fi

action="${1:-}"

case "$action" in
  on)
    echo "温めます: ${SERVICE_NAME} (${REGION}) min-instances=1 ..."
    gcloud run services update "$SERVICE_NAME" \
      --project "$PROJECT_ID" --region "$REGION" \
      --min-instances=1
    echo "完了。1インスタンス常駐。発表が終わったら必ず ./scripts/demo_warm.sh off で冷ましてください。"
    ;;
  off)
    echo "冷まします: ${SERVICE_NAME} (${REGION}) min-instances=0 ..."
    gcloud run services update "$SERVICE_NAME" \
      --project "$PROJECT_ID" --region "$REGION" \
      --min-instances=0
    echo "完了。ゼロスケール。次の1発目はコールドスタート（10〜30秒）になります。"
    ;;
  status)
    echo "現在の設定: ${SERVICE_NAME} (${REGION})"
    gcloud run services describe "$SERVICE_NAME" \
      --project "$PROJECT_ID" --region "$REGION" \
      --format="value(spec.template.metadata.annotations['autoscaling.knative.dev/minScale'])" \
      | sed 's/^/  min-instances=/'
    ;;
  *)
    echo "使い方: $0 {on|off|status}" >&2
    echo "  on     発表直前に温める (min-instances=1)" >&2
    echo "  off    発表後に冷ます (min-instances=0)" >&2
    echo "  status 現在の min-instances を表示" >&2
    exit 1
    ;;
esac
