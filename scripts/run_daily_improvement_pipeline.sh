#!/bin/bash
# 日次改善パイプラインの入口
# core と post を順番に呼ぶラッパー

PROJECT_ROOT="/Users/kobayashiisaoryou/clawd/tune_lease_55"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_DATE="$(date +%Y%m%d)"
LOG_DIR="${HOME}/Library/Logs/tunelease"
mkdir -p "${LOG_DIR}/reports"
LOG_FILE="${LOG_DIR}/improvement_${LOG_DATE}.log"
RESULT_FILE="${LOG_DIR}/reports/improvement_report_${LOG_DATE}.json"
EXPORT_FILE="/tmp/obsidian_improvements_export.txt"

# ログへリダイレクト（stdout + stderr を同一ファイルへ）
exec >> "${LOG_FILE}" 2>&1

echo "========================================"
echo "改善パイプライン開始: $(date '+%Y-%m-%d %H:%M:%S')"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "========================================"

# Python バイナリ確認
if [ ! -x "${PYTHON}" ]; then
    echo "エラー: Python バイナリが見つかりません: ${PYTHON}"
    exit 1
fi

# launchd/cron は .zshrc を読まないため、gcloud が見つからない場合は標準的な
# インストール先を PATH に補完する（GCS取り込み・Secret Manager・GCSアップロードで使用）
if ! command -v gcloud >/dev/null 2>&1; then
    export PATH="${PATH}:/opt/homebrew/bin:/usr/local/bin:${HOME}/google-cloud-sdk/bin"
fi
echo "gcloud: $(command -v gcloud || echo '見つかりません（クラウド連携ステップは失敗します）')"

echo ""
echo "[core] 改善コア処理を実行中..."
bash "${PROJECT_ROOT}/scripts/run_daily_improvement_core.sh"
CORE_EXIT=$?

echo ""
echo "[post] 補助処理を実行中..."
bash "${PROJECT_ROOT}/scripts/run_daily_improvement_post.sh"
POST_EXIT=$?

FINAL_EXIT=${CORE_EXIT}
if [ ${FINAL_EXIT} -eq 0 ] && [ ${POST_EXIT} -ne 0 ]; then
    FINAL_EXIT=${POST_EXIT}
fi

echo ""
echo "========================================"
echo "改善パイプライン終了: $(date '+%Y-%m-%d %H:%M:%S')"
echo "終了コード: ${FINAL_EXIT}"
echo "core 終了コード: ${CORE_EXIT}"
echo "post 終了コード: ${POST_EXIT}"
if [ -f "${RESULT_FILE}" ]; then
    echo "結果ファイル: ${RESULT_FILE}"
fi
echo "ログファイル: ${LOG_FILE}"
echo "========================================"
exit "${FINAL_EXIT}"
