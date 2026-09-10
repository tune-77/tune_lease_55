#!/bin/bash
# 日次改善パイプラインの入口
# core と post を順番に呼ぶラッパー

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
# core.sh/post.sh は別プロセス（bash 起動）として呼ぶため、export しないと
# ここで動的解決した値が渡らず、双方のハードコード既定値に必ずフォールバックしていた。
export PROJECT_ROOT
export PYTHON
LOG_DATE="$(date +%Y%m%d)"
LOG_DIR="${HOME}/Library/Logs/tunelease"
mkdir -p "${LOG_DIR}/reports"
LOG_FILE="${LOG_DIR}/improvement_${LOG_DATE}.log"
RESULT_FILE="${LOG_DIR}/reports/improvement_report_${LOG_DATE}.json"
EXPORT_FILE="/tmp/obsidian_improvements_export.txt"
# PROJECT_ROOT/PYTHON と同じ理由（別プロセス起動のため export が無いと子スクリプトへ
# 渡らない）。値自体は core.sh 側の既定値と同一だが、将来ズレたときに気づけるようにする。
export EXPORT_FILE

# launchd と手動/別スケジューラが同時に入口を起動しても、成果物・SQLite・
# ChromaDB を二重更新しない。mkdir は同一Mac上で原子的なので、先着1本だけを
# 実行し、後発は正常スキップする。異常終了で残ったロックは PID を確認して回収する。
# 起動元ごとに TMPDIR が異なると排他できないため、全経路で同じ固定パスを使う。
PIPELINE_LOCK_DIR="${PIPELINE_LOCK_DIR:-/tmp/tunelease-daily-improvement.lock}"
PIPELINE_LOCK_PID_FILE="${PIPELINE_LOCK_DIR}/pid"

cleanup_pipeline_lock() {
    rm -f "${PIPELINE_LOCK_PID_FILE}"
    rmdir "${PIPELINE_LOCK_DIR}" 2>/dev/null || true
}

install_pipeline_lock_traps() {
    trap cleanup_pipeline_lock EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

acquire_pipeline_lock() {
    if mkdir "${PIPELINE_LOCK_DIR}" 2>/dev/null; then
        printf '%s\n' "$$" > "${PIPELINE_LOCK_PID_FILE}"
        install_pipeline_lock_traps
        return 0
    fi

    local owner_pid=""
    # mkdir 直後・PID書き込み直前のごく短い競合窓では、正当な先行実行の
    # ロックを残存扱いしないよう一度だけ待って読み直す。
    if [ ! -s "${PIPELINE_LOCK_PID_FILE}" ]; then
        sleep 1
    fi
    owner_pid="$(sed -n '1p' "${PIPELINE_LOCK_PID_FILE}" 2>/dev/null || true)"
    if [[ "${owner_pid}" =~ ^[0-9]+$ ]] && kill -0 "${owner_pid}" 2>/dev/null; then
        echo "改善パイプラインは既に実行中です（PID ${owner_pid}）。二重起動を正常スキップします。"
        return 1
    fi

    echo "警告: 残存した改善パイプラインロックを回収します: ${PIPELINE_LOCK_DIR}"
    rm -f "${PIPELINE_LOCK_PID_FILE}"
    rmdir "${PIPELINE_LOCK_DIR}" 2>/dev/null || return 1
    mkdir "${PIPELINE_LOCK_DIR}" 2>/dev/null || return 1
    printf '%s\n' "$$" > "${PIPELINE_LOCK_PID_FILE}"
    install_pipeline_lock_traps
    return 0
}

if ! acquire_pipeline_lock; then
    exit 0
fi

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

# パイプライン失敗を記録（リカバリー監視用）
if [ ${FINAL_EXIT} -ne 0 ]; then
    echo ""
    echo "[監視] パイプライン失敗を検出・記録中..."
    "${PYTHON}" "${PROJECT_ROOT}/scripts/detect_pipeline_failures.py" \
        --record-failure "${LOG_FILE}" || true
fi

exit "${FINAL_EXIT}"
