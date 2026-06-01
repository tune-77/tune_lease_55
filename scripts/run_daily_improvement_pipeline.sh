#!/bin/bash
# Obsidian [[改善]] インデックス → auto-improvement-pipeline 自動実行
# 毎日 AM 4:00 に com.tunelease.improvement-pipeline LaunchAgent から起動される

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

# --- Step -1: git コミット履歴から実装済みを Obsidian に自動追記 ---
echo ""
echo "[Step -1] 実装済み改善を Obsidian インデックスに自動同期中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_implemented_to_obsidian.py" || true

# --- Step 0: Obsidian 改善インデックス抽出 ---
echo ""
echo "[Step 0] Obsidian 改善インデックスから改善案を抽出中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/extract_obsidian_improvements.py"
STEP0_EXIT=$?
if [ ${STEP0_EXIT} -ne 0 ]; then
    echo "警告: Step 0 が終了コード ${STEP0_EXIT} で終了しました（パイプラインを継続します）"
fi

# エクスポートファイルが空 / 存在しない場合は中断
if [ ! -s "${EXPORT_FILE}" ]; then
    echo "警告: ${EXPORT_FILE} が空またはが存在しません。パイプラインをスキップします。"
    echo "========================================"
    echo "改善パイプライン終了（スキップ）: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    exit 0
fi

IMPROVEMENT_COUNT=$(grep -c '^\[改善\]\|^\[TODO\]' "${EXPORT_FILE}" 2>/dev/null || echo 0)
echo "抽出された改善案タグ数: ${IMPROVEMENT_COUNT}件"

# --- Step 1-3: pipeline_runner.py 経由で一括実行 ---
echo ""
echo "[Step 1-3] auto-improvement-pipeline 実行中..."
"${PYTHON}" "${PROJECT_ROOT}/.agents/skills/auto-improvement-pipeline/pipeline_runner.py" \
    "${EXPORT_FILE}" \
    --output "${RESULT_FILE}" \
    --workspace "${PROJECT_ROOT}"
PIPELINE_EXIT=$?

# --- Step 4: reports/latest.json を更新（Step 5 の前提） ---
LATEST_FILE="${PROJECT_ROOT}/reports/latest.json"
GIST_ID="3980215df65cf75e972471f048b10d15"
FINAL_EXIT="${PIPELINE_EXIT}"
if [ -f "${RESULT_FILE}" ]; then
    cp "${RESULT_FILE}" "${LATEST_FILE}"
fi

# --- Step 5: 改善済み登録を report/latest に同期 ---
if [ -f "${RESULT_FILE}" ]; then
    echo ""
    echo "[Step 5] 改善済み項目を report/latest に同期中..."
    "${PYTHON}" "${PROJECT_ROOT}/.agents/skills/improvement-report-sync/scripts/sync_improvement_reports.py" \
        --report "${RESULT_FILE}" \
        --latest "${LATEST_FILE}" \
        --from-report
    SYNC_EXIT=$?
    if [ ${SYNC_EXIT} -ne 0 ]; then
        echo "警告: Step 5 の同期に失敗しました（終了コード ${SYNC_EXIT}）"
        if [ ${FINAL_EXIT} -eq 0 ]; then
            FINAL_EXIT=${SYNC_EXIT}
        fi
    fi
fi

# --- Step 6: 最終結果を Gist に push ---
if [ -f "${LATEST_FILE}" ]; then
    echo ""
    echo "[Step 6] Gist に最終結果を更新中..."
    if [ ${FINAL_EXIT} -eq 0 ]; then
        if command -v gh >/dev/null 2>&1; then
            if gh gist edit "${GIST_ID}" "${LATEST_FILE}" 2>/dev/null; then
                echo "Gist 更新完了: https://gist.github.com/tune-77/${GIST_ID}"
                GIST_EXIT=0
            else
                GIST_EXIT=$?
                echo "警告: Gist 更新に失敗しました（ローカル結果は保存済み）"
            fi
            if [ ${GIST_EXIT} -ne 0 ] && [ ${FINAL_EXIT} -eq 0 ]; then
                FINAL_EXIT=${GIST_EXIT}
            fi
        else
            echo "警告: gh コマンドが見つかりません（Gist 更新スキップ）"
            if [ ${FINAL_EXIT} -eq 0 ]; then
                FINAL_EXIT=1
            fi
        fi
    else
        echo "警告: 前段で失敗したため Gist 更新をスキップします"
    fi
fi

echo ""
echo "========================================"
echo "改善パイプライン終了: $(date '+%Y-%m-%d %H:%M:%S')"
echo "終了コード: ${FINAL_EXIT}"
if [ -f "${RESULT_FILE}" ]; then
    echo "結果ファイル: ${RESULT_FILE}"
fi
echo "Gist: https://gist.githubusercontent.com/tune-77/${GIST_ID}/raw/latest.json"
echo "ログファイル: ${LOG_FILE}"
echo "========================================"
exit "${FINAL_EXIT}"
