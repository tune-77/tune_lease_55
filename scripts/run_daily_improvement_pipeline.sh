#!/bin/bash
# Obsidian [[改善]] インデックス → auto-improvement-pipeline 自動実行
# 毎日 AM 4:00 に com.tunelease.improvement-pipeline LaunchAgent から起動される

PROJECT_ROOT="/Users/kobayashiisaoryou/clawd/tune_lease_55"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_DATE="$(date +%Y%m%d)"
LOG_FILE="/tmp/improvement_pipeline_${LOG_DATE}.log"
RESULT_FILE="/tmp/improvement_pipeline_${LOG_DATE}_result.json"
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

echo ""
echo "========================================"
echo "改善パイプライン終了: $(date '+%Y-%m-%d %H:%M:%S')"
echo "終了コード: ${PIPELINE_EXIT}"
if [ -f "${RESULT_FILE}" ]; then
    echo "結果ファイル: ${RESULT_FILE}"
fi
echo "ログファイル: ${LOG_FILE}"
echo "========================================"
