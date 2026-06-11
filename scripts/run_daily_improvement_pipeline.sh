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

# --- Step 0.0: マクロデータ更新 ---
echo ""
echo "[Step 0.0] マクロデータ更新..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/fetch_fincept_data.py" || true

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

# --- Step 0.5: lease-wiki-vault @AI_Insight_Evolved から差分抽出（追記） ---
echo ""
echo "[Step 0.5] lease-wiki-vault @AI_Insight_Evolved から改善案を差分抽出中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/extract_wiki_vault_insights.py" >> "${EXPORT_FILE}" || true

# --- Step 0.6: DB スコアリング指標の自動分析（追記） ---
echo ""
echo "[Step 0.6] DB スコアリング指標の自動分析中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_scoring_drift.py" >> "${EXPORT_FILE}" || true

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

# --- Step 6: Codex 自動実行キューを生成 ---
if [ -f "${RESULT_FILE}" ]; then
    echo ""
    echo "[Step 6] Codex 自動実行キューを生成中..."
    CODEX_QUEUE_FILE="${PROJECT_ROOT}/reports/codex_auto_queue_${LOG_DATE}.json"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/build_codex_auto_queue.py" \
        --report "${RESULT_FILE}" \
        --latest "${LATEST_FILE}" \
        --output "${CODEX_QUEUE_FILE}" \
        --limit 3
    QUEUE_EXIT=$?
    if [ ${QUEUE_EXIT} -ne 0 ]; then
        echo "警告: Codex 自動実行キュー生成に失敗しました（終了コード ${QUEUE_EXIT}）"
        if [ ${FINAL_EXIT} -eq 0 ]; then
            FINAL_EXIT=${QUEUE_EXIT}
        fi
    fi
fi

# --- Step 7: Wiki 昇格キューを生成 ---
echo ""
echo "[Step 7] Wiki 昇格キューを生成中..."
WIKI_QUEUE_FILE="${PROJECT_ROOT}/reports/wiki_promotion_queue_${LOG_DATE}.json"
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_wiki_promotion_queue.py" \
    --latest "${LATEST_FILE}" \
    --output "${WIKI_QUEUE_FILE}" \
    --limit 3
WIKI_QUEUE_EXIT=$?
if [ ${WIKI_QUEUE_EXIT} -ne 0 ]; then
    echo "警告: Wiki 昇格キュー生成に失敗しました（終了コード ${WIKI_QUEUE_EXIT}）"
    if [ ${FINAL_EXIT} -eq 0 ]; then
        FINAL_EXIT=${WIKI_QUEUE_EXIT}
    fi
else
    echo ""
    echo "[Step 7.1] Wiki 昇格キューを自動適用中..."
    "${PYTHON}" "${PROJECT_ROOT}/scripts/promote_wiki_queue.py" \
        --queue "${WIKI_QUEUE_FILE}" \
        --latest "${LATEST_FILE}" \
        --limit 3
    WIKI_PROMOTE_EXIT=$?
    if [ ${WIKI_PROMOTE_EXIT} -ne 0 ]; then
        echo "警告: Wiki 昇格キューの自動適用に失敗しました（終了コード ${WIKI_PROMOTE_EXIT}）"
        if [ ${FINAL_EXIT} -eq 0 ]; then
            FINAL_EXIT=${WIKI_PROMOTE_EXIT}
        fi
    fi
fi

# --- Step 8: 最終結果を Gist に push ---
if [ -f "${LATEST_FILE}" ]; then
    echo ""
    echo "[Step 8] Gist に最終結果を更新中..."
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
echo "[Step 9] 滞留改善案の自動 parking（21日以上 needs_review）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/rotate_weekly_focus.py" || true

echo ""
echo "[Step 10] e-Stat業種別統計更新..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/fetch_estat_industry.py" || true

echo ""
echo "[Step 11] Sidecar Agent Brief を生成（読み取り専用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/agent_sidecar_reader.py" || true

echo ""
echo "[Step 12] DAILY-BRIEF.md を Obsidian Vault に書き出し..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/write_daily_brief.py" || true

echo ""
echo "[Step 13] 週次セルフマネジメントサマリ（月曜のみ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/weekly_self_management.py" || true

echo ""
echo "[Step 14] レシピ生成（承認はしない）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/generate_recipes.py" || true
PENDING_RECIPE_COUNT=$(ls "${PROJECT_ROOT}/data/recipes/pending/"*.json 2>/dev/null | wc -l | tr -d ' ')
echo "承認待ちレシピ: ${PENDING_RECIPE_COUNT}件"

echo ""
echo "========================================"
echo "改善パイプライン終了: $(date '+%Y-%m-%d %H:%M:%S')"
echo "終了コード: ${FINAL_EXIT}"
if [ -f "${RESULT_FILE}" ]; then
    echo "結果ファイル: ${RESULT_FILE}"
fi
if [ -n "${CODEX_QUEUE_FILE:-}" ] && [ -f "${CODEX_QUEUE_FILE}" ]; then
    echo "Codex 自動実行キュー: ${CODEX_QUEUE_FILE}"
fi
if [ -n "${WIKI_QUEUE_FILE:-}" ] && [ -f "${WIKI_QUEUE_FILE}" ]; then
    echo "Wiki 昇格キュー: ${WIKI_QUEUE_FILE}"
fi
if [ -f "${PROJECT_ROOT}/reports/wiki_promotion_status.json" ]; then
    echo "Wiki 昇格ステータス: ${PROJECT_ROOT}/reports/wiki_promotion_status.json"
fi
echo "Gist: https://gist.githubusercontent.com/tune-77/${GIST_ID}/raw/latest.json"
echo "ログファイル: ${LOG_FILE}"
echo "========================================"
exit "${FINAL_EXIT}"
