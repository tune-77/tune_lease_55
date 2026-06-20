#!/bin/bash
# 日次改善の本体
# 改善候補の収集・検証・反映をまとめて実行する

PROJECT_ROOT="${PROJECT_ROOT:-/Users/kobayashiisaoryou/clawd/tune_lease_55}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
LOG_DATE="${LOG_DATE:-$(date +%Y%m%d)}"
RESULT_FILE="${RESULT_FILE:-${HOME}/Library/Logs/tunelease/reports/improvement_report_${LOG_DATE}.json}"
EXPORT_FILE="${EXPORT_FILE:-/tmp/obsidian_improvements_export.txt}"

# ステップ結果を構造化ログに記録するヘルパー
log_step() {
    local step_name="$1"
    local exit_code="$2"
    local duration_s="${3:-0}"
    local log_file="${PROJECT_ROOT}/data/pipeline_step_log.jsonl"
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "{\"ts\":\"${ts}\",\"run_date\":\"${LOG_DATE}\",\"step\":\"${step_name}\",\"exit_code\":${exit_code},\"duration_s\":${duration_s}}" >> "${log_file}"
}

echo ""
echo "[入力・同期] 実装済み改善を Obsidian インデックスに自動同期中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_implemented_to_obsidian.py" || true

echo ""
echo "[診断] マクロデータ更新..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/fetch_fincept_data.py" || true

# aurion の異常は改善候補より先に拾う
echo ""
echo "[診断] aurion 自動診断ステータス確認..."
EXPORT_FILE="${EXPORT_FILE}" "${PYTHON}" "${PROJECT_ROOT}/scripts/check_aurion_state.py" || true

# 診断用の改善候補抽出
echo ""
echo "[診断] Obsidian 改善インデックスから改善案を抽出中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/extract_obsidian_improvements.py"
STEP0_EXIT=$?
log_step "extract_obsidian_improvements" ${STEP0_EXIT}
if [ ${STEP0_EXIT} -ne 0 ]; then
    echo "警告: 改善インデックス抽出が終了コード ${STEP0_EXIT} で終了しました（パイプラインを継続します）"
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

# 補助ソースを加算して改善候補を厚くする
echo ""
echo "[診断] lease-wiki-vault @AI_Insight_Evolved から改善案を差分抽出中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/extract_wiki_vault_insights.py" >> "${EXPORT_FILE}" || true

# スコア指標のドリフトも追記
echo ""
echo "[診断] DB スコアリング指標の自動分析中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_scoring_drift.py" >> "${EXPORT_FILE}" || true

echo ""
echo "[改善] auto-improvement-pipeline 実行中..."
"${PYTHON}" "${PROJECT_ROOT}/.agents/skills/auto-improvement-pipeline/pipeline_runner.py" \
    "${EXPORT_FILE}" \
    --output "${RESULT_FILE}" \
    --workspace "${PROJECT_ROOT}"
PIPELINE_EXIT=$?
log_step "auto_improvement_pipeline" ${PIPELINE_EXIT}

LATEST_FILE="${PROJECT_ROOT}/reports/latest.json"
GIST_ID="3980215df65cf75e972471f048b10d15"
FINAL_EXIT="${PIPELINE_EXIT}"
if [ -f "${RESULT_FILE}" ]; then
    cp "${RESULT_FILE}" "${LATEST_FILE}"
fi

# 改善済みを latest に同期
if [ -f "${RESULT_FILE}" ]; then
    echo ""
    echo "[反映] 改善済み項目を report/latest に同期中..."
    "${PYTHON}" "${PROJECT_ROOT}/.agents/skills/improvement-report-sync/scripts/sync_improvement_reports.py" \
        --report "${RESULT_FILE}" \
        --latest "${LATEST_FILE}" \
        --from-report
    SYNC_EXIT=$?
    log_step "sync_improvement_reports" ${SYNC_EXIT}
    if [ ${SYNC_EXIT} -ne 0 ]; then
        echo "警告: レポート反映の同期に失敗しました（終了コード ${SYNC_EXIT}）"
        if [ ${FINAL_EXIT} -eq 0 ]; then
            FINAL_EXIT=${SYNC_EXIT}
        fi
    fi
fi

echo ""
echo "[反映] RAG フィードバック分析 — ブースト/ペナルティ候補を台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_rag_feedback.py" || true

echo ""
echo "[反映] パイプラインヘルス分析 — 失敗率の高いステップをルール台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_pipeline_health.py" || true

echo ""
echo "[反映] batch_apply — 台帳ルールを自動適用中..."
"${PYTHON}" "${PROJECT_ROOT}/api/rule_engine/batch_apply.py" --apply || true

echo ""
echo "[反映] 再帰的自己改善レポートを生成中..."
RECURSIVE_JSON_FILE="${PROJECT_ROOT}/reports/recursive_self_improvement_${LOG_DATE}.json"
RECURSIVE_MD_FILE="${PROJECT_ROOT}/reports/recursive_self_improvement_${LOG_DATE}.md"
RECURSIVE_LATEST_JSON="${PROJECT_ROOT}/reports/recursive_self_improvement_latest.json"
RECURSIVE_LATEST_MD="${PROJECT_ROOT}/reports/recursive_self_improvement_latest.md"
"${PYTHON}" "${PROJECT_ROOT}/scripts/recursive_self_improvement.py" \
    --report "${LATEST_FILE}" \
    --prompt-log "${PROJECT_ROOT}/data/prompt_feedback_log.jsonl" \
    --output-json "${RECURSIVE_JSON_FILE}" \
    --output-md "${RECURSIVE_MD_FILE}" \
    --latest-json "${RECURSIVE_LATEST_JSON}" \
    --latest-md "${RECURSIVE_LATEST_MD}"
RECURSIVE_EXIT=$?
log_step "recursive_self_improvement" ${RECURSIVE_EXIT}
if [ ${RECURSIVE_EXIT} -ne 0 ]; then
    echo "警告: 再帰的自己改善レポート生成に失敗しました（終了コード ${RECURSIVE_EXIT}）"
    if [ ${FINAL_EXIT} -eq 0 ]; then
        FINAL_EXIT=${RECURSIVE_EXIT}
    fi
fi

echo ""
echo "[反映] Obsidian RAG評価・安全な自動修正を実行中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/auto_fix_obsidian_rag.py" \
    --eval-set "${PROJECT_ROOT}/api/knowledge/rag_eval_set.json" \
    --config "${PROJECT_ROOT}/config/rag_ranking.json" \
    --report "${PROJECT_ROOT}/reports/rag_auto_fix_latest.json" || \
    echo "警告: RAG自動修正は完了しませんでした（改善パイプラインは継続します）"

# Codex PR ステータス同期（merged / rejected を status ファイルに書き戻す）
echo ""
echo "[反映] Codex PR マージ/クローズ状態を execution_status に同期中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_codex_pr_status.py" || true

# Codex キュー
if [ -f "${RESULT_FILE}" ]; then
    echo ""
    echo "[反映] Codex 自動実行キューを生成中..."
    CODEX_QUEUE_FILE="${PROJECT_ROOT}/reports/codex_auto_queue_${LOG_DATE}.json"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/build_codex_auto_queue.py" \
        --report "${RESULT_FILE}" \
        --latest "${LATEST_FILE}" \
        --output "${CODEX_QUEUE_FILE}" \
        --limit 3
    QUEUE_EXIT=$?
    log_step "build_codex_auto_queue" ${QUEUE_EXIT}
    if [ ${QUEUE_EXIT} -ne 0 ]; then
        echo "警告: Codex 自動実行キュー生成に失敗しました（終了コード ${QUEUE_EXIT}）"
        if [ ${FINAL_EXIT} -eq 0 ]; then
            FINAL_EXIT=${QUEUE_EXIT}
        fi
    fi

    if [ -f "${CODEX_QUEUE_FILE}" ]; then
        echo ""
        echo "[反映] Codex 自動実行キューを実行中..."
        "${PYTHON}" "${PROJECT_ROOT}/scripts/execute_codex_queue.py" \
            --queue "${CODEX_QUEUE_FILE}" || true
        EXECUTE_EXIT=$?
        log_step "execute_codex_queue" ${EXECUTE_EXIT}
        if [ ${EXECUTE_EXIT} -ne 0 ]; then
            echo "警告: Codex 自動実行キューの実行に失敗しました（終了コード ${EXECUTE_EXIT}）"
        fi
    fi
fi

# Wiki 昇格キュー
echo ""
echo "[反映] Wiki 昇格キューを生成中..."
WIKI_QUEUE_FILE="${PROJECT_ROOT}/reports/wiki_promotion_queue_${LOG_DATE}.json"
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_wiki_promotion_queue.py" \
    --latest "${LATEST_FILE}" \
    --output "${WIKI_QUEUE_FILE}" \
    --limit 3
WIKI_QUEUE_EXIT=$?
log_step "build_wiki_promotion_queue" ${WIKI_QUEUE_EXIT}
if [ ${WIKI_QUEUE_EXIT} -ne 0 ]; then
    echo "警告: Wiki 昇格キュー生成に失敗しました（終了コード ${WIKI_QUEUE_EXIT}）"
    if [ ${FINAL_EXIT} -eq 0 ]; then
        FINAL_EXIT=${WIKI_QUEUE_EXIT}
    fi
else
    echo ""
    echo "[反映] Wiki 昇格キューを自動適用中..."
    "${PYTHON}" "${PROJECT_ROOT}/scripts/promote_wiki_queue.py" \
        --queue "${WIKI_QUEUE_FILE}" \
        --latest "${LATEST_FILE}" \
        --limit 3
    WIKI_PROMOTE_EXIT=$?
    log_step "promote_wiki_queue" ${WIKI_PROMOTE_EXIT}
    if [ ${WIKI_PROMOTE_EXIT} -ne 0 ]; then
        echo "警告: Wiki 昇格キューの自動適用に失敗しました（終了コード ${WIKI_PROMOTE_EXIT}）"
        if [ ${FINAL_EXIT} -eq 0 ]; then
            FINAL_EXIT=${WIKI_PROMOTE_EXIT}
        fi
    fi
fi

if [ -f "${LATEST_FILE}" ]; then
    echo ""
    echo "[配布] Gist に最終結果を更新中..."
    if [ ${FINAL_EXIT} -eq 0 ]; then
        if command -v gh >/dev/null 2>&1; then
            if gh gist edit "${GIST_ID}" "${LATEST_FILE}" 2>/dev/null; then
                echo "Gist 更新完了: https://gist.github.com/tune-77/${GIST_ID}"
                GIST_EXIT=0
            else
                GIST_EXIT=$?
                echo "警告: Gist 更新に失敗しました（ローカル結果は保存済み）"
            fi
            log_step "gist_update" ${GIST_EXIT}
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

exit "${FINAL_EXIT}"
