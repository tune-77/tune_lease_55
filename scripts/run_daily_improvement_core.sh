#!/bin/bash
# 日次改善の本体
# 改善候補の収集・検証・反映をまとめて実行する

PROJECT_ROOT="${PROJECT_ROOT:-/Users/kobayashiisaoryou/clawd/tune_lease_55}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
LOG_DATE="${LOG_DATE:-$(date +%Y%m%d)}"
RESULT_FILE="${RESULT_FILE:-${HOME}/Library/Logs/tunelease/reports/improvement_report_${LOG_DATE}.json}"
EXPORT_FILE="${EXPORT_FILE:-/tmp/obsidian_improvements_export.txt}"

# ステップ結果を構造化ログに記録するヘルパー（core/post 共通・pipeline_log_step.sh）
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_log_step.sh"

echo ""
echo "[入力・同期] Cloud Run入力イベントを GCS から取り込み中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_cloudrun_inputs_from_gcs.py"
SYNC_GCS_EXIT=$?
log_step "sync_cloudrun_inputs_from_gcs" ${SYNC_GCS_EXIT}
if [ ${SYNC_GCS_EXIT} -ne 0 ]; then
    echo "警告: Cloud Run入力イベントのGCS取り込みに失敗しました（終了コード ${SYNC_GCS_EXIT}）"
fi

echo ""
echo "[入力・同期] Cloud Run入力イベントを Obsidian 要約へ反映中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_cloudrun_inputs_to_obsidian.py"
SYNC_INPUT_OBSIDIAN_EXIT=$?
log_step "sync_cloudrun_inputs_to_obsidian" ${SYNC_INPUT_OBSIDIAN_EXIT}
if [ ${SYNC_INPUT_OBSIDIAN_EXIT} -ne 0 ]; then
    echo "警告: Cloud Run入力イベントのObsidian反映に失敗しました（終了コード ${SYNC_INPUT_OBSIDIAN_EXIT}）"
fi

echo ""
# Cloud SQL（tune-lease-db）は 2026-07 に廃止済み。Cloud Run 会話は GCS chat_exchange
# 経路で Obsidian に反映されるため、この同期ステップは既定で無効化している。
# 有効なままだと廃止済みインスタンスへの接続を毎晩試みて exit 1 で落ち、
# パイプラインヘルスを汚染していた（REV-027a）。将来 cloud-sql-proxy 経由などで
# 再利用する場合のみ ENABLE_CLOUDSQL_SYNC=1 を指定して有効化する。
if [ "${ENABLE_CLOUDSQL_SYNC:-0}" = "1" ]; then
    echo "[入力・同期] Cloud SQL 会話ログを Obsidian 要約へ反映中..."
    DATABASE_URL_SECRET_NAME="${DATABASE_URL_SECRET_NAME:-DATABASE_URL}" \
        "${PYTHON}" "${PROJECT_ROOT}/scripts/sync_cloudsql_to_obsidian.py"
    SYNC_CLOUDSQL_OBSIDIAN_EXIT=$?
    log_step "sync_cloudsql_to_obsidian" ${SYNC_CLOUDSQL_OBSIDIAN_EXIT}
    if [ ${SYNC_CLOUDSQL_OBSIDIAN_EXIT} -ne 0 ]; then
        echo "警告: Cloud SQL会話ログのObsidian反映に失敗しました（終了コード ${SYNC_CLOUDSQL_OBSIDIAN_EXIT}）"
    fi
else
    echo "[入力・同期] Cloud SQL 会話ログ同期は無効（Cloud SQL 廃止済み）。ENABLE_CLOUDSQL_SYNC=1 で有効化できます。スキップします。"
    # 意図的なスキップは「健全（exit 0）」として記録する。
    # これを記録しないと、廃止前に残った失敗ログが7日ウィンドウに残り続け、
    # analyze_pipeline_health が「直近成功なし＝障害継続」と誤検出して
    # REV-028a のようなゾンビ検出を毎晩作り直してしまう（REV-027a 対応の残課題）。
    log_step "sync_cloudsql_to_obsidian" 0
fi

echo ""
echo "[入力・同期] Cloud Runチャット用 Public Memory Pack を生成中..."
DATABASE_URL_SECRET_NAME="${DATABASE_URL_SECRET_NAME:-DATABASE_URL}" \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/build_cloud_chat_memory_pack.py"
MEMORY_PACK_EXIT=$?
log_step "build_cloud_chat_memory_pack" ${MEMORY_PACK_EXIT}
if [ ${MEMORY_PACK_EXIT} -ne 0 ]; then
    echo "警告: Public Memory Pack 生成に失敗しました（終了コード ${MEMORY_PACK_EXIT}）"
fi

echo ""
echo "[入力・同期] 実装済み改善を Obsidian インデックスに自動同期中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_implemented_to_obsidian.py" || true

JUDGMENT_PREVIEW_DATE="${JUDGMENT_PREVIEW_DATE:-$(date +%F)}"
JUDGMENT_PREVIEW_DAYS="${JUDGMENT_PREVIEW_DAYS:-3}"

echo ""
echo "[判断記憶] 判断材料 preview を生成中（昇格なし・人間レビュー前）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_judgment_materials_preview.py" \
    --date "${JUDGMENT_PREVIEW_DATE}" --days "${JUDGMENT_PREVIEW_DAYS}"
JUDGMENT_MATERIALS_EXIT=$?
log_step "build_judgment_materials_preview" ${JUDGMENT_MATERIALS_EXIT}
if [ ${JUDGMENT_MATERIALS_EXIT} -ne 0 ]; then
    echo "警告: 判断材料 preview 生成に失敗しました（終了コード ${JUDGMENT_MATERIALS_EXIT}）"
fi

echo ""
echo "[判断記憶] canonical rules preview を生成中（active判断基準へは未昇格）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_canonical_judgment_rules.py" \
    --date "${JUDGMENT_PREVIEW_DATE}"
CANONICAL_PREVIEW_EXIT=$?
log_step "build_canonical_judgment_rules_preview" ${CANONICAL_PREVIEW_EXIT}
if [ ${CANONICAL_PREVIEW_EXIT} -ne 0 ]; then
    echo "警告: canonical rules preview 生成に失敗しました（終了コード ${CANONICAL_PREVIEW_EXIT}）"
fi

# 紫苑記憶のメンテナンス。従来はデプロイ時（package_cloud_run_bundle.sh）のみで
# 記憶と鮮度が「最後にデプロイした日」で止まっていたため、夜間に毎日回す
echo ""
echo "[記憶] 紫苑記憶インデックスを再構築中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_memory_index.py"
MEMORY_INDEX_EXIT=$?
log_step "build_shion_memory_index" ${MEMORY_INDEX_EXIT}
if [ ${MEMORY_INDEX_EXIT} -ne 0 ]; then
    echo "警告: 記憶インデックス再構築に失敗しました（終了コード ${MEMORY_INDEX_EXIT}）"
fi

echo ""
echo "[記憶] 記憶の鮮度（last_used_at / stale降格）を更新中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/update_shion_memory_freshness.py"
MEMORY_FRESHNESS_EXIT=$?
log_step "update_shion_memory_freshness" ${MEMORY_FRESHNESS_EXIT}
if [ ${MEMORY_FRESHNESS_EXIT} -ne 0 ]; then
    echo "警告: 記憶鮮度更新に失敗しました（終了コード ${MEMORY_FRESHNESS_EXIT}）"
fi

echo ""
echo "[記憶] 記憶ベクトル索引（ハイブリッド想起用）を再構築中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_memory_vector_index.py"
MEMORY_VECTOR_EXIT=$?
log_step "build_shion_memory_vector_index" ${MEMORY_VECTOR_EXIT}
if [ ${MEMORY_VECTOR_EXIT} -ne 0 ]; then
    echo "警告: 記憶ベクトル索引の再構築に失敗しました（終了コード ${MEMORY_VECTOR_EXIT}）"
fi

echo ""
echo "[記憶] 記憶インデックスのヘルスチェック（件数急減の検知）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_shion_memory_health.py"
MEMORY_HEALTH_EXIT=$?
log_step "check_shion_memory_health" ${MEMORY_HEALTH_EXIT}
if [ ${MEMORY_HEALTH_EXIT} -ne 0 ]; then
    echo "警告: 記憶インデックスのヘルスチェックが異常を検知しました（終了コード ${MEMORY_HEALTH_EXIT}）"
fi

echo ""
echo "[記憶] 記憶想起の回帰評価（評価セット）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/eval_shion_memory_recall.py" \
    --index "${PROJECT_ROOT}/data/shion_memory_index.json" \
    --min-pass-rate 0.9
MEMORY_EVAL_EXIT=$?
log_step "eval_shion_memory_recall" ${MEMORY_EVAL_EXIT}
if [ ${MEMORY_EVAL_EXIT} -ne 0 ]; then
    echo "警告: 記憶想起の回帰評価が基準を下回りました（終了コード ${MEMORY_EVAL_EXIT}）"
fi

echo ""
echo "[記憶] 記憶の矛盾候補を検出中（レポートのみ・自動修正なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/detect_shion_memory_contradictions.py" || true

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
        --from-report \
        --include-known-cleanup
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
echo "[反映] ウィザード入力ログ分析 — 空欄率の高いフィールドを台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_wizard_inputs.py" || true

echo ""
echo "[反映] RAG フィードバック分析 — ブースト/ペナルティ候補を台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_rag_feedback.py" || true

echo ""
echo "[反映] RAG未評価通知の自動整理 — 古い/重複 RAG-UNRATED をアーカイブへ..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/cleanup_rag_unrated_rules.py" --apply || true

echo ""
echo "[反映] RAG 鮮度分析 — 長期アクセスなしノードを台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_rag_staleness.py" || true

echo ""
echo "[反映] パイプラインヘルス分析 — 失敗率の高いステップをルール台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_pipeline_health.py"; log_step "analyze_pipeline_health" $?

echo ""
echo "[通知] パイプライン障害検出を Slack へ通知中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/notify_pipeline_alerts.py" --apply; log_step "notify_pipeline_alerts" $?

echo ""
echo "[反映] エラーログ解析 — 頻発エラーをルール台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_error_logs.py" || true

echo ""
echo "[反映] 安全な修正案（紫苑auto・低リスク）を自動で適用待ちへ..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/auto_approve_safe_recipes.py" --apply || true

echo ""
echo "[反映] batch_apply — 台帳ルールを自動適用中..."
"${PYTHON}" "${PROJECT_ROOT}/api/rule_engine/batch_apply.py" --apply; log_step "batch_apply" $?

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
echo "[学習] PDCAルールのライフサイクル管理 — 効果のあるルールを自動延長中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/pdca_rule_lifecycle.py" --apply || true

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

# 改善レポート品質評価
echo ""
echo "[品質] 改善レポート品質スコアを計算中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_improvement_quality.py" || true

# スクリーニングレポート品質フィードバック集計
echo ""
echo "[品質] スクリーニングレポート品質フィードバックを集計中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_report_quality.py" || true

# 改善ループ・係数・モデルの読み取りヘルスチェック
echo ""
echo "[品質] ループ/係数/モデルのヘルスチェックを生成中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/loop_metrics.py" || true

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

# スコアリング重み最適化の自動トリガー
echo ""
echo "[最適化] 30日以上前の未登録ケースを失注補完 → 重み最適化トリガー..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/auto_trigger_optimizer.py" || true

# スコア乖離学習 — 高スコア失注/低スコア成約/高スコア延滞を台帳に追記
echo ""
echo "[学習] スコア乖離パターンを検出して台帳に追記中..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/learn_from_case_differences.py" || true

if [ -f "${LATEST_FILE}" ]; then
    echo ""
    echo "[配布] Gist に最終結果を更新中..."
    if [ ${FINAL_EXIT} -eq 0 ]; then
        # 公開前の機微情報チェック（Cloud Run由来の自由文が混ざるため）。
        # 検出時はGist更新のみスキップし、パイプラインは失敗させない
        if ! "${PYTHON}" "${PROJECT_ROOT}/scripts/check_gist_payload_safety.py" --file "${LATEST_FILE}"; then
            echo "警告: 機微情報の疑いを検出したため Gist 更新をスキップします（ローカル結果は保存済み）"
            log_step "gist_safety_block" 1
        elif command -v gh >/dev/null 2>&1; then
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
