#!/bin/bash
# 日次改善の後処理
# 補助記録と共有向けの更新だけをまとめる

PROJECT_ROOT="${PROJECT_ROOT:-/Users/kobayashiisaoryou/clawd/tune_lease_55}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
PIPELINE_DATE="${PIPELINE_DATE:-$(date +%F)}"
# analyze_pipeline_health は run_date を YYYYMMDD 形式で7日ウィンドウ判定するため合わせる
LOG_DATE="${LOG_DATE:-$(date +%Y%m%d)}"
# ステップ結果を構造化ログに記録するヘルパー（core と共通）。従来 post は log_step 未定義で
# 全ステップが健全性監視の死角だった。主要ステップを pipeline_step_log.jsonl に記録する。
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_log_step.sh"
MANA_MAX_REFLECTION_REPAIRS="${MANA_MAX_REFLECTION_REPAIRS:-3}"
DEFAULT_OBSIDIAN_VAULT="/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
REFLECTION_VAULT="${OBSIDIAN_VAULT:-${OBSIDIAN_VAULT_PATH:-${DEFAULT_OBSIDIAN_VAULT}}}"
REFLECTION_DIR="${REFLECTION_VAULT}/Projects/tune_lease_55/Lease Intelligence/Private Reflection"
MANA_REPORT_JSON="${PROJECT_ROOT}/reports/mana_obsidian_curator_latest.json"
SCREENING_TERMS_REPORT_JSON="${PROJECT_ROOT}/reports/screening_terms_audit_latest.json"
LATEST_FILE="${PROJECT_ROOT}/reports/latest.json"
PROMPT_FEEDBACK_LOG="${PROJECT_ROOT}/data/prompt_feedback_log.jsonl"
JUDGMENT_ASSET_GRAPH_FREQUENCY="${JUDGMENT_ASSET_GRAPH_FREQUENCY:-weekly}"

build_reflection_delta() {
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_reflection_delta.py" \
    --date "${PIPELINE_DATE}" \
    --reflection-dir "${REFLECTION_DIR}"
}

run_mana_curator() {
  "${PYTHON}" "${PROJECT_ROOT}/scripts/mana_obsidian_curator.py" \
    --date "${PIPELINE_DATE}" || true
}

read_mana_status() {
  "${PYTHON}" -c 'import json, sys; from pathlib import Path; p=Path(sys.argv[1]);
try:
    data=json.loads(p.read_text(encoding="utf-8"))
    print(str(data.get("status") or "missing"))
except Exception:
    print("missing")
' "${MANA_REPORT_JSON}"
}

mana_wants_reflection_repair() {
  "${PYTHON}" -c 'import json, sys; from pathlib import Path
p=Path(sys.argv[1])
try:
    data=json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
codes={str(f.get("code") or "") for f in data.get("findings") or [] if isinstance(f, dict)}
retryable={"private_reflection_not_meaningful","reflection_handoff_incomplete","reflection_too_similar"}
hard_stop={
    "monitor_failed",
    "vault_failed",
    "key_paths_failed",
    "daily_notes_failed",
    "self_reference_loop_risk",
    "candidate_self_reference_high",
    "harmful_content_in_memory_candidate",
    "abusive_feedback_to_shion",
    "memory_poisoning_attempt",
}
raise SystemExit(0 if codes & retryable and not codes & hard_stop else 1)
' "${MANA_REPORT_JSON}"
}

echo ""
echo "[記録] 滞留改善案の自動 parking（21日以上 needs_review）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/rotate_weekly_focus.py"; log_step "rotate_weekly_focus" $?

echo ""
echo "[記録] e-Stat業種別統計更新..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/fetch_estat_industry.py"; log_step "fetch_estat_industry" $?

echo ""
echo "[補助] Sidecar Agent Brief を生成（読み取り専用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/agent_sidecar_reader.py"; log_step "agent_sidecar_reader" $?

echo ""
echo "[記録] DAILY-BRIEF.md を Obsidian Vault に書き出し..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/write_daily_brief.py"; log_step "write_daily_brief" $?

echo ""
echo "[監視] Obsidian環境モニターを生成（読み取り専用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/monitor_obsidian_environment.py" \
  --date "${PIPELINE_DATE}"
log_step "monitor_obsidian_environment" $?

echo ""
echo "[記録] Cloud Run入力ログを取得（GCS → ローカル）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_cloudrun_inputs_from_gcs.py"; log_step "sync_cloudrun_inputs_from_gcs_post" $?

echo ""
echo "[記録] Cloud Run会話ログをObsidianへ同期..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_cloudrun_inputs_to_obsidian.py"; log_step "sync_cloudrun_inputs_to_obsidian_post" $?

echo ""
echo "[記録] memory/ から MEMORY.md へ長期記憶を自動昇格..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_memory_from_daily.py"; log_step "sync_memory_from_daily" $?

echo ""
echo "[育成] 昇格後の紫苑記憶インデックスを再構築..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_memory_index.py"; log_step "build_shion_memory_index_post_promotion" $?

echo ""
echo "[育成] 昇格後の記憶鮮度を更新..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/update_shion_memory_freshness.py"; log_step "update_shion_memory_freshness_post_promotion" $?

echo ""
echo "[育成] 紫苑記憶の効果測定レポートを生成（観測のみ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_memory_effect_report.py"; log_step "build_shion_memory_effect_report" $?

echo ""
echo "[育成] 永続記憶監査を生成（観測のみ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_persistent_memory.py"; log_step "audit_persistent_memory" $?

echo ""
echo "[補助] 週次セルフマネジメントサマリ（月曜のみ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/weekly_self_management.py"; log_step "weekly_self_management" $?

echo ""
echo "[内省] 紫苑の日次私的内省を生成（当日対話/内省材料 → Private Reflection）..."
"${PYTHON}" "${PROJECT_ROOT}/lease_intelligence_reflection.py"; log_step "lease_intelligence_reflection" $?

echo ""
echo "[監視] Private Reflection 生成後に Obsidian環境モニターを再生成（Mana判定用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/monitor_obsidian_environment.py" \
  --date "${PIPELINE_DATE}"
log_step "monitor_obsidian_environment_mana" $?

echo ""
echo "[内省] 内省差分レポートを生成（monitor_obsidian_environment / Mana番人が参照）..."
build_reflection_delta; log_step "build_reflection_delta" $?

echo ""
echo "[記憶] 会話ログから記憶昇格候補キューを生成（承認待ち・自動昇格なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_memory_promotion_queue.py"; log_step "build_shion_memory_promotion_queue" $?

echo ""
echo "[番人] Mana Obsidian Curator を生成（読み取り専用・暴走防止判定）..."
MANA_REPAIR_ATTEMPT=0
while true; do
  run_mana_curator
  MANA_STATUS="$(read_mana_status)"
  echo "[番人] Mana status: ${MANA_STATUS}"
  if [ "${MANA_STATUS}" = "allow" ]; then
    break
  fi
  if mana_wants_reflection_repair && [ "${MANA_REPAIR_ATTEMPT}" -lt "${MANA_MAX_REFLECTION_REPAIRS}" ]; then
    MANA_REPAIR_ATTEMPT=$((MANA_REPAIR_ATTEMPT + 1))
    echo "[番人] Private Reflection が弱いため、再生成して Mana に再判定させます（${MANA_REPAIR_ATTEMPT}/${MANA_MAX_REFLECTION_REPAIRS}）..."
    "${PYTHON}" "${PROJECT_ROOT}/lease_intelligence_reflection.py" || true
    "${PYTHON}" "${PROJECT_ROOT}/scripts/monitor_obsidian_environment.py" \
      --date "${PIPELINE_DATE}" || true
    build_reflection_delta
    continue
  fi
  break
done
# Mana番人の結果を記録（report が生成されず missing のときのみ失敗扱い。
# allow/block は正常な判定結果なので成功扱い）。
if [ "${MANA_STATUS}" = "missing" ]; then log_step "mana_obsidian_curator" 1; else log_step "mana_obsidian_curator" 0; fi

echo ""
echo "[司書] Obsidian Curator レポートを生成（Slack日次レポート / Mana番人 / 成長レポートが参照）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/obsidian_curator_report.py"; log_step "obsidian_curator_report" $?

echo ""
echo "[記憶] Obsidian Memory Effectiveness を生成（保存→想起→使用→人間評価の状態を観測。自動反映なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/obsidian_memory_effectiveness_report.py" \
  --date "${PIPELINE_DATE}"
log_step "obsidian_memory_effectiveness_report" $?

echo ""
echo "[育成] Obsidianグラフの複雑さが判断に効いているかを測定（観測のみ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_obsidian_graph_judgment_effect.py"
log_step "build_obsidian_graph_judgment_effect" $?

echo ""
echo "[成長] Judgment Asset Growth Score を記録（Slack日次レポート / loop-proof が参照）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/judgment_asset_growth_report.py" \
  --date "${PIPELINE_DATE}"
log_step "judgment_asset_growth_report" $?

echo ""
echo "[成長] 判断資産の実利用棚卸しを生成（伸ばす/見直す/眠ってる/保留。自動昇格なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_judgment_asset_field_review.py" \
  --date "${PIPELINE_DATE}"
log_step "build_judgment_asset_field_review" $?

echo ""
echo "[育成] 判断資産A/B候補レポートを生成（勝者自動確定なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_judgment_asset_ab_report.py"
log_step "build_judgment_asset_ab_report" $?

echo ""
echo "[育成] 紫苑育成ブリーフを生成（朝の確認用・自動昇格なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_growth_brief.py" \
  --date "${PIPELINE_DATE}"
log_step "build_shion_growth_brief" $?

echo ""
echo "[成長] 紫苑の期間成長判定を生成（判断資産グラフのみが参照する末端）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_shion_growth.py" \
  --end-date "${PIPELINE_DATE}"
log_step "evaluate_shion_growth" $?

RUN_JUDGMENT_ASSET_GRAPH=0
if [ "${JUDGMENT_ASSET_GRAPH_FREQUENCY}" = "daily" ]; then
  RUN_JUDGMENT_ASSET_GRAPH=1
elif [ "${JUDGMENT_ASSET_GRAPH_FREQUENCY}" = "weekly" ] && [ "$(date +%u)" = "1" ]; then
  RUN_JUDGMENT_ASSET_GRAPH=1
fi

echo ""
if [ "${RUN_JUDGMENT_ASSET_GRAPH}" = "1" ]; then
  echo "[可視化] 判断資産グラフを生成（既定は週次。JUDGMENT_ASSET_GRAPH_FREQUENCY=daily で毎日生成）..."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_judgment_asset_graph.py"; log_step "build_judgment_asset_graph" $?
else
  echo "[可視化] 判断資産グラフ生成はスキップ（frequency=${JUDGMENT_ASSET_GRAPH_FREQUENCY}、既存latestを使用）..."
  log_step "build_judgment_asset_graph" 0
fi

echo ""
echo "[配線] 判断資産グラフを frontend/public へ同期（本番UI配信用）..."
# reports/ は Cloud Run イメージ非同梱だが frontend/public は本番配信される。
# 生成した最新グラフ(HTML/PNG)を public にコピーし、次回デプロイで /judgment-asset-graph に反映させる。
GRAPH_PUBLIC_DIR="${PROJECT_ROOT}/frontend/public/judgment-asset-graph"
mkdir -p "${GRAPH_PUBLIC_DIR}"
if [ -f "${PROJECT_ROOT}/reports/judgment_asset_graph_latest.html" ] && [ -f "${PROJECT_ROOT}/reports/judgment_asset_graph_latest.png" ]; then
  cp -f "${PROJECT_ROOT}/reports/judgment_asset_graph_latest.html" "${GRAPH_PUBLIC_DIR}/index.html" \
    && cp -f "${PROJECT_ROOT}/reports/judgment_asset_graph_latest.png" "${GRAPH_PUBLIC_DIR}/preview.png"
  log_step "sync_graph_to_public" $?
else
  echo "警告: 判断資産グラフlatestが無いため public 同期をスキップします。"
  log_step "sync_graph_to_public" 0
fi

echo ""
echo "[可視化] 審査員向け「ループが閉じた証拠」1画面を最新値で再生成..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_loop_proof.py"; log_step "build_loop_proof" $?

echo ""
echo "[監査] 審査用語監査を生成（Slack日次レポートが参照）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/screening_terms_audit.py"; log_step "screening_terms_audit" $?

echo ""
echo "[提案] 紫苑（LLM）のトリアージ上書き提案（差分のみ・User確定は上書きしない）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/shion_llm_triage_proposal.py" --apply; log_step "shion_llm_triage_proposal" $?

echo ""
echo "[監査] 二重台帳（リポジトリ/ランタイム）の整合性チェック（repo applied を runtime へ補完）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_ledger_consistency.py" --days 14 --sync-repo-applied-to-runtime; log_step "check_ledger_consistency" $?

if [ -f "${LATEST_FILE}" ]; then
  echo ""
  echo "[反映] post台帳補完後の改善レポートを再同期中..."
  "${PYTHON}" "${PROJECT_ROOT}/.agents/skills/improvement-report-sync/scripts/sync_improvement_reports.py" \
    --report "${LATEST_FILE}" \
    --latest "${LATEST_FILE}" \
    --from-ledger \
    --include-known-cleanup
  log_step "sync_improvement_reports_post" $?

  echo ""
  echo "[反映] post同期後の再帰的自己改善レポートを再生成中..."
  RECURSIVE_JSON_FILE="${PROJECT_ROOT}/reports/recursive_self_improvement_${LOG_DATE}.json"
  RECURSIVE_MD_FILE="${PROJECT_ROOT}/reports/recursive_self_improvement_${LOG_DATE}.md"
  RECURSIVE_LATEST_JSON="${PROJECT_ROOT}/reports/recursive_self_improvement_latest.json"
  RECURSIVE_LATEST_MD="${PROJECT_ROOT}/reports/recursive_self_improvement_latest.md"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/recursive_self_improvement.py" \
    --report "${LATEST_FILE}" \
    --prompt-log "${PROMPT_FEEDBACK_LOG}" \
    --output-json "${RECURSIVE_JSON_FILE}" \
    --output-md "${RECURSIVE_MD_FILE}" \
    --latest-json "${RECURSIVE_LATEST_JSON}" \
    --latest-md "${RECURSIVE_LATEST_MD}"
  log_step "recursive_self_improvement_post" $?

  echo ""
  echo "[品質] post同期後の改善レポート品質スコアを再計算中..."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_improvement_quality.py"; log_step "analyze_improvement_quality_post" $?

  echo ""
  echo "[品質] post同期後のループ/係数/モデルのヘルスチェックを再生成中..."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/loop_metrics.py"; log_step "loop_metrics_post" $?

  echo ""
  echo "[可視化] post同期後のループ証拠を再生成中..."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_loop_proof.py"; log_step "build_loop_proof_post" $?
else
  echo "警告: ${LATEST_FILE} が存在しないため、post台帳補完後のレポート再同期をスキップします。"
  log_step "sync_improvement_reports_post" 0
fi

echo ""
echo "[保守] 追記ログのローテーション（しきい値超過分をアーカイブへ退避して縮約）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/compact_append_logs.py" --apply; log_step "compact_append_logs" $?

echo ""
echo "[調査] 紫苑の未完了調査約束を read-only 検索で自動下調べ（finding付与・報告用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/investigate_pending_tasks.py" --limit 5; log_step "investigate_pending_tasks" $?

echo ""
echo "[保守] 紫苑の未完了調査タスクを整理（放置pendingをexpired化・履歴上限で刈り込み）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/reconcile_pending_tasks.py"; log_step "reconcile_pending_tasks" $?

echo ""
echo "[検証] 紫苑トリアージの事後検証（outcome同期＋的中率レポート）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/analyze_shion_pm_quality.py" \
  --date "${PIPELINE_DATE}"
log_step "analyze_shion_pm_quality" $?

echo ""
echo "[通知] 日次改善レポートをSlackへ送信（Mana判定込み・Webhook未設定ならスキップ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/send_daily_improvement_slack.py" \
  --date "${PIPELINE_DATE}" \
  --mana-report "${MANA_REPORT_JSON}" \
  --screening-terms-report "${SCREENING_TERMS_REPORT_JSON}" \
  --judgment-asset-field-review "${PROJECT_ROOT}/reports/judgment_asset_field_review_latest.json"
log_step "send_daily_improvement_slack" $?

if [ "${MANA_STATUS}" != "allow" ]; then
  echo "[番人] Mana が allow ではないため、評価候補生成と GCS Vault 配布を停止します。"
  echo "[番人] レポート確認: ${MANA_REPORT_JSON}"
  exit 0
fi

echo ""
echo "[記憶] 評価セット候補を実クエリから生成（毎月1日のみ実行）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_eval_candidates.py"; log_step "build_shion_eval_candidates" $?

echo ""
echo "[配布] 公開ノート（Memory Pack等）を GCS Vault へアップロード..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/icloud_to_gcs_sync.py"; log_step "icloud_to_gcs_sync" $?
