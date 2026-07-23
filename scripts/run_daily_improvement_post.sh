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
echo "[成長] Judgment Asset Growth Score を記録（Slack日次レポート / loop-proof が参照）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/judgment_asset_growth_report.py" \
  --date "${PIPELINE_DATE}"
log_step "judgment_asset_growth_report" $?

echo ""
echo "[成長] 紫苑の期間成長判定を生成（判断資産グラフのみが参照する末端）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_shion_growth.py" \
  --end-date "${PIPELINE_DATE}"
log_step "evaluate_shion_growth" $?

echo ""
echo "[可視化] 判断資産グラフを生成（末端: 生成HTML/JSONを読むコードは無し。手動閲覧しなければ純コスト → 棚卸し候補）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_judgment_asset_graph.py"; log_step "build_judgment_asset_graph" $?

echo ""
echo "[配線] 判断資産グラフを frontend/public へ同期（本番UI配信用）..."
# reports/ は Cloud Run イメージ非同梱だが frontend/public は本番配信される。
# 生成した最新グラフ(HTML/PNG)を public にコピーし、次回デプロイで /judgment-asset-graph に反映させる。
GRAPH_PUBLIC_DIR="${PROJECT_ROOT}/frontend/public/judgment-asset-graph"
mkdir -p "${GRAPH_PUBLIC_DIR}"
cp -f "${PROJECT_ROOT}/reports/judgment_asset_graph_latest.html" "${GRAPH_PUBLIC_DIR}/index.html" \
  && cp -f "${PROJECT_ROOT}/reports/judgment_asset_graph_latest.png" "${GRAPH_PUBLIC_DIR}/preview.png"
log_step "sync_graph_to_public" $?

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
echo "[監査] 二重台帳（リポジトリ/ランタイム）の整合性チェック（読み取り専用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_ledger_consistency.py" --days 14; log_step "check_ledger_consistency" $?

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
  --screening-terms-report "${SCREENING_TERMS_REPORT_JSON}"
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
