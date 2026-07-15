#!/bin/bash
# 日次改善の後処理
# 補助記録と共有向けの更新だけをまとめる

PROJECT_ROOT="${PROJECT_ROOT:-/Users/kobayashiisaoryou/clawd/tune_lease_55}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
PIPELINE_DATE="${PIPELINE_DATE:-$(date +%F)}"
MANA_MAX_REFLECTION_REPAIRS="${MANA_MAX_REFLECTION_REPAIRS:-3}"
DEFAULT_OBSIDIAN_VAULT="/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
REFLECTION_VAULT="${OBSIDIAN_VAULT:-${OBSIDIAN_VAULT_PATH:-${DEFAULT_OBSIDIAN_VAULT}}}"
REFLECTION_DIR="${REFLECTION_VAULT}/Projects/tune_lease_55/Lease Intelligence/Private Reflection"
MANA_REPORT_JSON="${PROJECT_ROOT}/reports/mana_obsidian_curator_latest.json"
SCREENING_TERMS_REPORT_JSON="${PROJECT_ROOT}/reports/screening_terms_audit_latest.json"

build_reflection_delta() {
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_reflection_delta.py" \
    --date "${PIPELINE_DATE}" \
    --reflection-dir "${REFLECTION_DIR}" || true
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
"${PYTHON}" "${PROJECT_ROOT}/scripts/rotate_weekly_focus.py" || true

echo ""
echo "[記録] e-Stat業種別統計更新..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/fetch_estat_industry.py" || true

echo ""
echo "[補助] Sidecar Agent Brief を生成（読み取り専用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/agent_sidecar_reader.py" || true

echo ""
echo "[記録] DAILY-BRIEF.md を Obsidian Vault に書き出し..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/write_daily_brief.py" || true

echo ""
echo "[監視] Obsidian環境モニターを生成（読み取り専用）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/monitor_obsidian_environment.py" \
  --date "${PIPELINE_DATE}" || true

echo ""
echo "[記録] Cloud Run入力ログを取得（GCS → ローカル）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_cloudrun_inputs_from_gcs.py" || true

echo ""
echo "[記録] Cloud Run会話ログをObsidianへ同期..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_cloudrun_inputs_to_obsidian.py" || true

echo ""
echo "[記録] memory/ から MEMORY.md へ長期記憶を自動昇格..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_memory_from_daily.py" || true

echo ""
echo "[補助] 週次セルフマネジメントサマリ（月曜のみ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/weekly_self_management.py" || true

echo ""
echo "[内省] 紫苑の日次私的内省を生成（当日対話/内省材料 → Private Reflection）..."
"${PYTHON}" "${PROJECT_ROOT}/lease_intelligence_reflection.py" || true

echo ""
echo "[内省] 内省差分レポートを生成（読み取り専用・未連携）..."
build_reflection_delta

echo ""
echo "[記憶] 会話ログから記憶昇格候補キューを生成（承認待ち・自動昇格なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_memory_promotion_queue.py" || true

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
    build_reflection_delta
    continue
  fi
  break
done

echo ""
echo "[司書] Obsidian Curator レポートを生成（読み取り専用・未連携）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/obsidian_curator_report.py" || true

echo ""
echo "[成長] Judgment Asset Growth Score を記録（ローカル履歴・未連携）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/judgment_asset_growth_report.py" \
  --date "${PIPELINE_DATE}" || true

echo ""
echo "[監査] 審査用語監査を生成（読み取り専用・未連携）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/screening_terms_audit.py" || true

echo ""
echo "[通知] 日次改善レポートをSlackへ送信（Mana判定込み・Webhook未設定ならスキップ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/send_daily_improvement_slack.py" \
  --date "${PIPELINE_DATE}" \
  --mana-report "${MANA_REPORT_JSON}" \
  --screening-terms-report "${SCREENING_TERMS_REPORT_JSON}" || true

if [ "${MANA_STATUS}" != "allow" ]; then
  echo "[番人] Mana が allow ではないため、評価候補生成と GCS Vault 配布を停止します。"
  echo "[番人] レポート確認: ${MANA_REPORT_JSON}"
  exit 0
fi

echo ""
echo "[記憶] 評価セット候補を実クエリから生成（毎月1日のみ実行）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_eval_candidates.py" || true

echo ""
echo "[配布] 公開ノート（Memory Pack等）を GCS Vault へアップロード..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/icloud_to_gcs_sync.py" || true
