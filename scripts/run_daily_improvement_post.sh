#!/bin/bash
# 日次改善の後処理
# 補助記録と共有向けの更新だけをまとめる

PROJECT_ROOT="${PROJECT_ROOT:-/Users/kobayashiisaoryou/clawd/tune_lease_55}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

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
  --date "$(date +%F)" || true

echo ""
echo "[通知] 日次改善レポートをSlackへ送信（Webhook未設定ならスキップ）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/send_daily_improvement_slack.py" \
  --date "$(date +%F)" || true

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
DEFAULT_OBSIDIAN_VAULT="/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
REFLECTION_VAULT="${OBSIDIAN_VAULT:-${OBSIDIAN_VAULT_PATH:-${DEFAULT_OBSIDIAN_VAULT}}}"
REFLECTION_DIR="${REFLECTION_VAULT}/Projects/tune_lease_55/Lease Intelligence/Private Reflection"
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_reflection_delta.py" \
  --date "$(date +%F)" \
  --reflection-dir "${REFLECTION_DIR}" || true

echo ""
echo "[記憶] 会話ログから記憶昇格候補キューを生成（承認待ち・自動昇格なし）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_memory_promotion_queue.py" || true

echo ""
echo "[番人] Mana Obsidian Curator を生成（読み取り専用・暴走防止判定）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/mana_obsidian_curator.py" \
  --date "$(date +%F)" || true

echo ""
echo "[記憶] 評価セット候補を実クエリから生成（毎月1日のみ実行）..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/build_shion_eval_candidates.py" || true

echo ""
echo "[配布] 公開ノート（Memory Pack等）を GCS Vault へアップロード..."
"${PYTHON}" "${PROJECT_ROOT}/scripts/icloud_to_gcs_sync.py" || true
