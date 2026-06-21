#!/bin/bash
# ハッカソンデモ用 自己改善パイプライン圧縮実行
# 通常24時間かかるサイクルを約3分に圧縮して見せる
# 使い方: bash scripts/demo/run_demo_pipeline.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
SCRIPTS_ROOT="${PROJECT_ROOT}/.agents/skills/auto-improvement-pipeline/scripts"
DEMO_DIR="${PROJECT_ROOT}/scripts/demo"

# ── カラー定義 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║  $1$(printf '%*s' $((52 - ${#1})) '')║${RESET}"
    echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

step_start() {
    echo -e "${YELLOW}${BOLD}▶ $1${RESET}"
}

step_ok() {
    echo -e "${GREEN}${BOLD}✅ $1${RESET}"
}

step_info() {
    echo -e "   ${CYAN}$1${RESET}"
}

# ── 前処理: デモ用の中間ファイルをクリア ─────────────────────────────────
rm -f "${DEMO_DIR}/demo_step1_output.json" \
      "${DEMO_DIR}/demo_step2_output.json" \
      "${DEMO_DIR}/demo_apply_summary.json"

# ── ヘッダー ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  🤖 リース審査AI — 自己改善パイプライン デモ実行${RESET}"
echo -e "${BOLD}  ハッカソン 2026-07-10 デモ用（圧縮実行モード）${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  通常は24時間かけて実行するサイクルを約3分で実演します"
echo -e "  入力: ${DEMO_DIR}/demo_chat_logs.json（5件のリアルな問い合わせ会話）"
echo ""

sleep 3

# ════════════════════════════════════════════════════════════════
banner "STEP 1: チャットログから課題を抽出"
step_start "営業担当・審査担当の会話ログを解析中..."
step_info "入力: demo_chat_logs.json（5件）"
step_info "処理: AIがミスや疑問を構造化データに変換"
echo ""

"${PYTHON}" "${SCRIPTS_ROOT}/step1_extract_and_structure.py" --demo

step_ok "Step 1 完了 — 課題を構造化データとして抽出しました"
echo ""
echo -e "  ${CYAN}↳ 「ブルドーザー 5年設定」「太陽光パネル未登録」などを検出${RESET}"

sleep 5

# ════════════════════════════════════════════════════════════════
banner "STEP 2: 改善案の妥当性を自動検証"
step_start "各改善案について根拠・影響範囲を検証中..."
step_info "チェック: データソース確認・修正値の整合性・適用範囲"
step_info "判定: APPROVED / REJECTED を自動決定"
echo ""

"${PYTHON}" "${SCRIPTS_ROOT}/step2_validation_checker.py" --demo

step_ok "Step 2 完了 — 妥当性検証・承認判定が完了しました"
echo ""
echo -e "  ${CYAN}↳ 国税庁データ根拠があるものは自動 APPROVED${RESET}"

sleep 5

# ════════════════════════════════════════════════════════════════
banner "STEP 3: 承認済み改善案を自動適用"
step_start "APPROVED 案件をシステムに自動反映中..."
step_info "書き込み先: demo_ledger.jsonl（本番台帳は変更しません）"
step_info "スコアリング重要ファイルは手動レビューキューへ振り分け"
echo ""

"${PYTHON}" "${SCRIPTS_ROOT}/step3_auto_apply.py" --demo

step_ok "Step 3 完了 — デモ台帳に改善を記録しました"

sleep 5

# ════════════════════════════════════════════════════════════════
banner "Before / After 審査比較"
step_start "ブルドーザーリース案件（申込: 5年）で改善効果を確認..."
echo ""

"${PYTHON}" "${DEMO_DIR}/show_before_after.py"

echo ""
step_ok "デモパイプライン 全ステップ完了！"

# ── フッター ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  📊 デモ実行サマリー${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

if command -v python3 &>/dev/null && [ -f "${DEMO_DIR}/demo_apply_summary.json" ]; then
    "${PYTHON}" - <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("scripts/demo/demo_apply_summary.json")
if not p.exists():
    sys.exit(0)
d = json.loads(p.read_text())
print(f"  ✅ 自動適用: {d['applied_count']} 件")
for a in d['applied']:
    print(f"     {a['id']}: {a['title']}")
    print(f"        └ {a.get('change','')}")
print(f"  👀 要レビュー: {d['needs_review_count']} 件")
for n in d['needs_review']:
    print(f"     {n['id']}: {n['title']}")
PYEOF
fi

echo ""
echo -e "  ${CYAN}生成されたファイル:${RESET}"
echo -e "  • ${DEMO_DIR}/demo_step1_output.json    — 抽出された改善案"
echo -e "  • ${DEMO_DIR}/demo_step2_output.json    — 検証・承認結果"
echo -e "  • ${DEMO_DIR}/demo_ledger.jsonl         — デモ用適用台帳"
echo -e "  • ${DEMO_DIR}/demo_apply_summary.json   — 適用サマリー"
echo ""
echo -e "${GREEN}${BOLD}🎉 デモ完了！ REV-139 自己改善パイプライン デモモード${RESET}"
echo ""
