#!/bin/bash
# GenAI App Builder クレジット期間の Vertex AI Search 日次処理（launchd: com.tunelease.vertex-credit-daily、05:30）
# 1. 利用額の見張り（モードに関係なく毎日。90%で朝報警告・100%で自動 off）
# 2. Jev の並べ替え shadow 評価（Vertexクレジットとは独立して毎日）
# 3. モードが on の時だけ: Obsidian/判断資産→データストア同期、ChromaDB と Vertex の品質比較
# off（VERTEX_CREDIT_MODE=off・期限 2027-02-01 以降・自動 off）なら Vertex処理だけ止まり、Jev評価は継続する。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
LOG_DATE="$(date +%Y%m%d)"
# Jev（TypeSafe）の鍵はキーチェーンから読む（core.sh と同じサービス名）。未設定だと Jev 並べ替えの評価が全問不通になる
TYPESAFE_API_KEYCHAIN_SERVICE="${TYPESAFE_API_KEYCHAIN_SERVICE:-typesafe-api-key}"
export PROJECT_ROOT PYTHON LOG_DATE TYPESAFE_API_KEYCHAIN_SERVICE
cd "${PROJECT_ROOT}" || exit 1
source "${PROJECT_ROOT}/scripts/pipeline_log_step.sh"

echo "==== Vertex クレジット日次処理: $(date '+%Y-%m-%d %H:%M:%S') ===="
"${PYTHON}" "${PROJECT_ROOT}/scripts/vertex_credit_monitor.py"
log_step "vertex_credit_monitor" $?

if ! "${PYTHON}" -c "import sys; from api.vertex_credit_mode import credit_mode_status as s; r = s(); print('VERTEX_CREDIT_MODE:', r['reason']); sys.exit(0 if r['active'] else 1)"; then
    echo "クレジットモード off: Jev単独評価を実行し、Vertex同期と品質比較はスキップ"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/eval_vertex_vs_chroma.py" --jev-only
    log_step "eval_jev_vs_chroma" $?
    exit 0
fi

# FULL＋GCS の不要オブジェクト削除で、データストアをエクスポート（除外・private・重複整理済み）と完全一致させる。
# INCREMENTAL だと、後から private にしたノートや統合で外れたノートが Vertex に残り続ける。
"${PYTHON}" "${PROJECT_ROOT}/scripts/sync_obsidian_to_vertex_agent_search.py" \
    --upload --import-documents --wait --reconciliation-mode FULL --delete-stale-gcs
log_step "vertex_search_sync" $?

"${PYTHON}" "${PROJECT_ROOT}/scripts/eval_vertex_vs_chroma.py"
log_step "eval_vertex_vs_chroma" $?
