# 共通ヘルパー: パイプラインステップの成否を構造化ログに記録する。
# core / post 両方の日次スクリプトから source して使う。
# 記録先は analyze_pipeline_health.py が集計する data/pipeline_step_log.jsonl。
#
# 使い方:
#   source "${PROJECT_ROOT}/scripts/pipeline_log_step.sh"
#   some_command; log_step "step_name" $?      # 記録するが継続は妨げない
#   some_command; log_step "step_name" $? 12   # 所要秒を自分で測った場合は第3引数で渡す
#
# duration_s について:
#   第3引数を省略すると「前回 log_step からの経過秒」を自動で入れる。source 時点を
#   起点にするので最初のステップもスクリプト開始からの経過が入る。
#   これはステップ単体の実行時間ではなく step 間の実時間（echo や未計測の前処理も含む
#   上限値）である。厳密に測りたいステップは呼び出し側で date +%s の差を取り第3引数で渡す
#   （先例: run_daily_improvement_core.sh の memory_chat_regression_tests）。
#   bash 3.2 / BSD date では秒未満が取れないため整数秒で記録する。
#
# 前提の環境変数:
#   PROJECT_ROOT  ログ出力先の起点
#   LOG_DATE      YYYYMMDD 形式（analyze_pipeline_health の評価窓判定に使用。
#                 通常ステップ7日 / 週次ステップ28日）

# 自動計測の起点。source 時点で初期化する（core と post は別プロセスなので各々が独立した起点を持つ）。
_LOG_STEP_LAST_TS="$(date +%s)"

log_step() {
    local step_name="$1"
    local exit_code="$2"
    local duration_s="$3"
    local log_file="${PROJECT_ROOT}/data/pipeline_step_log.jsonl"
    local now_ts
    now_ts="$(date +%s)"

    if [ -z "${duration_s}" ]; then
        duration_s=$(( now_ts - _LOG_STEP_LAST_TS ))
        # 時刻巻き戻し（NTP補正など）で負になった場合は 0 に丸める
        [ "${duration_s}" -lt 0 ] && duration_s=0
    fi
    _LOG_STEP_LAST_TS="${now_ts}"

    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "{\"ts\":\"${ts}\",\"run_date\":\"${LOG_DATE}\",\"step\":\"${step_name}\",\"exit_code\":${exit_code},\"duration_s\":${duration_s}}" >> "${log_file}"
}
