# 共通ヘルパー: パイプラインステップの成否を構造化ログに記録する。
# core / post 両方の日次スクリプトから source して使う。
# 記録先は analyze_pipeline_health.py が集計する data/pipeline_step_log.jsonl。
#
# 使い方:
#   source "${PROJECT_ROOT}/scripts/pipeline_log_step.sh"
#   some_command; log_step "step_name" $?      # 記録するが継続は妨げない
#
# 前提の環境変数:
#   PROJECT_ROOT  ログ出力先の起点
#   LOG_DATE      YYYYMMDD 形式（analyze_pipeline_health の 7日ウィンドウ判定に使用）

log_step() {
    local step_name="$1"
    local exit_code="$2"
    local duration_s="${3:-0}"
    local log_file="${PROJECT_ROOT}/data/pipeline_step_log.jsonl"
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "{\"ts\":\"${ts}\",\"run_date\":\"${LOG_DATE}\",\"step\":\"${step_name}\",\"exit_code\":${exit_code},\"duration_s\":${duration_s}}" >> "${log_file}"
}
