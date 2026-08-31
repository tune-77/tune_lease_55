#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 既定はプロジェクトの venv。launchd 側の PYTHON_BIN と揃える。
# 旧既定の "/usr/bin/env python3" は exec に単一コマンド名として渡され、
# PYTHON_BIN 未設定時（手動実行時）に必ず失敗していた。
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[run_lease_news_collection] ${PYTHON_BIN} が無いため python3 にフォールバックします" >&2
  PYTHON_BIN="$(command -v python3)"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/collect_lease_news_to_obsidian.py" "$@"

# 収集直後にGCSへ同期する。日次改善パイプライン（4時）の同期を待つと、
# その後6時に収集した当日分がCloud Runへ反映されるまで最大1日遅れるため、
# ここで前倒しして同期する。失敗してもニュース収集自体の成否には影響させない
# （次回の日次パイプラインの同期ステップで再試行される）。
"$PYTHON_BIN" "$SCRIPT_DIR/icloud_to_gcs_sync.py" || echo "[run_lease_news_collection] GCS同期に失敗しました" >&2
