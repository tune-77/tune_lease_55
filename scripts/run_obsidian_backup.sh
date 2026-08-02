#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 既定はプロジェクトの venv。launchd 側の PYTHON_BIN と揃える。
# 旧既定の "/usr/bin/env python3" は exec に単一コマンド名として渡され、
# PYTHON_BIN 未設定時（手動実行時）に必ず失敗していた。
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[run_obsidian_backup] ${PYTHON_BIN} が無いため python3 にフォールバックします" >&2
  PYTHON_BIN="$(command -v python3)"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/backup_obsidian_vault.py" "$@"
