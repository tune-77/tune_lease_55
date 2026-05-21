#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/env python3}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/backup_obsidian_vault.py" "$@"
