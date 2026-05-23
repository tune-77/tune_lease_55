#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/env python3}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/collect_lease_news_to_obsidian.py" "$@"
