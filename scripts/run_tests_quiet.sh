#!/usr/bin/env bash
# テスト実行ラッパー
# 成功時: サマリーのみ出力（大量のドット/ログでトークンを消費しない）
# 失敗時: 失敗したテストだけ --lf で再実行し、詳細（-v --tb=short）を出力
set -uo pipefail

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

python3 -m pytest tests/ -q --tb=no "$@" >"$LOG" 2>&1
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
  tail -n 5 "$LOG"
  exit 0
fi

echo "=== テスト失敗を検出。失敗したテストのみ詳細再実行 ==="
tail -n 15 "$LOG"
echo
python3 -m pytest tests/ --lf -v --tb=short --color=yes "$@"
exit $?
