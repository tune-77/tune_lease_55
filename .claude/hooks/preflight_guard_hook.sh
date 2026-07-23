#!/usr/bin/env bash
# PreToolUse フック: git push / gh pr create の直前にプリフライト検証ガードを実行する。
#
# 方針は warn-only。警告があっても **決してブロックしない**（常に exit 0）。
# stdin には Claude Code の tool 入力 JSON（tool_name, tool_input.command …）が渡る。
# 対象コマンド（git push / gh pr create）以外は何もせず素通しする。
#
# 中身はスタンドアロン CLI scripts/preflight_pr_guard.py を呼ぶだけの薄いラッパ。

set -u

# フック位置（.claude/hooks/）から見たリポジトリルート
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT="$(cat)"

# tool 入力 JSON から実行コマンド文字列を取り出す（jq 非依存で python 解析）。
COMMAND="$(
  printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
ti = data.get("tool_input") or {}
print(ti.get("command", "") if isinstance(ti, dict) else "")
' 2>/dev/null
)"

# 対象は git push / gh pr create のみ。それ以外は素通し。
case "$COMMAND" in
  *"git push"*|*"gh pr create"*) ;;
  *) exit 0 ;;
esac

# 警告のみ。CLI は既定で exit 0（--strict は付けない）。出力は stderr に流して可視化する。
python3 "$REPO_ROOT/scripts/preflight_pr_guard.py" 1>&2 || true

exit 0
