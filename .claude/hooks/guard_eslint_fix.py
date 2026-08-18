#!/usr/bin/env python3
"""guard_eslint_fix_hook.sh から呼ばれる判定ロジック。

stdin から PreToolUse フックの JSON を受け取り、実際にコマンドとして
実行される部分に eslint --fix 系のパターンがあれば "BLOCK"、
なければ "OK" を stdout に出力する。

ヒアドキュメント本文やクォート文字列（コミットメッセージ等の自由記述
テキスト）は実行対象ではないため、判定前に除去して誤検出を防ぐ。
"""
import json
import re
import sys


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        print("OK")
        return

    tool_input = data.get("tool_input") or {}
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    if not command:
        print("OK")
        return

    # ヒアドキュメント本文（<<EOF ... EOF / <<'EOF' ... EOF）を除去。
    stripped = re.sub(
        r"<<-?\s*['\"]?(\w+)['\"]?\r?\n.*?\r?\n\1\b",
        " ",
        command,
        flags=re.S,
    )

    # シングル/ダブルクォート文字列の中身も除去（-m "..." 等の自由記述テキスト対策）。
    stripped = re.sub(r"'[^']*'", " ", stripped)
    stripped = re.sub(r'"[^"]*"', " ", stripped)

    pattern = re.compile(
        r"(eslint[^&|;\n]*--fix|eslint:fix|next\s+lint\s+--fix)",
        re.IGNORECASE,
    )
    print("BLOCK" if pattern.search(stripped) else "OK")


if __name__ == "__main__":
    main()
