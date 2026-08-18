#!/usr/bin/env bash
# PreToolUse フック: `eslint --fix` 系コマンドの実行を機械的にブロックする。
#
# 背景: CLAUDE.md / .claude/rules/frontend.md に「eslint --fix 絶対禁止」とあるのは
# 過去に UI コンポーネントや API エンドポイントが誤って削除された事故があったため。
# `npm run lint:fix` は frontend/package.json 側で無効化済みだが、
# `npx eslint --fix` 等を直接叩くと素通りしてしまうため、この hook で塞ぐ。
#
# 方針: これは「絶対禁止」事項のため warn ではなく block（exit 2）する。
# 判定ロジックは同ディレクトリの guard_eslint_fix.py に分離している
# （commit メッセージ等の自由記述テキストを誤検出しないよう、ヒアドキュメント・
# クォート文字列を除去してから判定するため）。
# 対象コマンド以外は何もせず素通しする。

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT="$(cat)"
RESULT="$(printf '%s' "$INPUT" | python3 "$REPO_ROOT/.claude/hooks/guard_eslint_fix.py" 2>/dev/null)"

if [ "$RESULT" = "BLOCK" ]; then
  {
    echo "🚫 ブロック: 'eslint --fix' 系のコマンドは絶対禁止です（CLAUDE.md）。"
    echo "   過去に eslint --fix が UI コンポーネント・API エンドポイントを誤って削除した事故があります。"
    echo "   代わりに: cd frontend && npm run lint でチェックのみ行い、エラーは手動で修正してください。"
  } >&2
  exit 2
fi

exit 0
