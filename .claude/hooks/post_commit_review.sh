#!/bin/bash
# post_commit_review.sh
# git commit 後にバックグラウンドでコードレビュー・セキュリティチェックを実行する。
# .claude/settings.json の PostToolUse フックから呼び出される。

REPO="/home/user/tune_lease_55"

# 直前のコミットで変更された Python ファイルを取得（最大10件）
FILES=$(git -C "$REPO" diff HEAD~1 --name-only 2>/dev/null | grep '\.py$' | head -10 | tr '\n' ' ')

# Python ファイルの変更がなければ終了
if [ -z "$FILES" ]; then
  exit 0
fi

LOG="/tmp/post_commit_review_$(date +%Y%m%d_%H%M%S).log"
{
  echo "=== Post-commit Review: $(date) ==="
  echo "対象ファイル: $FILES"
  echo ""
} > "$LOG"

# claude CLI が利用可能な場合のみ実行
if command -v claude &>/dev/null; then
  cd "$REPO" && claude -p "$(cat <<PROMPT
以下のPythonファイルを含むコミットを対象に、コードレビューとセキュリティチェックを実施してください。
問題点があれば日本語で簡潔に報告してください（深刻な問題を優先）。

対象ファイル: $FILES

チェック観点:
1. バグリスク（型エラー・例外処理漏れ・ゼロ除算等）
2. セキュリティ問題（インジェクション・機密情報露出等）
3. コード品質（重複・可読性・保守性）
PROMPT
)" \
    --output-format text \
    >> "$LOG" 2>&1
fi

echo "Post-commit review 完了: $LOG" >&2
