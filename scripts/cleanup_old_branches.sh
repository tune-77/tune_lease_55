#!/bin/bash
# マージ済み auto-improve/* ブランチを削除（local + remote）
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$HOME/Library/Logs/tunelease/branch_cleanup.log"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "ブランチクリーンアップ開始"

cd "$REPO_DIR"

# set -euo pipefail の下では、該当0件の grep が終了コード1でスクリプトを黙って止める
# （2026-08-31〜09-28 は fetch 直後に毎週終了し、launchd の終了コード1だけが残っていた）。
# 0件は正常なので grep には `|| true` を付ける。

# リモートの最新状態を取得（削除済みリモートブランチも刈り取る）
git fetch --prune origin 2>&1 | tee -a "$LOG_FILE" || {
    log "警告: git fetch 失敗。ローカルのマージ済みブランチのみ削除します"
}

# マージ済みリモートブランチを削除
REMOTE_BRANCHES=$(git branch -r --merged origin/master \
    | { grep 'origin/auto-improve/' || true; } \
    | sed 's|origin/||' \
    | tr -d ' ')

if [ -n "$REMOTE_BRANCHES" ]; then
    echo "$REMOTE_BRANCHES" | while IFS= read -r branch; do
        log "リモート削除: $branch"
        git push origin --delete "$branch" 2>&1 | tee -a "$LOG_FILE" || \
            log "警告: リモート削除失敗 ($branch)"
    done
else
    log "削除対象のマージ済みリモートブランチなし"
fi

# ローカルのマージ済みブランチを削除
LOCAL_BRANCHES=$(git branch --merged master \
    | { grep 'auto-improve/' || true; } \
    | tr -d ' ')

if [ -n "$LOCAL_BRANCHES" ]; then
    echo "$LOCAL_BRANCHES" | while IFS= read -r branch; do
        log "ローカル削除: $branch"
        git branch -d "$branch" 2>&1 | tee -a "$LOG_FILE" || \
            log "警告: ローカル削除失敗 ($branch)"
    done
else
    log "削除対象のマージ済みローカルブランチなし"
fi

log "ブランチクリーンアップ完了"
