#!/bin/bash
# マージ済み auto-improve/* ブランチを削除（local + remote）
set -euo pipefail

REPO_DIR="/Users/kobayashiisaoryou/clawd/tune_lease_55"
LOG_FILE="$HOME/Library/Logs/tunelease/branch_cleanup.log"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "ブランチクリーンアップ開始"

cd "$REPO_DIR"

# リモートの最新状態を取得（削除済みリモートブランチも刈り取る）
git fetch --prune origin 2>&1 | tee -a "$LOG_FILE" || {
    log "警告: git fetch 失敗。ローカルのマージ済みブランチのみ削除します"
}

# マージ済みリモートブランチを削除
REMOTE_BRANCHES=$(git branch -r --merged origin/master \
    | grep 'origin/auto-improve/' \
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
    | grep 'auto-improve/' \
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
