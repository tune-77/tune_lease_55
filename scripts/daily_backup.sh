#!/bin/bash
# 毎日の定期バックアップスクリプト（アプリ未起動時の保険）
#
# crontab への登録例:
#   crontab -e
#   0 2 * * * /home/user/tune_lease_55/scripts/daily_backup.sh >> /home/user/tune_lease_55/data/backup_cron.log 2>&1
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] バックアップ開始"
python3 -c "
from backup_manager import run_backup
import json
result = run_backup(force=False)
print('backed_up:', len(result['backed_up']), 'files')
print('skipped:  ', len(result['skipped']), 'files')
for f in result['backed_up']:
    print('  ->', f['dst'])
"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] バックアップ完了"
