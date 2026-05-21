# Obsidian Backup Launchd

This repo ships a launchd job for daily Obsidian backups.

Files:
- `scripts/backup_obsidian_vault.py`
- `scripts/run_obsidian_backup.sh`
- `launchd/com.tunelease.obsidian-backup.plist`

Default behavior:
- source vault: `/Users/kobayashiisaoryou/Documents/Obsidian Vault`
- backup root: `/Users/kobayashiisaoryou/Library/Mobile Documents/com~apple~CloudDocs/tune_lease_55_backups/obsidian`
- retention: 14 snapshots per vault prefix
- schedule: once per day via `StartInterval`
- logs: `~/Library/Logs/tune_lease_55_obsidian_backup.{out,err}.log`

Install steps on macOS:

1. Copy the plist to `~/Library/LaunchAgents/`.
2. Load it with `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.tunelease.obsidian-backup.plist`.
3. Start it immediately with `launchctl kickstart -k gui/$UID/com.tunelease.obsidian-backup`.
4. Verify logs in `~/Library/Logs/`.

Uninstall:

1. `launchctl bootout gui/$UID/com.tunelease.obsidian-backup`
2. Remove the plist from `~/Library/LaunchAgents/`

Notes:
- The job uses the repo-local backup script and never deletes the source vault.
- The script supports `--dry-run` and `--keep`.
