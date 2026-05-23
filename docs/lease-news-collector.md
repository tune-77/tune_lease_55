# Lease News Collector

Daily news collector for lease-related information.

Files:
- `scripts/collect_lease_news_to_obsidian.py`
- `scripts/run_lease_news_collection.sh`
- `launchd/com.tunelease.lease-news-collector.plist`

What it does:
- searches Google News RSS with industry-specific query sets
- reads official RSS feeds from METI, FSA, and MLIT
- deduplicates articles
- writes a daily digest note to Obsidian
- appends a short summary to `Daily/YYYY-MM-DD.md`

Default schedule:
- daily at 07:20 via `launchd`
- launchd default profile: `industry-watch`

Default Obsidian targets:
- `Projects/tune_lease_55/News/YYYY-MM-DD_lease-news.md`
- `Daily/YYYY-MM-DD.md`

Built-in profiles:
- `lease-core`
- `policy-watch`
- `industry-watch`
- `all`

Install steps on macOS:

1. Copy the plist to `~/Library/LaunchAgents/`.
2. Load it with `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.tunelease.lease-news-collector.plist`.
3. Start it immediately with `launchctl kickstart -k gui/$UID/com.tunelease.lease-news-collector`.
4. Check `~/Library/Logs/tune_lease_55_lease_news.{out,err}.log`.

Uninstall:

1. `launchctl bootout gui/$UID/com.tunelease.lease-news-collector`
2. Remove the plist from `~/Library/LaunchAgents/`

Optional env overrides:
- `OBSIDIAN_VAULT`
- `PYTHON_BIN`
- `LEASE_NEWS_PROFILE`
- `LEASE_NEWS_QUERIES`

Optional CLI overrides:
- `--profile`
- `--news-dir`
- `--daily-dir`
- `--queries`
- `--limit`
- `--per-query`
- `--per-feed`
