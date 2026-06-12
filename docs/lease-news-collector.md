# Lease News Collector

Daily news collector for lease-related information.

Files:
- `scripts/collect_lease_news_to_obsidian.py`
- `scripts/run_lease_news_collection.sh`
- `launchd/com.tunelease.lease-news-collector.plist`

What it does:
- searches Google News RSS with industry-specific query sets
- reads official RSS feeds from METI, FSA, and MLIT
- classifies articles with Gemini, with a deterministic rule fallback
- adds industry, lease asset, credit impact, screening checks, impact direction,
  source reliability, validity date, and canonical topic metadata
- deduplicates by normalized URL, title similarity, and canonical topic
- merges duplicate coverage into the existing note's `関連報道` section
- writes one searchable note per news topic to Obsidian
- writes a short daily focus note under `Projects/tune_lease_55/News/` so the latest news points are reused instead of only archived
- indexes saved or updated notes into the Obsidian RAG store
- appends a short summary to `Daily/YYYY-MM-DD.md`
- keeps the latest 30 days of `lease-grumble` illustrations under the frontend
  and archives older images to
  `Projects/tune_lease_55/Archive/Lease Grumble/Images/YYYY/MM/` in the normal Obsidian Vault

Default schedule:
- daily at 06:00 via `launchd`
- launchd default profile: `industry-watch`

Default Obsidian targets:
- `05-クリップ_記事/リースニュース/YYYY-MM-DD_リースニュース_*.md`
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
- `LEASE_NEWS_AI_CLASSIFY` (`0` disables Gemini classification)
- `GEMINI_API_KEY`
- `GEMINI_MODEL`

Optional CLI overrides:
- `--profile`
- `--news-dir`
- `--daily-dir`
- `--queries`
- `--limit`
- `--per-query`
- `--per-feed`
- `--no-ai-classify`
