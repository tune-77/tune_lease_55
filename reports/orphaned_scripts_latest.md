# Orphaned Scripts Report

- Generated at: `2026-08-20T19:13:31+00:00`
- Mode: `read_only_orphaned_script_audit`
- Caveat: スクリプトがリポジトリ外（cron/launchd等）から呼ばれている場合は検知できない。誤検知の可能性があるため、必ず人間が内容を確認してから対応すること。
- Entry-point scripts scanned: 199
- Wired (referenced elsewhere): 181
- Orphaned (no reference found): 18

## Orphaned Scripts
- `scripts/backfill_private_reflection.py` (no test)
- `scripts/build_judgment_asset_rules_v2.py` (no test)
- `scripts/compare_embedding_models.py` (test exists)
- `scripts/convert_edinet_to_training.py` (no test)
- `scripts/convert_leasedb_to_markdown.py` (no test)
- `scripts/convert_leasedb_to_markdown_full.py` (no test)
- `scripts/create_hackathon_devops_video.py` (no test)
- `scripts/demo_predict_yield.py` (no test)
- `scripts/experiment_contradiction.py` (no test)
- `scripts/export_to_obsidian.py` (no test)
- `scripts/fetch_edinet_data.py` (no test)
- `scripts/gcs_to_local_sync.py` (test exists)
- `scripts/migrate_add_deal_timestamps.py` (no test)
- `scripts/migrate_add_funding_rates.py` (no test)
- `scripts/obsidian_archive.py` (no test)
- `scripts/review_recipes.py` (no test)
- `scripts/run_experience_replay_answer_quality.py` (no test)
- `scripts/run_monthly_prompt_feedback_report.py` (no test)
