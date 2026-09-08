# Orphaned Scripts Report

- Generated at: `2026-09-07T19:17:40+00:00`
- Mode: `read_only_orphaned_script_audit`
- Caveat: スクリプトがリポジトリ外（cron/launchd等）から呼ばれている場合は検知できない。誤検知の可能性があるため、必ず人間が内容を確認してから対応すること。
- Entry-point scripts scanned: 206
- Wired (referenced elsewhere): 201
- Orphaned (no reference found): 5

## Orphaned Scripts
- `scripts/analyze_debate_quality.py` (test exists)
- `scripts/build_judgment_asset_promotion_readiness_report.py` (test exists)
- `scripts/build_judgment_asset_response_manifest.py` (test exists)
- `scripts/build_shion_auto_queue.py` (no test)
- `scripts/reconcile_needs_review_from_prs.py` (test exists)
