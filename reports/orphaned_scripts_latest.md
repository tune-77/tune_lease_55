# Orphaned Scripts Report

- Generated at: `2026-08-23T19:16:19+00:00`
- Mode: `read_only_orphaned_script_audit`
- Caveat: スクリプトがリポジトリ外（cron/launchd等）から呼ばれている場合は検知できない。誤検知の可能性があるため、必ず人間が内容を確認してから対応すること。
- Entry-point scripts scanned: 200
- Wired (referenced elsewhere): 193
- Orphaned (no reference found): 7

## Orphaned Scripts
- `scripts/build_judgment_asset_promotion_readiness_report.py` (test exists)
- `scripts/build_judgment_asset_response_manifest.py` (test exists)
- `scripts/build_shion_auto_queue.py` (no test)
- `scripts/compare_llamaindex_rag.py` (test exists)
- `scripts/decide_shion_candidates.py` (test exists)
- `scripts/evaluate_rag_retrieval_variants.py` (test exists)
- `scripts/experiment_identity_comparison.py` (no test)
