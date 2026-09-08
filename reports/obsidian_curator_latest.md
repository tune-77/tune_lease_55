# Obsidian Curator Report

## Summary
- generated_at: `2026-09-08T04:18:02`
- agent: `Obsidian Curator`
- mode: `read_only_report_only`
- guardrail: `no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun`
- materials: `0`
- mana_status: `watch`

## Material Counts
- なし

## Inbox Candidates
- なし

## Duplicate Clusters
- なし

## Search Terms To Add
- なし

## Related Gaps
- `Projects/tune_lease_55/検索語インデックス.md` related_section_missing missing=

## Mana Review Items
- {"action": "Inbox整理は可。GCS/RAG/active store接続は停止。", "reason": "mana_not_allow", "status": "watch"}

## After Hackathon Only
- Obsidianディレクトリ再編
- GCS Vault include/exclude変更
- accepted判断資産のactive store連携
- 判断資産レビューUI

## Next Safe Step
- Inbox候補を人間が採用・修正・却下・後回しに分類する。
- このレポート自体はObsidian本文、RAG、Cloud Run、active storeへ接続しない。
- 採用する場合は candidate_id を data/curator_inbox_decisions.jsonl に {"candidate_id": "...", "decision": "adopt"} として追記し、scripts/promote_curator_inbox_candidates.py を実行する（Mana許可時のみpreviewへ反映、activeへは既存のpromote_canonical_judgment_rules.pyを人間が実行）。
