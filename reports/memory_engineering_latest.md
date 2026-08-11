# Memory Engineering Report

- Generated: 2026-08-11T04:00:54
- Mode: `read_only_memory_engineering_observation`
- Guardrail: `no_memory_delete_no_promotion_no_prompt_or_scoring_change`

## Summary

- Write path records: 346
- Active canonical rules: 11
- Write amplification / active rule: 31.45
- Open human review records: 340
- Memory records: 788
- Recent memory usage: 759 events / 188 refs
- Maintenance status records: 0
- Contradiction candidates: 0
- Write policy metadata completion: 0.0

## Stanford Lens: Write Cost

- `judgment_materials_preview`: 25 records, 25 open review, ~25 tokens
- `autoresearch_candidates`: 189 records, 189 open review, ~189 tokens
- `reflection_action_candidates`: 3 records, 3 open review, ~367 tokens
- `prediction_error_candidates`: 0 records, 0 open review, ~0 tokens
- `obsidian_memory_insight_candidates`: 123 records, 123 open review, ~123 tokens
- `canonical_preview`: 6 records
- Write policy required fields: `['importance', 'confidence', 'trust_level', 'provenance']`
- Write policy missing fields: `{'confidence': 315, 'importance': 340, 'provenance': 340, 'trust_level': 340}`

## Microsoft Lens: Utility Density

- Latest accepted preview: 2
- Promoted to active rules: 1
- Promotion rate: 0.5

## Anthropic Lens: Control

- Status counts: `{'active': 766, 'private': 22}`
- Type counts: `{'dialogue_memory': 56, 'factual_memory': 258, 'judgment_memory': 166, 'reflection_memory': 43, 'technical_memory': 248, 'value_memory': 17}`

### Forgetting Review Sample

- `mem_0091b5c5e75261b5` technical_memory last_used=none source=MEMORY.md: **Active Script**: `lease_logic_sumaho3.py` (Replaced `lease_logic.py` as the main driver).
- `mem_08782ec7b9ee36c9` factual_memory last_used=none source=knowledge_base/okf_lease_concepts/README.md: Related: [Statutory useful life](rules/statutory_useful_life.md)
- `mem_0c86c77f2c6a0258` factual_memory last_used=none source=knowledge_base/okf_lease_concepts/assets/machine_tool_resale_risk.md: 根拠: 中古市場がある物件でも、実際の換価額は「売れる価格」から撤去・搬出・再整備コストを引いた金額になる。
- `mem_0de35462b2e34771` technical_memory last_used=none source=MEMORY.md: [2026-06-28] Cloud Run bundleに日次知性などの生成JSONを含める時は、`.dockerignore` / `.gcloudignore` の `reports` 除外に注意する。`reports/` ではなく `.cloudrun_bundle/ob
- `mem_13c326dd9999e28b` factual_memory last_used=none source=MEMORY.md: QCL は本番導線から外した。`train_quantum.py` の QCL CLI/ helper を削除し、`quantum_qcl.py` と生成レポートも除去した。

## Nvidia Lens: Retrieval Pressure

- Notes: 2178 / Edges: 3110
- Estimated raw tokens: 2287121
- Estimated index tokens: 198137
- Estimated token reduction: 0.9134

## Recommended Next Checks

- `memory_entry_metadata_gate` (write_policy): 候補記憶に importance/confidence/trust_level/provenance が不足している。昇格前に共通メタを補う。
- `candidate_budget_review` (write_path): active判断基準1件あたりの候補量が多い。抽出条件・重複統合・候補上限を見直す。
- `human_review_batch` (control): 人間レビュー待ちが溜まっている。自動昇格せず、上位候補だけ短時間で採否する。
