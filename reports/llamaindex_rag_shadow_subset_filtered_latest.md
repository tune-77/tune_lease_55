# LlamaIndex RAG Comparison

- Generated: 2026-08-24T02:21:37
- Status: completed
- Method: llama_index.core.VectorStoreIndex + local char-ngram embedding
- Top K: 5
- Guardrail: sidecar_only_no_rag_rank_change_no_prompt_change_no_scoring_no_obsidian_write

## Summary

- hit@1: 3/4 (75.0%)
- hit@5: 4/4 (100.0%)
- mrr: 0.833
- forbidden_cases: 0/4
- passed: True

## Cases

### finance_lease_definition

- Status: PASS
- Rank: 1

1. 03-知識_業界/リース基礎知識/ファイナンスリース.md
2. 07-アーカイブ/tune_lease_55_archived_2026-06-12/取り込み完了_Phase1-3/2026-05-12_ファイナンスリース_autoresearch.md
3. 03-知識_業界/リース基礎知識/リース基本ルール.md
4. 03-知識_業界/リース基礎知識/メンテナンスリース.md
5. 07-アーカイブ/tune_lease_55_archived_2026-06-12/取り込み完了_Phase1-3/2026-05-12_リース基本ルール_autoresearch.md

### maintenance_lease_scope

- Status: PASS
- Rank: 1

1. 03-知識_業界/リース基礎知識/メンテナンスリース.md
2. 07-アーカイブ/tune_lease_55_archived_2026-06-12/取り込み完了_Phase1-3/2026-05-12_メンテナンスリース_autoresearch.md
3. 03-知識_業界/リース基礎知識/ファイナンスリース.md
4. Projects/tune_lease_55/Lease Intelligence/Knowledge/オイル価格高騰とメンテナンスリース_2026-06-29.md
5. 03-知識_業界/リース基礎知識/INDEX.md

### credit_risk_classification

- Status: PASS
- Rank: 1

1. 03-知識_業界/リース審査実務/信用リスク分類.md
2. 03-知識_業界/業種分析/業種別傾向.md
3. 03-知識_業界/リース審査実務/INDEX.md
4. 07-アーカイブ/tune_lease_55_archived_2026-06-12/取り込み完了_Phase1-3/2026-05-12_信用リスク群とスコアリング高速化.md
5. Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-06-17.md

### medical_equipment_resale

- Status: PASS
- Rank: 3

1. Projects/tune_lease_55/2026-05-13_リース審査AI_知識分解.md
2. 03-知識_業界/リース実務知識/再リース.md
3. Projects/tune_lease_55/Asset Knowledge/医療機器/医療機器 残価・保守期限・薬機法.md
4. Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-06-28.md
5. Projects/tune_lease_55/Asset Knowledge/車両/冷蔵冷凍車・商用車 メンテリース 残価・再販リスク.md
