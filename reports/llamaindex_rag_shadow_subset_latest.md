# LlamaIndex RAG Comparison

- Generated: 2026-08-24T02:18:50
- Status: completed
- Method: llama_index.core.VectorStoreIndex + local char-ngram embedding
- Top K: 5
- Guardrail: sidecar_only_no_rag_rank_change_no_prompt_change_no_scoring_no_obsidian_write

## Summary

- hit@1: 3/4 (75.0%)
- hit@5: 3/4 (75.0%)
- mrr: 0.750
- forbidden_cases: 1/4
- passed: False

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
2. 05-クリップ_記事/業界リスクニュース/2026-08-13_業界リスクニュース_カインホア省では建設資材が不足している。 - vietnam.md
3. 05-クリップ_記事/業界リスクニュース/2026-07-24_業界リスクニュース_NECソリューションイノベータ_物流に関するリサーチ結果公開.md
4. 05-クリップ_記事/業界リスクニュース/2026-08-19_業界リスクニュース_7月の倒産件数 月別で今年最多に【岩手】 - IAT岩手朝日.md
5. 05-クリップ_記事/業界リスクニュース/2026-08-13_業界リスクニュース_日本工作機械工業会、7月の工作機械受注額(速報)を発表 - .md

### medical_equipment_resale

- Status: MISS
- Rank: -
- Forbidden paths: ['Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md']

1. Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-15.md
2. 05-クリップ_記事/業界リスクニュース/2026-07-24_業界リスクニュース_AI半導体の設備投資急増で装置業界が「スーパー乙」に君臨、「.md
3. 05-クリップ_記事/業界リスクニュース/2026-07-16_業界リスクニュース_「ソニー×台湾TSMC」提携による半導体イメージセンサー共同.md
4. 05-クリップ_記事/業界リスクニュース/2026-07-18_業界リスクニュース_最大2億円!東京都「躍進的な事業推進のための設備投資支援事業.md
5. 05-クリップ_記事/業界リスクニュース/2026-08-22_業界リスクニュース_「ソニー×台湾TSMC」提携による半導体イメージセンサー共同.md
