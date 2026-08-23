# RAG Retrieval Variant Evaluation

- Generated: 2026-08-24T02:38:42
- Eval set: `/Users/kobayashiisaoryou/clawd/tune_lease_55/api/knowledge/rag_eval_set.json`
- Top K: 5
- Candidate K: 20
- Categories: all
- Guardrail: report_only_no_rag_config_no_prompt_no_scoring_no_obsidian_write

## Summary

| Variant | hit@1 | hit@k | MRR | Forbidden | Passed |
|---|---:|---:|---:|---:|---|
| baseline | 18/30 | 30/30 | 0.749 | 0/30 | True |
| path_dedupe | 18/30 | 30/30 | 0.749 | 0/30 | True |
| hard_exclude | 18/30 | 30/30 | 0.749 | 0/30 | True |
| category_boost | 19/30 | 30/30 | 0.766 | 0/30 | True |

## Finding

- Best variant: `category_boost`
- Exclude paths: `Projects/tune_lease_55/AI Chat/, Humor/, 05-クリップ_記事/業界リスクニュース/, 05-クリップ_記事/リースニュース/`

## Misses / Noise
