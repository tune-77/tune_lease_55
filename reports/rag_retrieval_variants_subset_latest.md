# RAG Retrieval Variant Evaluation

- Generated: 2026-08-24T02:26:27
- Eval set: `/Users/kobayashiisaoryou/clawd/tune_lease_55/api/knowledge/rag_eval_set.json`
- Top K: 5
- Candidate K: 20
- Categories: ['lease_basics', 'asset_medical', 'credit']
- Guardrail: report_only_no_rag_config_no_prompt_no_scoring_no_obsidian_write

## Summary

| Variant | hit@1 | hit@k | MRR | Forbidden | Passed |
|---|---:|---:|---:|---:|---|
| baseline | 3/4 | 4/4 | 0.875 | 0/4 | True |
| path_dedupe | 3/4 | 4/4 | 0.875 | 0/4 | True |
| hard_exclude | 3/4 | 4/4 | 0.875 | 0/4 | True |
| category_boost | 4/4 | 4/4 | 1.000 | 0/4 | True |

## Finding

- Best variant: `category_boost`
- Exclude paths: `Projects/tune_lease_55/AI Chat/, Humor/, 05-クリップ_記事/業界リスクニュース/, 05-クリップ_記事/リースニュース/`

## Misses / Noise
