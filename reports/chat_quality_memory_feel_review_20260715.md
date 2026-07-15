# Local vs Cloud Run Chat Quality Review - 2026-07-15

## Targets

- Local: `http://127.0.0.1:8000`
- Cloud Run: `https://tune-lease-55-web-6mijhyebkq-an.a.run.app`
- Raw report: `reports/chat_quality_local_vs_cloudrun_latest.json`

## Quantitative Result

| Metric | Local | Cloud Run |
|---|---:|---:|
| Transport OK | 10/10 | 10/10 |
| Passed | 6/10 | 6/10 |
| Average score | 83.7 | 90.7 |
| Concept coverage | 76.7% | 86.7% |
| Average chars | 963.2 | 1211.3 |
| Average elapsed ms | 18380.1 | 17858.0 |

## Knowledge Surface

- Local Chroma chunks: 7045
- Cloud Run Chroma chunks: 1018
- Local synced GCS Vault: disabled locally; local Chroma uses the richer local index.
- Cloud Run GCS Vault: 213 Markdown files, Chroma 1018 chunks.

## Case Winners

- Local stronger: `asset_resale`
- Cloud Run stronger: `subsidy_lease`, `cashflow_weak`, `financial_statement_fraud`
- Tie: 6 cases

## Memory/RAG Evidence

Both environments returned `memory_debug` in all 10 cases.

- Average `knowledge_refs`: Local 4.0 / Cloud Run 4.0
- `identity_memory.used`: Local 10/10 / Cloud Run 10/10
- `rag_context_used`: Local 8/10 / Cloud Run 8/10
- `memory_recall.used`: Local 0/10 / Cloud Run 0/10

## Interpretation

Cloud Run scored higher on this fixed evaluation set, mostly because its answers were longer and hit more required terms in three cases. This does not mean Cloud Run has more knowledge. The local environment has far more indexed chunks and should be stronger for long-tail Obsidian-specific recall.

The practical difference is:

- Cloud Run is close enough for demo and common lease-knowledge questions.
- Local is the richer workbench for deep Obsidian context, judgment-asset curation, and unpublished/private knowledge.
- Both surfaces currently show the same Shion identity scaffolding and RAG usage signals.

## Demo-Safe Statement

Cloud Run has a narrower, curated knowledge copy. Local has the richer Obsidian-backed workspace. The system intentionally uses Cloud for execution and demo access, then routes judgment traces back through local human review before redistributing approved knowledge.

## Next Checks

- Re-run after deploying the latest local changes.
- Add cases that depend on newly promoted Judgment Assets.
- Add a separate metric for exact Obsidian-note recall, not only required-term scoring.
