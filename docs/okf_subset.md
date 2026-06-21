# OKF-Style Knowledge Subset

This project uses a small Markdown + YAML frontmatter subset for lease knowledge that needs to be shared across RAG, agents, and human review.

It is intentionally conservative. The goal is not to rename the whole Vault or depend on an external platform. The goal is to make important concepts machine-readable while keeping notes easy to edit in Obsidian and Git.

## Scope

Use this subset for curated, reusable knowledge:

- Lease concepts and rules
- Asset-life and useful-life notes
- Scoring interpretation notes
- Q_risk / Mahalanobis / anomaly explanation notes
- Agent prompt policy notes
- Cross-note indexes that should be stable RAG sources

Do not use it for raw chat logs, temporary daily notes, private reflection, generated reports, or one-off work logs unless they are promoted into reusable knowledge.

## Required Field

Only one field is required:

```yaml
type: lease_concept
```

Allowed initial types:

- `lease_concept`
- `lease_rule`
- `asset_profile`
- `risk_signal`
- `agent_policy`
- `index`

## Recommended Fields

```yaml
title: 油圧ショベルのリース期間
domain: asset_life
tags: [建機, 油圧ショベル, リース期間]
source: internal_review
confidence: medium
status: active
updated: 2026-06-20
related:
  - 期待使用期間
  - 建機
```

Recommended values:

- `domain`: `asset_life`, `credit`, `pricing`, `q_risk`, `rag`, `agent`, `operations`, `industry`, `contract`
- `confidence`: `low`, `medium`, `high`
- `status`: `active`, `draft`, `deprecated`
- `source`: short provenance such as `internal_review`, `obsidian`, `official`, `analysis`, `user_feedback`

## Body Shape

Keep the body regular Markdown. These sections are preferred, not mandatory:

```md
## 要点

## 判断ルール

## 根拠

## 関連
```

RAG already indexes frontmatter through `api/knowledge/obsidian_loader.py`, so these fields become searchable metadata without changing the vector store schema.

## Adoption Rule

Start with 10 to 30 curated notes. Do not bulk-convert the Vault.

1. Create or migrate one concept note.
2. Run `python scripts/validate_okf_subset.py <path-or-dir>`.
3. Rebuild the RAG index only after validation passes.
4. Promote more notes only when they improve retrieval or agent behavior.

