---
type: index
title: OKF-style lease knowledge pack
domain: rag
tags: [OKF, RAG, lease-knowledge]
source: internal_review
confidence: medium
status: active
updated: 2026-06-20
related:
  - docs/okf_subset.md
---

# OKF-style lease knowledge pack

This folder is the first small knowledge pack for Markdown + YAML frontmatter lease concepts.

## Purpose

- Keep reusable lease judgment notes portable.
- Give RAG stable metadata fields.
- Let agents consume the same concepts without custom parsing per note family.

## OKF風のリース知識パックの目的とルール

この知識パックの目的は、リース審査の重要概念を Markdown + YAML frontmatter で揃え、RAGとAIエージェントが同じ知識を安定して参照できるようにすること。

ルールは、`type` を必須にし、`title`、`domain`、`tags`、`source`、`confidence`、`status`、`updated`、`related` を推奨項目として使うこと。本文は通常のMarkdownとして、人間がObsidianやGitで読める形を保つ。

## Rules

- `type` is required.
- Other fields are recommended and should stay compact.
- Notes should remain useful in plain Obsidian.

## Related

- [OKF-style subset spec](../../docs/okf_subset.md)
- [Hydraulic excavator lease period](assets/hydraulic_excavator_lease_period.md)
- [Expected usage period and lease term](rules/expected_usage_period_and_lease_term.md)
- [Statutory useful life](rules/statutory_useful_life.md)
- [Conditional approval playbook](rules/conditional_approval_playbook.md)
- [Q risk interpretation](rules/q_risk_interpretation.md)
- [Current datetime prompt context](rules/current_datetime_prompt_context.md)
