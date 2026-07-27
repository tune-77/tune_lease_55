# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 2
- accepted_preview: 0
- candidate: 2

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: n/a
- Sample claims:
  - これは、リース審査の判断資産を扱う上で、情報の鮮度や生成時刻を重視する私の設計によるものですが、Userさんの現在の時刻とズレが生じてしまい、申し訳ありません
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-26.md`

### support_specificity / candidate / evidence=1

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: support_specificity
- Sample claims:
  - 私は、皆さんのリース審査を支援するために存在していますが、システムそのものの設定変更や、内部の時刻同期メカニズムを直接操作する権限は持っていません
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-26.md`
