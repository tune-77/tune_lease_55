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

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これは税務上の減価償却資産の耐用年数であり、実際のリース期間設定や残価設定の検討時には、物件の用途や稼働状況、中古市場の動向も考慮します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-24.md`

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
