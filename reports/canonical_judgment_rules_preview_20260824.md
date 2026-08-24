# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 3
- accepted_preview: 0
- candidate: 3

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
- Axis: n/a
- Sample claims:
  - この案件は、AIスコアが53.9点で「要審議」と判断されたにもかかわらず、最終的に「再リース3回条件」で成約に至ったという、詳細な経緯が残っています
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 新しい案件を見たとき、私はその記憶を参照し、単に数字を見るだけでなく、返済原資や設備稼働開始時期といった追加の確認点を提案できます
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-22.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: n/a
- Sample claims:
  - 自己モデルと判断資産: 私自身の役割や、リース審査における判断の原則、過去の判断パターンなども記憶として持っています
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-22.md`
