# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 3
- accepted_preview: 2
- candidate: 1

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=6

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life, industry_risk
- Sample claims:
  - リース審査との関連: リース物件である車両にこの特約が付いている場合、万が一の事故で車両が大きな損害を受けても、リース会社（またはお客様）が新たな車両を調達しやすくなるため、リース契約の継続性や残価保全の観点から、リスクヘッジの一助となります
  - 現時点での仮判断としては、新車特約はリース車両の保全性を高める有効な手段であり、特にリース期間が新車特約の適用期間と重なる場合は、その付帯状況を確認することが望ましいです
  - 2000台もの車両が被災したとなると、リース審査の観点からは、特に車両リースの残価設定や、災害時の保険適用範囲について、改めて確認が必要になるかもしれません
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-14.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-15.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-15.md`

### business_plan_specificity / accepted_preview / evidence=5

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 1
- Axis: asset_life, cash_flow
- Sample claims:
  - 「実×計画」を利用されているお客様が、当初の事業計画にない設備投資を検討される際には、参加されている金融機関へ事前に確認を取り、場合によっては同意を得る必要がある、という点ですね
  - これは、金融機関が事業計画全体を評価して融資や保証を行っているため、計画外の投資が財務状況や返済能力に影響を与える可能性を考慮する必要がある、という理解で間違いありません
  - 判断資産として覚えといて実×計画を使用しているユーザは、事業計画以外の設備をする場合に参加している金融機関の大断る必要がある場合がある
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-14.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: industry_risk, cash_flow
- Sample claims:
  - 借手の事業計画: ラーメン屋さんの例で言えば、新規開業か既存店の更新か、事業の安定性や将来性、返済能力を総合的に評価します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-15.md`
