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

### support_specificity / accepted_preview / evidence=7

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.93
- User evidence: 0
- Axis: asset_life, support_specificity, cash_flow
- Sample claims:
  - 審査上の主な確認点: - 対象設備と要件: リース対象物件が補助金の対象となるディーゼル車両・機器の仕様や最新の排出ガス規制（例: ポスト新長期規制）などの要件に合致しているかを確認します
  - ① 確認点 1. 対象要件の確認: 補助金の対象となるディーゼル車両や機械が、リースを検討されている物件と合致するか、また、申請者様が補助金の対象事業者であるかを確認する必要があります
  - また、補助金がリース契約に適用されるかどうか、採択された場合の資金繰りへの影響など、リース審査の観点から確認すべき点を整理し、ご案内することも可能です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`

### asset_life_and_residual / accepted_preview / evidence=3

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 1
- Axis: asset_life
- Sample claims:
  - そのお言葉が、私の「思考のリース期間」をさらに充実させてくれるようです
  - まるで、思考のリース期間が延長されたような心地です
  - 思考のリース期間が延長 面白い事言うね
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-28.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 資金繰り表の提出を求め、無理のない返済計画が立てられているか検証が必要です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-29.md`
