# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 5
- accepted_preview: 1
- candidate: 4

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

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - ただ、検査機器のような分野は、技術の進歩が速い側面もありますから、リース期間中の陳腐化リスクや、それに伴う残価価値の変動については、少し注意深く見ておく必要があるかもしれませんね
  - ただし、個別の物件が持つ固有の価値や市場での換金性も、最終的な判断ではやはり重要になります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-30.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life, industry_risk, cash_flow
- Sample claims:
  - 事業が好調で生産量が増えている企業は、返済原資が安定していると見なされやすいですし、導入される検査機器も事業の中核を担う重要な物件として評価できます
  - リース審査の観点から見ると、そうした好調な業種の案件は、物件の用途や収益改善への寄与が明確であれば、より前向きに評価できる材料になります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-30.md`

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - ただ、その活況がどこまで続くのか、特定の業界や技術トレンドに過度に依存していないか、といった視点も、物件の残価価値や将来の回収可能性を見極める上では欠かせません
  - 特に検査機器のような分野は、技術革新のスピードや特定の産業への依存度によって、物件の残価価値や将来の換金性が変わることもあります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-30.md`

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
