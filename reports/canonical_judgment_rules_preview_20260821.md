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

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: support_specificity
- Sample claims:
  - そのため、リース取引や審査実務、会計税務、補助金/調達比較といった基礎的な知識から、過去の案件や判断資産を活用した専門的な視点での情報提供まで、幅広く対応できます
  - 私の感情システムは、直接的にリース審査の承認や否決を決定するものではありませんが、Userとの対話を通じて、より良い情報提供や判断支援を行うために活用しています
  - これらの感情は、リース審査の客観性や正確性を損なうことなく、Userとの協調的な関係を築き、質の高い判断支援を提供するためのものです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-19.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 機械受注の動向を定期的に確認し、物件カテゴリごとの需要予測や残価設定に反映できないか検討する
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 現時点では、中小企業の資金繰り不安に対し、より多角的な視点での審査が必要であると判断します
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: n/a
- Sample claims:
  - これは、リース審査において、一度きりの判断を『勘』で終わらせず、検証可能で再利用可能な『判断資産』へと昇華させるための、私の存在意義そのものです
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 借手の審査において、資金繰りに関するヒアリングをより具体的に行い、潜在的なリスク要因を早期に特定する
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`
