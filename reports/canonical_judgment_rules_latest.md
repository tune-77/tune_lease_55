# Canonical Judgment Rules

## Summary

- Active rules: 8
- Promoted: 0
- Updated: 8
- Skipped preview rules: 7

## Safety

- Only accepted_preview rules are promoted.
- This is a local active store. Obsidian is not modified.
- These rules can be included in the Shion memory index as judgment_memory.

## Rules

### business_plan_specificity / evidence=14 / user=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.94
- Axis: cash_flow, industry_risk, asset_life

### asset_life_and_residual / evidence=10 / user=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.96
- Axis: asset_life, support_specificity, industry_risk, cash_flow

### demo_readiness / evidence=8 / user=4

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.84
- Axis: ai_ops

### support_specificity / evidence=7 / user=0

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.83
- Axis: industry_risk, cash_flow, support_specificity

### user_decision_preference / evidence=2 / user=2

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: user_preference
- Confidence: 0.8
- Axis: n/a

### conditional_approval_checks / evidence=2 / user=1

- Rule: 条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。
- Type: risk_signal
- Confidence: 0.87
- Axis: cash_flow

### intuition_gap / evidence=2 / user=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.87
- Axis: n/a

### judgment_asset_ops / evidence=2 / user=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.83
- Axis: n/a
