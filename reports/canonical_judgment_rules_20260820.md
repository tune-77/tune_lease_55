# Canonical Judgment Rules

## Summary

- Active rules: 9
- Promoted: 0
- Updated: 1
- Skipped preview rules: 0

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
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### asset_life_and_residual / evidence=10 / user=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.96
- Axis: asset_life, support_specificity, industry_risk, cash_flow
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### demo_readiness / evidence=8 / user=4

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.84
- Axis: ai_ops
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### support_specificity / evidence=7 / user=0

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.86
- Axis: industry_risk, cash_flow, support_specificity
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### demo_renewal_asset / evidence=3 / user=3

- Rule: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- Type: risk_signal
- Confidence: 0.8500000000000001
- Axis: n/a
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### demo_subsidy_machinery / evidence=3 / user=2

- Rule: 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。
- Type: judgment_rule
- Confidence: 0.8200000000000001
- Axis: n/a
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### conditional_approval_checks / evidence=2 / user=1

- Rule: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- Type: risk_signal
- Confidence: 0.87
- Axis: cash_flow
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### intuition_gap / evidence=2 / user=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項・稟議補足・条件設定のいずれかに変換する。
- Type: judgment_rule
- Confidence: 0.87
- Axis: n/a
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### judgment_asset_ops / evidence=2 / user=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.83
- Axis: n/a
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### demo_renewal_asset / evidence=1 / user=1

- Rule: 更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。
- Type: risk_signal
- Confidence: 0.81
- Axis: n/a
- 親判断: 根の判断資産
- 派生理由: 根の判断資産

### purchase_option_guidance / evidence=1 / user=1

- Rule: 購入選択権は5〜30%程度が多い前提で、残価・再リース・満了後出口と整合するか確認する。
- Type: user_preference
- Confidence: 0.78
- Axis: n/a
- 親判断: 根の判断資産
- 派生理由: 根の判断資産
