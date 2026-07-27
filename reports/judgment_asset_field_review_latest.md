# Judgment Asset Field Review

- Date: 2026-07-28
- Mode: local_review_only
- Guardrail: review_only_no_promotion_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write
- Active rules: 9
- Grow: 0 / Review: 0 / Sleeping: 9 / Hold: 0
- Simulation feedback: 3 / included: False
- Unknown feedback rows: 0

## 伸ばす

- None

## 見直す

- None

## 眠ってる

### asset_life_and_residual
- Rule ID: `cf61a9701fc8cc42`
- Statement: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### business_plan_specificity
- Rule ID: `b259411afb954d6d`
- Statement: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### conditional_approval_checks
- Rule ID: `a61f3a316a651126`
- Statement: 条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### demo_renewal_asset
- Rule ID: `64e054542be673e4`
- Statement: 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### demo_renewal_asset
- Rule ID: `a34492fe19a18e3a`
- Statement: 更新設備の申込では、既存設備の稼働実績と受注増の根拠を並べ、増額後も返済原資が説明できるかを確認する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### demo_subsidy_machinery
- Rule ID: `0d0f11e77fba045d`
- Statement: 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### intuition_gap
- Rule ID: `ab0c516c139499b8`
- Statement: 数字が悪くない案件でも、違和感は追加確認事項・稟議補足・条件設定のいずれかに変換する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### purchase_option_guidance
- Rule ID: `7d5692520fad998c`
- Statement: 購入選択権は5〜30%程度が多い前提で、残価・再リース・満了後出口と整合するか確認する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

### support_specificity
- Rule ID: `766ad4e39e346f4f`
- Statement: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Feedback: used=0, helped=0, challenged=0, rejected=0, neutral=0
- Reason: active だが実利用フィードバックが未記録

## 保留

- None

## Notes

- この棚卸しは実利用フィードバックの見える化だけを行う。
- source=simulation または sim-* case は既定では試運転として除外する。
- grow は昇格ではなく、次回案件で優先して試す候補。
- review は自動修正せず、人間が文面・適用条件・使わない条件を見る。
- sleeping は削除候補ではなく、実案件でまだ試されていない active ルール。
