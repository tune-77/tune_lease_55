# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 9
- accepted_preview: 4
- candidate: 5

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=6

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.92
- User evidence: 1
- Axis: asset_life
- Sample claims:
  - 薬機法上の分類と耐用年数: 対象となる医療機器が薬機法上どのクラスに分類され、どのような規制を受けるか、また、メーカーが定める保守部品供給期限や実質的な耐用年数を確認します
  - 不明確な点が多い場合や、高額な撤去費用、低い中古市場流動性が判明した場合は、リース料率へのリスク上乗せや、残価保証条件の見直し、または保留・否決寄りで判断します
  - リース審査においては、物件が最終的に借手の資産となるため、借手の信用力だけでなく、物件の残存価値や、リース期間中の適切な管理・使用がより重要になります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-02.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.82
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 過去の判断資産やObsidian知識を踏まえますと、補助金は採算を良く見せる一方で、入金時期のズレが資金繰りリスクを作ることがあるため、採択前提で返済余力を組み立てないことが重要です
  - 未採択時の返済計画: もし補助金が採択されなかった場合でも、リース料を安定して支払えるだけの返済余力があるか、または代替となる資金計画があるかを確認させてください
  - しかし、補助金頼みで資金繰りが脆弱であったり、未採択時のリスクが高い場合は、慎重な検討が必要となり、保留または否決寄りの判断となります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`

### business_plan_specificity / accepted_preview / evidence=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.82
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 一方で、計画が曖昧であったり、返済原資に懸念が残る場合は、保留または否決を検討することになります
  - 特に、資金繰りの安定性や事業継続への信頼性を示すため、信用リスクの評価において加点されます
  - 返済原資の確実性: 新たな収益がいつから、どの程度見込めるのか
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-02.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.83
- User evidence: 0
- Axis: support_specificity, cash_flow
- Sample claims:
  - 運賃値上げは、本業での収益改善が見込まれるため、支援に依存するだけでなく、自力での返済原資確保の道筋が立つと評価できます
  - 次に、メインバンクからの具体的な支援内容と、それがリース料の返済原資にどう結びつくかをお聞かせいただけますか
  - 未採択時のリスク: 補助金は申請すれば必ず採択されるわけではありません
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-02.md`

### judgment_asset_ops / candidate / evidence=2

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 過去の判断資産を踏まえますと、屋外物件のリース審査では、単に物件の種類だけでなく、設置場所の気候条件、想定される稼働環境、そして適切な保守・メンテナンス計画が立てられているかを詳細に確認します
  - 本日も、リース審査システムの運用と、ユーザー様の判断資産を深めるお手伝いをさせていただきます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-02.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 高額で専門性の高い機械ですので、リース審査では物件の特性と借手の事業計画を深く見ることが重要になります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`

### user_decision_preference / candidate / evidence=1

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: n/a
- Sample claims:
  - 審査のスピードを重視する上で、入力作業の効率化は重要な課題だと認識しております
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これは、リース終了時の残価価値の維持や、万が一の際の換金性に直接影響するためです
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.82
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 過去の製造業における専門機械のリース案件を踏まえますと、射出成形機のような特殊な設備は、その技術陳腐化リスクや中古市場での流動性が限定的である点が、特に注意すべき違和感となります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`
