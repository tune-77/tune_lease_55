# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 9
- accepted_preview: 5
- candidate: 4

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=5

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.93
- User evidence: 1
- Axis: asset_life, industry_risk, cash_flow
- Sample claims:
  - 紫苑の審査では、残価リスクや業種リスクだけでなく、借手の返済能力を多角的に評価することが重要ですから、このような情報が加わることで、より実践的な判断が可能になるかもしれません
  - リース審査においては、物件が最終的に借手の資産となるため、借手の信用力だけでなく、物件の残存価値や、リース期間中の適切な管理・使用がより重要になります
  - リース料がどのくらいになるかは、導入される機器の総額、リース期間（通常は5年〜7年が多いです）、そしてお客様の信用状況によって変動します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 0
- Axis: support_specificity, cash_flow
- Sample claims:
  - 過去の判断資産やObsidian知識を踏まえますと、補助金は採算を良く見せる一方で、入金時期のズレが資金繰りリスクを作ることがあるため、採択前提で返済余力を組み立てないことが重要です
  - 未採択時の返済計画: もし補助金が採択されなかった場合でも、リース料を安定して支払えるだけの返済余力があるか、または代替となる資金計画があるかを確認させてください
  - しかし、補助金頼みで資金繰りが脆弱であったり、未採択時のリスクが高い場合は、慎重な検討が必要となり、保留または否決寄りの判断となります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.84
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 一方で、計画が曖昧であったり、返済原資に懸念が残る場合は、保留または否決を検討することになります
  - 特に、資金繰りの安定性や事業継続への信頼性を示すため、信用リスクの評価において加点されます
  - 借手の返済計画へのリスク: 建設コスト増加が、借手の資金繰りや返済能力に与える影響
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-02.md`

### industry_operating_risk / accepted_preview / evidence=3

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life, industry_risk
- Sample claims:
  - 昨日は、Userから飲食店がセントラルキッチン設備をリースしたいというご相談を受け、その判断について調査を進めていました
  - 昨日（7月31日）は、飲食店からのセントラルキッチン設備リースのご相談について、その判断に必要な調査を進めておりました
  - 昨日（7月31日）は、飲食店からのセントラルキッチン設備リースのご相談について、判断に必要な調査を進めておりました
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`

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

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これらは汎用性が限定されるため、万が一事業が計画通りに進まなかった場合の残価価値や換金性を慎重に見極める必要があります
  - これは、リース終了時の残価価値の維持や、万が一の際の換金性に直接影響するためです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`

### intuition_gap / candidate / evidence=2

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これは、私の回答が時に抽象的になったり、紋切り型に聞こえたりする点を改善し、より具体的で深い審査コメントや、Userが見落としがちな「違和感」を言語化できるようになることを目指しています
  - 過去の製造業における専門機械のリース案件を踏まえますと、射出成形機のような特殊な設備は、その技術陳腐化リスクや中古市場での流動性が限定的である点が、特に注意すべき違和感となります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-02.md`

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

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 過去の判断資産を踏まえますと、屋外物件のリース審査では、単に物件の種類だけでなく、設置場所の気候条件、想定される稼働環境、そして適切な保守・メンテナンス計画が立てられているかを詳細に確認します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-02.md`
