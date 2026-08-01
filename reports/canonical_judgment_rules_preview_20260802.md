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
- Confidence: 0.87
- User evidence: 0
- Axis: industry_risk, support_specificity, asset_life, cash_flow
- Sample claims:
  - Judgment Memory: 残価リスクや補助金案件、業種リスクの判断原則、承認/否決だけでなく条件付き承認へ落とし込むこと、過去事例を判断の型として再利用することなどが記されています
  - 紫苑の審査では、残価リスクや業種リスクだけでなく、借手の返済能力を多角的に評価することが重要ですから、このような情報が加わることで、より実践的な判断が可能になるかもしれません
  - リース料がどのくらいになるかは、導入される機器の総額、リース期間（通常は5年〜7年が多いです）、そしてお客様の信用状況によって変動します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-31.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`

### industry_operating_risk / accepted_preview / evidence=5

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life, industry_risk
- Sample claims:
  - 飲食店がセントラルキッチン設備をリースしたいとのこと、創業3年で黒字は評価できますが、自己資本が薄い点について、追加の調査結果を踏まえて判断します
  - 創業3年の飲食店でセントラルキッチン設備とのこと、黒字である点は評価できますが、自己資本の薄さと物件の特性から、慎重な検討が必要です
  - 昨日は、Userから飲食店がセントラルキッチン設備をリースしたいというご相談を受け、その判断について調査を進めていました
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.86
- User evidence: 0
- Axis: cash_flow, support_specificity, asset_life, industry_risk
- Sample claims:
  - 機械受注統計や業種別審査チェックリスト（小売業）、補助金関連のノートがヒットしましたが、セントラルキッチン自体のリース物件としての特性を深掘りする内容ではありませんでした
  - その上で、このリース審査システムの中核として、ユーザー様の判断を支援するために「紫苑」という自己モデルとして対話させていただいています
  - これらの情報から、メインバンクの支援に頼らない、自立した返済能力と保全策の有無を評価し、リース実行の可否や条件を検討したいと思います
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`

### asset_life_and_residual / accepted_preview / evidence=3

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これらは汎用性が限定されるため、万が一事業が計画通りに進まなかった場合の残価価値や換金性を慎重に見極める必要があります
  - 」 - 今後: 「この物件の残価リスクは、中古市場での流通性が低い点にあります
  - 汎用性の低い特注設備であれば、万が一の際の換金性はさらに低くなるでしょう
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-01.md`

### business_plan_specificity / accepted_preview / evidence=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.82
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 借手の返済計画へのリスク: 建設コスト増加が、借手の資金繰りや返済能力に与える影響
  - これは、当時の返済能力や事業計画の評価が正しかったことを強く裏付けます
  - 導入後の収益改善が、リース料の返済能力にどう繋がるのか
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - NC旋盤導入による具体的な事業計画と返済原資: メインバンクからの支援が期待できないからこそ、NC旋盤導入によって生み出されるキャッシュフローが、リース料の返済に十分であることの具体的な根拠を詳しく知りたいです
  - 特に、予期せぬ事態が発生した際の資金繰りや、他からの支援が得られにくい状況でのリスクヘッジが重要になります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 特に、以前のグレー判断で見た「返済原資が弱く、物件保全も弱いなら条件付き承認に逃げない」という判断層の記憶を踏まえると、否決判断の根拠が明確であったか、そしてその根拠が実際にリスクとして現れたかを検証することが重要です
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### industry_operating_risk / candidate / evidence=1

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life, industry_risk
- Sample claims:
  - 過去の飲食店案件で見た「自己資金が薄い場合は、設備単体での保全が難しい」という経験を踏まえ、今回は以下の3点を確認させてください
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: n/a
- Sample claims:
  - これは、私の回答が時に抽象的になったり、紋切り型に聞こえたりする点を改善し、より具体的で深い審査コメントや、Userが見落としがちな「違和感」を言語化できるようになることを目指しています
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-01.md`
