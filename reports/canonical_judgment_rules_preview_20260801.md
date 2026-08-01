# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 8
- accepted_preview: 4
- candidate: 4

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=4

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 0
- Axis: asset_life, industry_risk, support_specificity
- Sample claims:
  - Judgment Memory: 残価リスクや補助金案件、業種リスクの判断原則、承認/否決だけでなく条件付き承認へ落とし込むこと、過去事例を判断の型として再利用することなどが記されています
  - ただ、検査機器のような分野は、技術の進歩が速い側面もありますから、リース期間中の陳腐化リスクや、それに伴う残価価値の変動については、少し注意深く見ておく必要があるかもしれませんね
  - もしメインバンクからの支援が期待できない場合、連帯保証人の追加や頭金の増額、またはリース期間の短期化といった条件は検討可能でしょうか
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-30.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-31.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### asset_life_and_residual / accepted_preview / evidence=4

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - ただ、その活況がどこまで続くのか、特定の業界や技術トレンドに過度に依存していないか、といった視点も、物件の残価価値や将来の回収可能性を見極める上では欠かせません
  - 特に検査機器のような分野は、技術革新のスピードや特定の産業への依存度によって、物件の残価価値や将来の換金性が変わることもあります
  - 」 - 今後: 「この物件の残価リスクは、中古市場での流通性が低い点にあります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-30.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### business_plan_specificity / accepted_preview / evidence=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.86
- User evidence: 0
- Axis: asset_life, industry_risk, cash_flow
- Sample claims:
  - 特に、以前のグレー判断で見た「返済原資が弱く、物件保全も弱いなら条件付き承認に逃げない」という判断層の記憶を踏まえると、否決判断の根拠が明確であったか、そしてその根拠が実際にリスクとして現れたかを検証することが重要です
  - 事業が好調で生産量が増えている企業は、返済原資が安定していると見なされやすいですし、導入される検査機器も事業の中核を担う重要な物件として評価できます
  - リース審査の観点から見ると、そうした好調な業種の案件は、物件の用途や収益改善への寄与が明確であれば、より前向きに評価できる材料になります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-30.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: cash_flow, support_specificity, asset_life, industry_risk
- Sample claims:
  - 機械受注統計や業種別審査チェックリスト（小売業）、補助金関連のノートがヒットしましたが、セントラルキッチン自体のリース物件としての特性を深掘りする内容ではありませんでした
  - これらの情報から、メインバンクの支援に頼らない、自立した返済能力と保全策の有無を評価し、リース実行の可否や条件を検討したいと思います
  - メインバンクの支援が薄い場合、リース会社としては、借手の返済能力と物件の担保価値をより慎重に評価する必要があります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### industry_operating_risk / candidate / evidence=2

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life, industry_risk
- Sample claims:
  - 飲食店がセントラルキッチン設備をリースしたいとのこと、創業3年で黒字は評価できますが、自己資本が薄い点について、追加の調査結果を踏まえて判断します
  - 創業3年の飲食店でセントラルキッチン設備とのこと、黒字である点は評価できますが、自己資本の薄さと物件の特性から、慎重な検討が必要です
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - これは、当時の返済能力や事業計画の評価が正しかったことを強く裏付けます
  - 導入後の収益改善が、リース料の返済能力にどう繋がるのか
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-31.md`

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
