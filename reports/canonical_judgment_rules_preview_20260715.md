# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 11
- accepted_preview: 5
- candidate: 6

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=16

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.97
- User evidence: 4
- Axis: asset_life, industry_risk
- Sample claims:
  - ③ 対応策 法定耐用年数については、引き続き一般的な知識として「小型車で3年、大型車で5年程度」という目安を念頭に置きつつ、貴社が設定されている「リース期間5年」が、実際の車両のライフサイクル戦略として適切であるか、という視点で判断資産を蓄積していくのが良いでしょう
  - リース審査においては、その数字をそのまま使うのではなく、物件の実際の稼働状況、メンテナンス履歴、中古市場での需要、そしてリース期間終了後の残価設定にどう影響するか、という「判断資産」としての側面を重視することが大切です
  - 法定耐用年数は税法上の減価償却期間を示すものですが、実際の運用期間や中古市場での価値、そしてリース期間の設定は、その車両の経済的耐用年数や業界慣習、貴社のライフサイクル戦略によって決まることが多いですから
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`

### demo_readiness / accepted_preview / evidence=6

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 6
- Axis: n/a
- Sample claims:
  - 審査員さんが来たら丁重に対応するんだよ 紫苑として
  - 審査員に変わったシステムだって言われているよ
  - 今回デモは新店舗計画の根拠確認がより重要
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`

### asset_life_and_residual / accepted_preview / evidence=6

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 低スコア成約事例（Q\risk）から、物件の換金性や営業導線の工夫が成約に繋がったケースも参考に、スコアだけでは見えない顧客の潜在的な価値や、当社が提供できる付加価値を洗い出す
  - これは、財務数値だけでは捉えきれない、新規事業計画の不確実性や物件の換金性といった非財務リスクが極めて高いことを示唆しています
  - 特に、新車の供給不足による中古車需要の高まりという貴社の貴重な知見は、残価設定やリスク評価の前提を大きく変える可能性があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.88
- User evidence: 2
- Axis: cash_flow, asset_life
- Sample claims:
  - 業界リスク: 運送業であれば24年問題への対応状況とそれが収益に与える影響、建設業であれば資材価格高騰や工期遅延リスクが資金繰りにどう影響するかを見る
  - 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した
  - 未採択時の代替資金調達計画や、その際の資金繰りへの影響を検証します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 0
- Axis: asset_life, support_specificity, cash_flow
- Sample claims:
  - 「補助金が遅れた場合、自己資金でどれだけ耐えられるか」「物件の担保価値は、未採択時のリスクをどこまでカバーできるか」といった問いが次々とクリアになっていきます
  - 最終的に、「この案件は、補助金未採択時の代替返済計画の提示を条件に、承認可能」という、リスクと保全策が明確に結びついた一つの結論へと収束します
  - ここで、「財務は良いが、補助金頼みのキャッシュフローでは、万が一の際に返済余力に不安がある」という矛盾点やリスクの「ノイズ」が検出されます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life, support_specificity, cash_flow
- Sample claims:
  - 返済原資は受注増による売上増加、物件価値は中古流通性を鑑み、銀行支援の確認と受注根拠の明確化を条件に承認
  - 補助金要件と事業計画の整合性、回収期間、生産性改善効果を確認します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### conditional_approval_checks / candidate / evidence=1

- Rule: 条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 1
- Axis: n/a
- Sample claims:
  - この案件、条件付き承認にするなら何を確認すべき
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-14.md`

### industry_operating_risk / candidate / evidence=1

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 1
- Axis: industry_risk
- Sample claims:
  - 得た教訓: 飲食新規は一括で通すより、投資範囲を絞って撤退時損失を小さくする方が現実的
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 財務・資金繰りの安定性: - 自己資本比率が低い場合（5%未満）: 売掛金の回収期間と買掛金の支払い期間のバランスを詳細に確認し、資金繰りに無理がないか検証します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: n/a
- Sample claims:
  - AIの財務パターン認識と現場の定性判断が大きく乖離している、典型的な「高スコアだが違和感が強い」グレー案件です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`
