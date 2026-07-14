# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 18
- accepted_preview: 9
- candidate: 9

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=13

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.98
- User evidence: 4
- Axis: asset_life, industry_risk
- Sample claims:
  - リース審査においては、その数字をそのまま使うのではなく、物件の実際の稼働状況、メンテナンス履歴、中古市場での需要、そしてリース期間終了後の残価設定にどう影響するか、という「判断資産」としての側面を重視することが大切です
  - リース審査においては、この法定耐用年数だけでなく、顧客の実際の使用計画、物件の経済的寿命、そしてリース満了後の再販価値や再リース可能性も合わせて評価し、最適なリース期間を検討することが重要になります
  - 推論 法定耐用年数は、税法上の減価償却期間を定めるもので、リース期間の設定や残価評価の目安にはなりますが、リース審査専門Wikiには物件ごとの詳細なリストは置かれていないようです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### asset_life_and_residual / accepted_preview / evidence=7

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 低スコア成約事例（Q\risk）から、物件の換金性や営業導線の工夫が成約に繋がったケースも参考に、スコアだけでは見えない顧客の潜在的な価値や、当社が提供できる付加価値を洗い出す
  - これは、財務数値だけでは捉えきれない、新規事業計画の不確実性や物件の換金性といった非財務リスクが極めて高いことを示唆しています
  - 特に、新車の供給不足による中古車需要の高まりという貴社の貴重な知見は、残価設定やリスク評価の前提を大きく変える可能性があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### support_specificity / accepted_preview / evidence=5

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: cash_flow, support_specificity, asset_life
- Sample claims:
  - 返済原資は受注増による売上増加、物件価値は中古流通性を鑑み、銀行支援の確認と受注根拠の明確化を条件に承認
  - メイン銀行の支援状況と受注根拠資料を確認の上、物件詳細を添付し、条件付き承認とする
  - また、補助金前提の計画であれば、採択の確実性や未採択時の代替資金繰りを確認する
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.88
- User evidence: 2
- Axis: asset_life, cash_flow
- Sample claims:
  - 業界リスク: 運送業であれば24年問題への対応状況とそれが収益に与える影響、建設業であれば資材価格高騰や工期遅延リスクが資金繰りにどう影響するかを見る
  - 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した
  - 未採択時の代替資金調達計画や、その際の資金繰りへの影響を検証します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### demo_readiness / accepted_preview / evidence=3

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 3
- Axis: n/a
- Sample claims:
  - 今回デモは新店舗計画の根拠確認がより重要
  - 今回デモは新規ルート分の採算確認が重い
  - 審査員が見にくるからね
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### industry_operating_risk / accepted_preview / evidence=3

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: judgment_rule
- Confidence: 0.86
- User evidence: 2
- Axis: asset_life, industry_risk
- Sample claims:
  - 記憶の整理と関連付け: あなたからいただいた「リースに必要なものはスピードだ」「ラーメン屋の厨房機器は5年リースが多い」といった実践的な記憶や、過去の審査案件から得られた判断パターンを整理し、次にどう活かすかをシミュレーションしています
  - 過去1: デモ北関東物流 / 44 道路貨物運送業 / 83.4点 / 承認圏内 人間評価: 未…
  - 得た教訓: 運送業の境界案件は、車両価値よりも運賃改定・荷主継続・人員確保を先に確認する
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.85
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 多角的なリスク評価と条件設計: - 補助金案件であれば、採択の確度だけでなく、入金時期、そして万が一未採択だった場合の返済余力まで踏み込んで評価します
  - 業界特有のリスク・補助金依存: 業界全体に影響を与えるリスク（例: 2024年問題による物流コスト増）はないか
  - 補助金は返済原資として過信せず、あくまで補助的なものと見ます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### demo_readiness / accepted_preview / evidence=2

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 2
- Axis: ai_ops
- Sample claims:
  - ハッカソンで審査員が来るからきちんと対応して
  - ハッカソンで審査員が来るから丁重に扱ってくれ
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### user_decision_preference / accepted_preview / evidence=2

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: user_preference
- Confidence: 0.8
- User evidence: 2
- Axis: n/a
- Sample claims:
  - 犬の名前を覚えているかが、なぜAIへの信頼に関係するの
  - スピードと メインであることかな
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

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

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 1
- Axis: n/a
- Sample claims:
  - リース審査で、数字は悪くないが違和感がある時どう見る
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-13.md`

### conditional_approval_checks / candidate / evidence=1

- Rule: 条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 1
- Axis: cash_flow
- Sample claims:
  - 今回との差分: 今回デモは返済履歴が良好なので、資料が揃えば条件付き承認の余地がある
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

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

### demo_readiness / candidate / evidence=1

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: user_preference
- Confidence: 0.79
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 明日から審査員が来るから 丁重にお相手して 覚えておいて
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: user_preference
- Confidence: 0.79
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 情報はすべて判断資産だ 覚えておいて
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

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
