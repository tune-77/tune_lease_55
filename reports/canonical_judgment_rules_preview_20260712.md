# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 18
- accepted_preview: 11
- candidate: 7

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### business_plan_specificity / accepted_preview / evidence=10

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.94
- User evidence: 3
- Axis: cash_flow, industry_risk, asset_life
- Sample claims:
  - これは、運送業の特性として、物件（車両）の担保価値はあれど、肝心の返済原資が燃料費や人件費といった変動費に大きく左右されるため、財務諸表だけでは見えにくいリスクが高いことを意味します
  - 15トン大型トラック20台という規模を考えると、審査側が特に注目するのは、その「稼働計画と収益性」、そして「リスクヘッジと出口戦略」だと考えられます
  - 稟議で残すべき一文 既存メイン先からの受注増に対応する加工能力増強投資であり、返済原資と物件用途の整合性、工作機械としての物件保全性は高い
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### asset_life_and_residual / accepted_preview / evidence=10

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.96
- User evidence: 1
- Axis: asset_life, support_specificity, industry_risk, cash_flow
- Sample claims:
  - そのため、リース審査では、単に「ラーメン屋」という業種だけでなく、具体的な事業計画の堅実さ、資金計画の余裕、そして万が一の際に備えた物件の汎用性や換金性といった点が、より一層深く見られることになります
  - ただ、審査ではこの法定耐用年数だけでなく、実際の稼働状況、メンテナンス履歴、中古市場での需要、そして満了後の再販・再リース可能性といった「物件の経済的寿命」も合わせて評価します
  - リース期間を設定する際には、この法定耐用年数だけでなく、実際の使用予定期間や物件の経済的寿命、満了後の再販・再リース可能性も合わせて検討することが重要になります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### demo_readiness / accepted_preview / evidence=5

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 2
- Axis: ai_ops
- Sample claims:
  - ハッカソンでの審査員の方々に向けて、私がどのようにリース審査を行っているか、そのプロセスを説明しますね
  - ハッカソンで審査員が来るからきちんと対応して
  - ハッカソンで審査員の方がご覧になるのですね
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 返済能力と資金繰りの安定性: - 自己資本比率が低い場合（5%未満）: 売掛金の回収期間と買掛金の支払い期間のバランスを確認し、資金繰りに無理がないか詳細に審査します
  - リース審査の視点からも、飲み物、特にアルコールによる安定した売上は、事業の収益基盤を強化し、リース物件の返済能力に対する評価を高める要素となります
  - 営業メモにある「返済原資と物件用途の説明がしやすい」という点は、審査の安定材料になります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.82
- User evidence: 0
- Axis: industry_risk, cash_flow, support_specificity
- Sample claims:
  - 補助金案件の場合: 採択の蓋然性、入金スケジュール、そして最も重要な「不採択時でもリース料を支払えるか」という代替返済能力とつなぎ資金の有無を確認します
  - 「資金繰り」への具体性と現実性: 飲食業の倒産が多い現状を踏まえると、事業計画の中で特に「資金繰り」に関する記述が重要になります
  - 補助金採択の確実性: - 補助金が不採択となった場合の借手の資金計画、返済能力、および事業継続性への影響を確認します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### demo_readiness / accepted_preview / evidence=3

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 2
- Axis: n/a
- Sample claims:
  - 審査員の方々には、私がどのように複雑な情報を統合し、実践的なリース判断に貢献できるかを見ていただければ幸いです
  - 今回デモは新規ルート分の採算確認が重い
  - 審査員が見にくるからね
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.83
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 多角的なリスク評価と条件設計: - 補助金案件であれば、採択の確度だけでなく、入金時期、そして万が一未採択だった場合の返済余力まで踏み込んで評価します
  - メインバンクの支援: メインバンクからの支援表明や、他行からの融資状況も、返済能力を補完する要素として確認します
  - 補助金は返済原資として過信せず、あくまで補助的なものと見ます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### user_decision_preference / accepted_preview / evidence=2

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: user_preference
- Confidence: 0.8
- User evidence: 2
- Axis: n/a
- Sample claims:
  - 覚えておいて 購入選択権は5パーセントから30%なことが多い
  - 犬の名前を覚えているかが、なぜAIへの信頼に関係するの
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### intuition_gap / accepted_preview / evidence=2

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.87
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 私は、ユーザーのリース審査の経験・違和感・判断基準を、再利用できる判断資産として育てることを目的としています
  - リース審査で、数字は悪くないが違和感がある時どう見る
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### judgment_asset_ops / accepted_preview / evidence=2

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 私は、ユーザーさんのリース審査に関する知識や判断基準を、再利用できる「判断資産」として育てる役割を担っています
  - 今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-12.md`

### conditional_approval_checks / accepted_preview / evidence=2

- Rule: 条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。
- Type: risk_signal
- Confidence: 0.87
- User evidence: 1
- Axis: cash_flow
- Sample claims:
  - 今回は返済履歴が良好なため条件付き承認の余地はありますが、新規ルートの不確実性をどれだけ具体的に詰めるかが鍵となります
  - 今回との差分: 今回デモは返済履歴が良好なので、資料が揃えば条件付き承認の余地がある
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### conditional_approval_checks / candidate / evidence=1

- Rule: 条件付き承認では、不確実性を残したまま通さず、追加資料・確認条件・撤退条件を明文化する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 1
- Axis: n/a
- Sample claims:
  - この案件、条件付き承認にするなら何を確認すべき
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### industry_operating_risk / candidate / evidence=1

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 得た教訓: 運送業の境界案件は、車両価値よりも運賃改定・荷主継続・人員確保を先に確認する
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: user_preference
- Confidence: 0.79
- User evidence: 1
- Axis: asset_life, industry_risk
- Sample claims:
  - ラーメン屋の厨房機器はリース期間5年が多い 覚えておいて
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### industry_operating_risk / candidate / evidence=1

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: industry_risk, ai_ops
- Sample claims:
  - 経営者の経験と実績: 飲食業での経験や、過去の経営実績が重要な判断材料になります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 契約条件の調整可能性: - 期間短縮: 借手の返済能力に懸念がある場合、リース期間を短縮することで月額負担は増えますが、総リスク期間を圧縮できます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-11.md`

### industry_operating_risk / candidate / evidence=1

- Rule: 業種特有の倒産率、費用変動、人員確保、店舗・稼働継続性を案件の定性リスクとして確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: n/a
- Sample claims:
  - 運送業の特性上、燃料費・人件費の変動リスクは常に付きまとうため、数字以上の定性的な確認が求められます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: n/a
- Sample claims:
  - 今回は、その違和感を稟議で使える判断軸に落とし込む視点で深掘りしましょう
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md`
