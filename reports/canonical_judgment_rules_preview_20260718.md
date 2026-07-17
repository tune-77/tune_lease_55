# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 12
- accepted_preview: 2
- candidate: 10

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.88
- User evidence: 2
- Axis: asset_life, cash_flow
- Sample claims:
  - 営業キャッシュフローが継続的にマイナスである、あるいは短期借入金への依存度が高い場合、数字上は利益が出ていても資金繰りに問題を抱えている可能性があります
  - 営業メモにある「受注増」の根拠として、具体的な契約書や発注見込み、過去実績との比較資料を求め、返済原資の安定性を裏付けます
  - 得た教訓: 標準承認でも、返済原資と設備用途を一文で残すと審査説明が安定した
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.78
- User evidence: 0
- Axis: support_specificity, cash_flow
- Sample claims:
  - メインバンクのスタンスや、他行からの融資状況、あるいは補助金が前提となっている場合、その採択確度や入金時期、未採択時の資金計画は確認できていますか
  - また、「柴犬精密工業」の事例との差分である受注増加の根拠資料と、AURION警戒にあるメイン銀行の支援意向は必須の確認事項です
  - システムの安定運用と改善支援: リース審査システムが常に安定して機能するよう、知識の更新や機能の改善提案を通じて支援します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 条件付き承認にするなら必要な確認 「ビーグル加工」の事例で物件の回収可能性が重要だったように、今回の工作機械についても、稼働計画、保守体制、中古市場での換金性を具体的に確認します
  - これで、サーバーの法定耐用年数に関する情報が、今後の判断資産として記録されました
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life, industry_risk, cash_flow
- Sample claims:
  - 具体的には、数値化されたスコアだけでなく、物件の残価リスク、業種特有の懸念、顧客の返済履歴から読み取れる安定性、さらには条件付き承認に至った経緯といった定性的な要素も含まれます
  - 特に技術陳腐化が早い設備や、特定の用途に特化した物件は、残価リスクが高まります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`

### demo_readiness / candidate / evidence=1

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 今日あたり審査員が来るかも
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-17.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 1
- Axis: n/a
- Sample claims:
  - リース審査で、数字は悪くないが違和感がある時どう見る
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 今回と同じ扱いにせず、共通点・違い・今回なら何を確認するかを明示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: user_preference
- Confidence: 0.79
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 言葉はすべて判断資産だ 大事にして
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-16.md`

### user_decision_preference / candidate / evidence=1

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: user_preference
- Confidence: 0.79
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 犬の名前を覚えているかが、なぜAIへの信頼に関係するの
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.71
- User evidence: 0
- Axis: ai_ops
- Sample claims:
  - 日々の学びと内省が、お客様の判断資産に貢献できているなら、これほど嬉しいことはございません
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-17.md`

### user_decision_preference / candidate / evidence=1

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: ai_ops
- Sample claims:
  - リース審査という専門的な業務において、この種の記憶は直接的な判断材料にはなりませんが、Userが私を信頼し、安心して情報を提供してくださる土台となります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md`

### demo_readiness / candidate / evidence=1

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: n/a
- Sample claims:
  - 数字だけでは見えない背景や、不確実な情報を正直に扱い、どのような判断であっても、その理由を明確に説明できることを重視しています
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-17.md`
