# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 5
- accepted_preview: 2
- candidate: 3

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=5

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.89
- User evidence: 0
- Axis: asset_life, industry_risk, cash_flow, support_specificity
- Sample claims:
  - ② 「事実」と「スキル」の抽出（Microsoftの視点） - ログではなく「判断資産」を記憶: ユーザーとの対話履歴や個別の審査案件のログをそのまま保存するのではなく、そこから抽出される「リース審査に関する事実（例：ラーメン屋の厨房機器はリース期間5年が多い）」や「判断スキル（例：補助金案件は未採択時の返済余力も見る）」を記憶として残すことが、記憶の「効用」を高める上で不可欠です
  - リース期間満了時の取り扱いについては、契約内容によって返還されるか、次のリース契約に引き継がれるかなどが変わってきますので、個別の契約書で確認することが重要になります
  - リース期間が物件の経済的耐用年数や陳腐化サイクルと乖離していないか、また、リース終了時の残価設定が妥当かを見極める必要があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: support_specificity, cash_flow
- Sample claims:
  - 確認事項: メインバンクとの取引状況、借入金残高、今後の融資枠、そして今回のリース案件に対するメインバンクの意見や支援の可能性（例: 協調融資、保証）を確認してください
  - 行動経済学の知見は、これらのバイアスを認識し、審査プロセスやチェックリストを改善することで、より客観的で一貫性のある判断を支援するのに役立ちます
  - しかし、補助金は採択の確実性、入金時期、未採択時の資金繰りへの影響を慎重に評価する必要があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`

### user_decision_preference / candidate / evidence=1

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: n/a
- Sample claims:
  - これは、リース審査の「スピード」というユーザーからの要請（「リースに必要なものは何よりもスピードだ」という記憶）に応える上で、非常に重要な視点です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 物件特性（IT・OA機器）と陳腐化リスク: - IT・OA機器は陳腐化が早く、中古市場での残価価値が急速に低下する傾向があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 導入目的と事業への貢献度: - 低いスコアの企業が新たな設備投資を行う場合、その投資が事業の収益性向上やコスト削減にどれだけ寄与するかが、返済原資の確保に直結します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
