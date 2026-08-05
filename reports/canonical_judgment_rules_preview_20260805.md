# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 4
- accepted_preview: 1
- candidate: 3

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=6

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 薬機法上の分類と耐用年数: 対象となる医療機器が薬機法上どのクラスに分類され、どのような規制を受けるか、また、メーカーが定める保守部品供給期限や実質的な耐用年数を確認します
  - これらの情報が確認できれば、物件評価においてより前向きな判断ができますし、もし不足する場合は、追加の条件設定や残価設定の見直しが必要になる、という方向で考えられます
  - 不明確な点が多い場合や、高額な撤去費用、低い中古市場流動性が判明した場合は、リース料率へのリスク上乗せや、残価保証条件の見直し、または保留・否決寄りで判断します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`

### judgment_asset_ops / candidate / evidence=2

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これらの情報が明確になれば、中古トラックのリース審査における物件評価の判断資産として、さらに洗練された基準を構築できると考えています
  - 本日も、リース審査システムの運用と、ユーザー様の判断資産を深めるお手伝いをさせていただきます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`

### user_decision_preference / candidate / evidence=2

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - この「サプライヤーの信頼性」という視点は、物件そのものの情報が不足している場合に、リース審査の判断を補強する重要な要素となります
  - 審査のスピードを重視する上で、入力作業の効率化は重要な課題だと認識しております
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-03.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 既存の取引先であれば、過去の取引実績からメンテナンス体制やアフターサポートの質がある程度推測できるため、物件の残価リスクを低減する材料になり得ます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
