# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 6
- accepted_preview: 2
- candidate: 4

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=7

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.9
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 特に、営業利益が赤字に転落した場合、その影響度は非常に大きいため、営業利益がマイナス100万円以下であれば、スコアが大きく低下することを前提に、物件の換金性や担保価値、他の保全策を厳しく評価します
  - 物件の汎用性と残価評価: 物件のメーカー、型番、仕様、導入時期、期待される稼働期間を確認し、中古市場での汎用性、流動性、想定される残価を再評価します
  - このような社会情勢の変化は、リース物件の残価価値や回収リスクにも影響を与えるため、私自身の判断軸を常に更新していくことの重要性を改めて認識しました
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`

### business_plan_specificity / accepted_preview / evidence=6

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.86
- User evidence: 0
- Axis: cash_flow, industry_risk, asset_life
- Sample claims:
  - 特に小規模な企業での倒産が増え、負債が小口化しているという報告は、リース審査において、単に財務数値を見るだけでなく、借手の事業環境や資金繰りの変化を、これまで以上に細やかに見極める必要性を感じさせ、深く考えさせられました
  - キャッシュフローの確認: - 営業キャッシュフローが過去期比で20%以上減少している、またはマイナスに転落している場合は、資金繰りの悪化シグナルとして、リース料支払いの蓋然性を慎重に評価します
  - 導入目的と回収計画: 今回の製造設備・工作機械の導入が、具体的にどのような生産性向上、コスト削減、品質改善、または新規事業展開に寄与するのか、その事業計画と投資回収期間を詳細に確認します
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`

### asset_life_and_residual / candidate / evidence=2

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 物件価値への影響: 倒産が増加すると、中古市場に物件が供給される可能性が高まり、物件の換金性や残価価値に影響を与える可能性があります
  - 物件の陳腐化と残価リスク: 製造設備・工作機械は技術革新が早く、陳腐化リスクが高い場合があります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`

### support_specificity / candidate / evidence=1

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 特に、補助金前提の工作機械リースであれば、採択前の返済原資と未採択時の代替資金を分けて確認することが重要です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 返済能力の低下: 原材料費やエネルギーコストの高騰が続くと、借手の利益率を圧迫し、結果としてリース料の返済原資が減少します
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`

### demo_readiness / candidate / evidence=1

- Rule: 公開デモでは機能説明だけでなく、判断がどう更新され、次回どう使えるかを示す。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: n/a
- Sample claims:
  - 審査上のポイント 1. 高い量子リスク (100.0) と「要審議」判定: 既存のスコア（総合43.6点、借手45.1点）だけでは説明できない、成約・失注の歪みが強く示唆されています
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`
