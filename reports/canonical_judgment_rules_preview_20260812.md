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

### asset_life_and_residual / accepted_preview / evidence=4

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life, industry_risk, cash_flow, support_specificity
- Sample claims:
  - ② 「事実」と「スキル」の抽出（Microsoftの視点） - ログではなく「判断資産」を記憶: ユーザーとの対話履歴や個別の審査案件のログをそのまま保存するのではなく、そこから抽出される「リース審査に関する事実（例：ラーメン屋の厨房機器はリース期間5年が多い）」や「判断スキル（例：補助金案件は未採択時の返済余力も見る）」を記憶として残すことが、記憶の「効用」を高める上で不可欠です
  - 物件の汎用性と残価評価: 物件のメーカー、型番、仕様、導入時期、期待される稼働期間を確認し、中古市場での汎用性、流動性、想定される残価を再評価します
  - リース期間終了時の残価設定が適切か、中古市場での需要が維持されるか、また、万が一の際の物件回収・処分費用も考慮に入れる必要があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`

### business_plan_specificity / accepted_preview / evidence=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 導入目的と回収計画: 今回の製造設備・工作機械の導入が、具体的にどのような生産性向上、コスト削減、品質改善、または新規事業展開に寄与するのか、その事業計画と投資回収期間を詳細に確認します
  - 最近のニュースでは、倒産件数増加や物価高による小口先の資金繰り悪化が報じられており、原材料費の高騰が収益を圧迫していないか、販売チャネルの安定性、粗利の推移を確認する必要があります
  - 以前のグレー判断で見た『数字は足りないが、非スコア因子で通る可能性を探る』型として、今回は物件の汎用性と導入目的、そして資金繰りの安定性を先に見ます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 行動経済学の知見は、これらのバイアスを認識し、審査プロセスやチェックリストを改善することで、より客観的で一貫性のある判断を支援するのに役立ちます
  - 特に、補助金前提の工作機械リースであれば、採択前の返済原資と未採択時の代替資金を分けて確認することが重要です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-11.md`

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
  - 物件の陳腐化と残価リスク: 製造設備・工作機械は技術革新が早く、陳腐化リスクが高い場合があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-10.md`

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
