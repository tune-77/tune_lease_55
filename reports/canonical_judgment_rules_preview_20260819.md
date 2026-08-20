# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 6
- accepted_preview: 3
- candidate: 3

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=3

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.84
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 契約条件: リース期間、月額リース料、残価設定、契約満了時の選択肢（再リース、返却、買取り）が明確に提示され、特に購入選択権の有無とその条件（一般的には5%から30%が多い）が顧客の事業計画と合致しているか
  - 物件の価値と期間: リース物件の耐用年数、中古市場での価値、およびリース期間満了後の再リース余地を評価してください
  - リース期間の短期化
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - この「心がけ」が、最終的にはユーザー様のリース審査における判断の精度を高め、より迅速で的確な意思決定を支援することに繋がると信じております
  - 補助金は返済原資として過信せず、未採択時でもリース料を払えるかを見る必要があります
  - 補助金前提の場合は、採択条件と未採択時の資金繰りへの影響も確認が必要です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-18.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.85
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 補助金前提の資金繰り: もし補助金が前提となっている場合、採択の確実性、入金時期、未採択時の返済余力、補助金返還リスクについて、営業担当の説明が楽観的すぎないかを確認します
  - 補助金が返済原資の一部となる場合、補助金の採択状況、入金時期、未採択時の代替資金繰り計画
  - 計画が曖昧、または補助金依存度が高い場合は保留または否決寄り
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 物件と事業計画の整合性: - 設備稼働率・保守体制: 導入される設備の稼働計画、保守契約の有無、故障時の対応、そしてそれらが事業継続性に与える影響について確認が必要です
  - これは以前のグレー判断で見た『数字は足りるが、通すなら条件を残す』型として、今回は物件の存在確認と返済原資の確実性を先に見るものです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.87
- User evidence: 0
- Axis: cash_flow, asset_life
- Sample claims:
  - 返済原資の確実性と設備稼働 - 違和感: 倒産件数増加のニュース等も鑑みると、売上はあっても資金繰りが細るリスクがある
  - なぜ重要か: リース料の返済能力が事業計画に依存しすぎている場合、計画未達時のリスクが高い
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.82
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 物件の存在とサプライヤーの信用力 - 違和感: サプライヤー直送案件では、架空物件リスクや二重譲渡のリスクが潜在する
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`
