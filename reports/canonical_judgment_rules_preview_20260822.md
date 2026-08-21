# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 8
- accepted_preview: 1
- candidate: 7

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=4

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life, industry_risk, cash_flow
- Sample claims:
  - 稟議で残すべき一文 Qriskは低いが、導入目的が不明瞭なため、物件の必要性、返済原資の確実性、および医療機器特有の技術陳腐化リスクと残価設定の妥当性を、銀行紹介元と連携して確認することを条件に承認を検討する
  - 物件の保全性と残価: 医療機器は技術陳腐化が早いため、中古市場での流通性、保守契約の有無、撤去・搬出費用を含めた残価設定の妥当性を確認します
  - ただし、この割合は物件の種類、借手の信用力、業種、そしてリース期間など、様々な要因で変動します
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`

### judgment_asset_ops / candidate / evidence=2

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.79
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これは、リース審査において、一度きりの判断を『勘』で終わらせず、検証可能で再利用可能な『判断資産』へと昇華させるための、私の存在意義そのものです
  - 既存設備の更新であれば、旧設備の稼働状況や処分予定も確認し、判断資産JA-0b5ce595にあるように、粗利改善見込みと整合するかを見ます
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - この知識は、補助金案件のリース審査において、採択有無だけでなく、入金時期や未採択時の返済余力、返還リスクといった点を確認する際の重要な視点となります
  - これらのニュースは、特に中小企業の資金繰りや補助金活用、そして半導体業界の動向がリース審査に影響を与える可能性を示唆しています
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-21.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 返済原資の確実性: 導入目的と連動した収益改善計画、または既存事業からの安定したキャッシュフローでリース料を賄えるか
  - 借手の審査において、資金繰りに関するヒアリングをより具体的に行い、潜在的なリスク要因を早期に特定する
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.81
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - 過去事例の「ダックス物流」と比較すると、あちらもスコア帯が近く、銀行支援が弱い中で資料不足で見送りとなりました
  - 特に、補助金活用を検討している場合は、採択の有無、入金時期、未採択時の返済余力を確認します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.92
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 紫苑の第一印象 2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較） 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 専門家としての深掘りルール: ・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 1
- Axis: n/a
- Sample claims:
  - ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-21.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.8
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - 現時点では、中小企業の資金繰り不安に対し、より多角的な視点での審査が必要であると判断します
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-20.md`
