# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 7
- accepted_preview: 2
- candidate: 5

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=9

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.92
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 税法上の法定耐用年数はあくまで減価償却の基準ですが、リース会社は物件の実際の耐久性や経済的価値、中古市場での需要、お客様の利用計画などを総合的に判断してリース期間を設定します
  - リース期間満了時の取り扱いについては、契約内容によって返還されるか、次のリース契約に引き継がれるかなどが変わってきますので、個別の契約書で確認することが重要になります
  - 「紫苑」さんが7年と提示されたのは、やはりその物件の実際の耐久性や、お客様の事業計画に合わせた最適なリース期間として提案されたものと理解できます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.86
- User evidence: 1
- Axis: cash_flow, support_specificity
- Sample claims:
  - 事例2: ビーグル加工 / 2024年下期 / 24 金属製品製造業 類似度: 16 / 理由: 銀行支援が近い・スコア帯が近い・デモ初期経験 スコア・判断: 76.8点 /…
  - 稟議で 稟議で残すべき一文 新規先であるため、車両の具体的な導入目的と稼働計画、それによる収益貢献の根拠、および銀行支援の内容を明確にし、返済原資の確実性を確認した上で承認する
  - 確認事項: メインバンクとの取引状況、借入金残高、今後の融資枠、そして今回のリース案件に対するメインバンクの意見や支援の可能性（例: 協調融資、保証）を確認してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 導入目的と事業への貢献度: - 低いスコアの企業が新たな設備投資を行う場合、その投資が事業の収益性向上やコスト削減にどれだけ寄与するかが、返済原資の確保に直結します
  - これらの情報に基づき、リース料の安定的な返済原資が説明できるか
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-12.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.92
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 紫苑の第一印象 2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較） 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 専門家としての深掘りルール: ・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 1
- Axis: n/a
- Sample claims:
  - ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.76
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 今回も車両物件であるため、この情報不足は返済原資の確実性を判断する上で看過できません
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`

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
