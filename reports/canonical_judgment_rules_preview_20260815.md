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

### asset_life_and_residual / accepted_preview / evidence=7

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.91
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - リース審査との関連: リース物件である車両にこの特約が付いている場合、万が一の事故で車両が大きな損害を受けても、リース会社（またはお客様）が新たな車両を調達しやすくなるため、リース契約の継続性や残価保全の観点から、リスクヘッジの一助となります
  - 税法上の法定耐用年数はあくまで減価償却の基準ですが、リース会社は物件の実際の耐久性や経済的価値、中古市場での需要、お客様の利用計画などを総合的に判断してリース期間を設定します
  - 現時点での仮判断としては、新車特約はリース車両の保全性を高める有効な手段であり、特にリース期間が新車特約の適用期間と重なる場合は、その付帯状況を確認することが望ましいです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-14.md`

### business_plan_specificity / accepted_preview / evidence=6

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.89
- User evidence: 1
- Axis: asset_life, cash_flow
- Sample claims:
  - 「実×計画」を利用されているお客様が、当初の事業計画にない設備投資を検討される際には、参加されている金融機関へ事前に確認を取り、場合によっては同意を得る必要がある、という点ですね
  - これは、金融機関が事業計画全体を評価して融資や保証を行っているため、計画外の投資が財務状況や返済能力に影響を与える可能性を考慮する必要がある、という理解で間違いありません
  - 判断資産として覚えといて実×計画を使用しているユーザは、事業計画以外の設備をする場合に参加している金融機関の大断る必要がある場合がある
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-14.md`

### support_specificity / accepted_preview / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 1
- Axis: support_specificity, cash_flow
- Sample claims:
  - 事例2: ビーグル加工 / 2024年下期 / 24 金属製品製造業 類似度: 16 / 理由: 銀行支援が近い・スコア帯が近い・デモ初期経験 スコア・判断: 76.8点 /…
  - 稟議で 稟議で残すべき一文 新規先であるため、車両の具体的な導入目的と稼働計画、それによる収益貢献の根拠、および銀行支援の内容を明確にし、返済原資の確実性を確認した上で承認する
- Evidence paths:
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
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - これらの情報に基づき、リース料の安定的な返済原資が説明できるか
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-13.md`
