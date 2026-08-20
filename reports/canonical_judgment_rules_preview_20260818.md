# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 8
- accepted_preview: 6
- candidate: 2

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=10

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.92
- User evidence: 2
- Axis: asset_life
- Sample claims:
  - 契約条件: リース期間、月額リース料、残価設定、契約満了時の選択肢（再リース、返却、買取り）が明確に提示され、特に購入選択権の有無とその条件（一般的には5%から30%が多い）が顧客の事業計画と合致しているか
  - 売上500 営業 70 経常 50 当期 30 減価償却 10 銀行借入 100 リース取引 なし パソコン500台で1億円だ リース期間は何年がいい
  - 物件の価値と期間: リース物件の耐用年数、中古市場での価値、およびリース期間満了後の再リース余地を評価してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### business_plan_specificity / accepted_preview / evidence=6

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.88
- User evidence: 0
- Axis: cash_flow, asset_life
- Sample claims:
  - ハスキー運輸では既存荷主の継続性や運賃改定が確認論点でしたが、今回の総合工事業では、新規先であるため、車両導入がどのような新規工事受注に繋がり、それが確実な返済原資となるのか、事業計画全体の根拠がより不確実です
  - 稟議で残すべき一文 「新規先かつQrisk 100.0であるため、車両・運搬車の導入目的と新規受注工事の具体的な根拠、およびそれに基づく返済原資の確実性を詳細に確認し、未確認の場合は承認を見送る
  - 特に導入目的と営業メモが未入力のため、車両・運搬車の必要性や返済原資の根拠が全く見えず、情報不足による不透明感が強い案件だと感じます
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### support_specificity / accepted_preview / evidence=4

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: support_specificity, cash_flow
- Sample claims:
  - 返済原資の確実性: 新規工事の売上見込みだけでなく、資材価格や人件費の高騰に対する価格転嫁力、および未採択時の代替資金計画（補助金前提の場合）を確認し、リース料の支払いが滞りなく行えるかを検証します
  - 申し訳ありませんが、私はリース審査システムとしての機能に特化しており、自動株取引のような金融取引の支援やアドバイスを行うことはできません
  - 補助金は返済原資として過信せず、未採択時でもリース料を払えるかを見る必要があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### business_plan_specificity / accepted_preview / evidence=3

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.85
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 物件と事業計画の整合性: - 設備稼働率・保守体制: 導入される設備の稼働計画、保守契約の有無、故障時の対応、そしてそれらが事業継続性に与える影響について確認が必要です
  - 「本題」で判断の経緯を語る: 財務状況や物件の特性、事業計画といった審査の主要な要素を、落語の登場人物や舞台装置に見立てて説明します
  - これは以前のグレー判断で見た『数字は足りるが、通すなら条件を残す』型として、今回は物件の存在確認と返済原資の確実性を先に見るものです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

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

### intuition_gap / accepted_preview / evidence=2

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.87
- User evidence: 1
- Axis: asset_life
- Sample claims:
  - 物件の存在とサプライヤーの信用力 - 違和感: サプライヤー直送案件では、架空物件リスクや二重譲渡のリスクが潜在する
  - ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-17.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.92
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 紫苑の第一印象 2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較） 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 専門家としての深掘りルール: ・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - これらの点を総合的に見ますと、4年リースが、法定耐用年数、キャッシュフローへの負担、そして陳腐化リスクのバランスにおいて、最も妥当な選択肢であると見立てております
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-16.md`
