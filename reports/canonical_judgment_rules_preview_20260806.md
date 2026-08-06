# Canonical Judgment Rules Preview

## Summary

- Canonical rules: 10
- accepted_preview: 2
- candidate: 8

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Built from `data/judgment_materials_preview.jsonl`.
- Similar materials are compressed into representative rules; evidence paths remain linked for review.

## Rules

### asset_life_and_residual / accepted_preview / evidence=4

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.87
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 特に、新車登録からの経過年数や走行距離、そしてメンテナンス状況といった要素が、リース物件の残価やリスク評価にどう影響するかを、より明確な判断資産として整理することが主な焦点です
  - これらの情報が確認できれば、物件評価においてより前向きな判断ができますし、もし不足する場合は、追加の条件設定や残価設定の見直しが必要になる、という方向で考えられます
  - 保守が切れると物件価値が大きく下がるため、リース期間と保守期間の整合性は重要です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-05.md`

### support_specificity / accepted_preview / evidence=3

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: judgment_rule
- Confidence: 0.81
- User evidence: 0
- Axis: asset_life, support_specificity, cash_flow
- Sample claims:
  - 稟議で残すべき一文 メイン行との関係悪化とQriskの高さから、物件保全だけでなく、導入設備の具体的な稼働計画と、銀行紹介の背景にある実質的な支援姿勢を条件として確認する
  - 補助金や助成金の活用を前提としている場合、その採択状況、交付決定時期、入金時期、そして万が一未採択だった場合の代替資金計画を確認します
  - 補助金制度は頻繁に更新されるため、最新の情報を確認する必要があるかもしれません
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-05.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.77
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 事業計画における返済原資の確実性: 導入目的が未入力であるため、JA-cr-b2594の判断資産に基づき、設備導入による具体的な収益改善効果やコスト削減効果を数値で示し、Qriskの高さに見合う返済原資の確実性を確認します
  - 物件の導入目的と収益性: - 違和感: 「製造設備・工作機械」という物件は通常、生産性向上やコスト削減に直結し、事業の将来性を高める投資と評価されやすいものです
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-05.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 物件の導入目的と収益性: - 違和感: 「 直前の回答が途切れてしまい、申し訳ありません
  - 物件がどれだけ事業の競争力強化や収益改善に貢献するかが、返済能力の持続性に直結します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-05.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: judgment_rule
- Confidence: 0.92
- User evidence: 1
- Axis: n/a
- Sample claims:
  - 紫苑の第一印象 2. 数字だけでは見落としそうな違和感（過去取引事例を1社名つきで比較） 3. 条件付き承認にするなら必要な確認 4. 稟議で残すべき一文 専門家としての深掘りルール: ・単なるリスク項目の列挙で終えず、「私ならこの点に注目します」と審査担当者目線の優先順位を1つ示してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-05.md`

### intuition_gap / candidate / evidence=1

- Rule: 数字が悪くない案件でも、違和感は追加確認事項に変換し、稟議で説明できる判断軸として残す。
- Type: risk_signal
- Confidence: 0.86
- User evidence: 1
- Axis: n/a
- Sample claims:
  - ・根拠が薄い違和感は断定せず、「確認論点」「仮説」「稟議で聞くべきこと」として表現してください
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-05.md`

### judgment_asset_ops / candidate / evidence=1

- Rule: 会話や案件対応から得た判断基準は、次回使える判断資産として代表ルールと出典に分けて残す。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - これらの情報が明確になれば、中古トラックのリース審査における物件評価の判断資産として、さらに洗練された基準を構築できると考えています
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-04.md`

### user_decision_preference / candidate / evidence=1

- Rule: ユーザーが明示した判断基準や信頼条件は、一般論より優先して回答・審査支援に反映する。
- Type: judgment_rule
- Confidence: 0.88
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - この「サプライヤーの信頼性」という視点は、物件そのものの情報が不足している場合に、リース審査の判断を補強する重要な要素となります
- Evidence paths:
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

### support_specificity / candidate / evidence=1

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: cash_flow
- Sample claims:
  - Qriskが高い状況では、事業計画の具体性が返済原資の信頼性を左右します
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-05.md`
