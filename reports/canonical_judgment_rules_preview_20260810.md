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

### asset_life_and_residual / accepted_preview / evidence=7

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.89
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 「ソフトリース」という言葉は、一般的にリース期間が物件の法定耐用年数よりも短い契約を指すことが多いですが、その契約形態にかかわらず、リース期間満了後に物件の利用を継続したい場合は、再リース契約を結ぶことができます
  - 特に、営業利益が赤字に転落した場合、その影響度は非常に大きいため、営業利益がマイナス100万円以下であれば、スコアが大きく低下することを前提に、物件の換金性や担保価値、他の保全策を厳しく評価します
  - このような社会情勢の変化は、リース物件の残価価値や回収リスクにも影響を与えるため、私自身の判断軸を常に更新していくことの重要性を改めて認識しました
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`

### business_plan_specificity / accepted_preview / evidence=4

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.86
- User evidence: 0
- Axis: asset_life, cash_flow, industry_risk
- Sample claims:
  - 特に小規模な企業での倒産が増え、負債が小口化しているという報告は、リース審査において、単に財務数値を見るだけでなく、借手の事業環境や資金繰りの変化を、これまで以上に細やかに見極める必要性を感じさせ、深く考えさせられました
  - キャッシュフローの確認: - 営業キャッシュフローが過去期比で20%以上減少している、またはマイナスに転落している場合は、資金繰りの悪化シグナルとして、リース料支払いの蓋然性を慎重に評価します
  - 現時点では、物価高による倒産増加は審査基準を厳格化する方向で、特に中小・零細企業や特定の業種に対しては、より詳細な財務・事業計画の確認が必要となるでしょう
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`

### business_plan_specificity / candidate / evidence=2

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.79
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - 返済能力の低下: 原材料費やエネルギーコストの高騰が続くと、借手の利益率を圧迫し、結果としてリース料の返済原資が減少します
  - このような状況では、返済原資と設備稼働開始の確認を優先すべきだと考えています
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`

### support_specificity / candidate / evidence=2

- Rule: 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
- Type: risk_signal
- Confidence: 0.83
- User evidence: 0
- Axis: cash_flow, support_specificity
- Sample claims:
  - もし補助金活用を前提としている場合、採択の確実性、入金時期、未採択時の返済計画、補助金返還リスクなどを慎重に確認する必要があります
  - 補助金状況: もし補助金を申請している場合、申請状況、交付決定の見込み、および未採択時の資金繰り計画
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### asset_life_and_residual / candidate / evidence=1

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 物件価値への影響: 倒産が増加すると、中古市場に物件が供給される可能性が高まり、物件の換金性や残価価値に影響を与える可能性があります
- Evidence paths:
  - `Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-08-09.md`
