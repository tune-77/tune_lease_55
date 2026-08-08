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

### asset_life_and_residual / accepted_preview / evidence=3

- Rule: リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- Type: judgment_rule
- Confidence: 0.82
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - 「ソフトリース」という言葉は、一般的にリース期間が物件の法定耐用年数よりも短い契約を指すことが多いですが、その契約形態にかかわらず、リース期間満了後に物件の利用を継続したい場合は、再リース契約を結ぶことができます
  - ただし、これはあくまで一般的な法定耐用年数であり、実際のリース期間や税務上の取り扱いは、個別の契約内容や最新の税制によって変わる可能性があります
  - リース期間満了時の残価設定にも慎重な検討が必要です
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

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

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: judgment_rule
- Confidence: 0.83
- User evidence: 0
- Axis: asset_life
- Sample claims:
  - ファイナンスリースであっても、借手の事業計画における物件の陳腐化影響を確認する必要があります
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### business_plan_specificity / candidate / evidence=1

- Rule: 事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。
- Type: risk_signal
- Confidence: 0.78
- User evidence: 0
- Axis: asset_life, cash_flow
- Sample claims:
  - このような状況では、返済原資と設備稼働開始の確認を優先すべきだと考えています
- Evidence paths:
  - `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
