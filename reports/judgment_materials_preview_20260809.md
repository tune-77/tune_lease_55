# Judgment Materials Preview (2026-08-07 to 2026-08-09)

## Summary

- Materials: 14
- judgment_rule: 10
- risk_signal: 4
- user_preference: 0

## Safety

- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.
- Sources are recent Cloud Run Conversation Log and Lease Intelligence Dialogue notes.
- Private Reflection is intentionally excluded from this extractor.

## Materials

### 2026-08-07 / judgment_rule / confidence=0.74

- Claim: 「法人車両としてBMWをリースする理由」について、私のリース審査専門WikiやObsidian Vaultの業務記録からは、業務部の具体的な視点での直接的な情報は見つかりませんでした
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: n/a
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-07.md`

### 2026-08-07 / judgment_rule / confidence=0.74

- Claim: これらの要素は、業務部が車両選定やリース契約を検討する際に、総合的な判断材料として重視する点と考えられます
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: n/a
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-07.md`

### 2026-08-08 / judgment_rule / confidence=0.82

- Claim: 「ソフトリース」という言葉は、一般的にリース期間が物件の法定耐用年数よりも短い契約を指すことが多いですが、その契約形態にかかわらず、リース期間満了後に物件の利用を継続したい場合は、再リース契約を結ぶことができます
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: asset_life
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / judgment_rule / confidence=0.74

- Claim: これにより、借手にとってはリース料の負担が増え、設備投資の判断に影響を与える可能性があります
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: asset_life
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / judgment_rule / confidence=0.74

- Claim: ただし、これはあくまで一般的な法定耐用年数であり、実際のリース期間や税務上の取り扱いは、個別の契約内容や最新の税制によって変わる可能性があります
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: asset_life
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / judgment_rule / confidence=0.82

- Claim: ファイナンスリースであっても、借手の事業計画における物件の陳腐化影響を確認する必要があります
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: asset_life
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / judgment_rule / confidence=0.78

- Claim: リース期間満了時の残価設定にも慎重な検討が必要です
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: asset_life
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / judgment_rule / confidence=0.78

- Claim: 借手の返済能力: リース料率の上昇は、借手の月々の支払い負担を増やすため、返済能力をより厳しく評価する必要があります
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: cash_flow
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / judgment_rule / confidence=0.7

- Claim: 審査上のポイント: - 業種特性: 「06 総合工事業」は、過去案件DB統計上、成約率64.9%、平均スコア61.0と比較的安定した業種です
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: industry_risk
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / judgment_rule / confidence=0.78

- Claim: 現時点では、金利上昇はリース料率の上昇を通じて借手の返済能力に影響を与え、審査をより慎重にする必要があると考えています
- Use when: 類似案件の判断理由や稟議コメントを作るとき
- Axis: cash_flow
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / risk_signal / confidence=0.76

- Claim: このような状況では、返済原資と設備稼働開始の確認を優先すべきだと考えています
- Use when: 案件の見落としリスクや追加確認事項を洗い出すとき
- Axis: asset_life, cash_flow
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / risk_signal / confidence=0.84

- Claim: もし補助金活用を前提としている場合、採択の確実性、入金時期、未採択時の返済計画、補助金返還リスクなどを慎重に確認する必要があります
- Use when: 外部支援を返済原資や保全材料として扱うとき
- Axis: cash_flow, support_specificity
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / risk_signal / confidence=0.76

- Claim: 物件の陳腐化リスク: 「IT・OA機器」は技術革新が速く、陳腐化リスクが高い物件です
- Use when: 案件の見落としリスクや追加確認事項を洗い出すとき
- Axis: asset_life
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`

### 2026-08-08 / risk_signal / confidence=0.76

- Claim: 補助金状況: もし補助金を申請している場合、申請状況、交付決定の見込み、および未採択時の資金繰り計画
- Use when: 外部支援を返済原資や保全材料として扱うとき
- Axis: cash_flow, support_specificity
- Evidence: `Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-08-08.md`
