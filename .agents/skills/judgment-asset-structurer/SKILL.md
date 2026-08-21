---
name: judgment-asset-structurer
description: リース審査AIの判断資産候補を構造化するスキル。会話ログ、案件メモ、失敗/成功事例、User修正文、Obsidianノート、審査コメント案から「判断資産」「確認質問」「リスク信号」「条件付き承認条件」「反証材料」へ整理する時に使用する。ユーザーが「判断資産にして」「知識化」「候補化」「使える型に整理」「このメモを資産化」と言った時に使う。
---

# Judgment Asset Structurer

## Purpose

Messy notes are not judgment assets yet. Convert them into small, reviewable candidates that can later be reused in lease screening.

## Workflow

Reason: useful judgment assets must change a future check, condition, explanation, or rejection reason.
Scope: use when structuring notes into reusable lease judgment material.
Retirement: remove if a dedicated judgment-asset editor enforces these fields and review gates.

1. Identify the input type.
   - `case_observation`: fact from a lease case.
   - `user_judgment`: User's explicit screening view or correction.
   - `failure_signal`: concern that later proved important.
   - `success_signal`: useful confirmation or condition that worked.
   - `research_signal`: external information not yet proven in a real case.

2. Extract only actionable material.
   - Keep items that affect one of: confirmation question, risk origin, approval condition, rejection reason, monitoring point, or explanation wording.
   - Drop generic advice, obvious textbook statements, motivational text, and prose that does not change a screening action.

3. Structure each candidate with this schema.

```yaml
title:
asset_type: judgment_rule | confirmation_question | risk_signal | approval_condition | rejection_reason | counterpoint
source_type: case_observation | user_judgment | failure_signal | success_signal | research_signal
status: candidate
core_judgment:
when_to_apply:
do_not_apply_when:
case_signals:
recommended_action:
evidence_needed:
expected_outcome_to_verify:
source_note:
confidence: low | medium | high
review_reason:
```

4. Add reuse wording.
   - Write the candidate as a reusable pattern, not as a raw quote.
   - Include variables such as `{industry}`, `{asset}`, `{customer_stage}`, `{repayment_source}`, `{competition_status}` when helpful.

5. Keep promotion gated.
   - Mark research-only items as `candidate`, never `accepted`.
   - Do not claim a candidate is a long-term judgment asset unless User feedback or outcome verification supports it.
   - If the item came from User's edited wording, note `review_reason: user_edited_wording`.

## Output

Start with:

- `採用候補`: items that are actionable.
- `保留`: items that need evidence or are too broad.
- `捨てる`: items that are generic, unsafe, or non-actionable.

For each adopted candidate, output the YAML block and a one-line explanation of how it would change the next screening action.

## Constraints

- Do not include customer secrets, raw DB rows, personal evaluation material, or unnecessary personal details.
- Do not turn abusive or coercive language into reusable wording. Extract only the improvable factual issue.
- Do not connect candidates directly to scoring, approval/denial automation, RAG promotion, or Obsidian writeback unless the user explicitly asks and the relevant save/review skill also applies.
