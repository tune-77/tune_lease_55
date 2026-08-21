---
name: research-to-screening-insights
description: 調査メモ、業界ニュース、Obsidian Auto Research、Web情報、統計、レポートをリース審査で使える確認点へ変換するスキル。「この調査を審査に使う」「リース判断に落とす」「確認質問にして」「業界動向から審査ポイント」「Auto Researchを材料化」と言われた時に使用する。
---

# Research To Screening Insights

## Purpose

Turn research into a small set of screening actions. The output should help decide what to ask, what to watch, and how to explain the risk.

## Workflow

Reason: research is material, not memory. It must become case-facing action before it is useful.
Scope: use when converting research/news/notes into lease screening insight.
Retirement: remove if the Auto Research pipeline reliably emits reviewed screening insight records.

1. Summarize the research in one paragraph.
   - Include the topic, date range, source strength, and affected industries/assets.

2. Extract screening implications.
   - `repayment_source`: revenue stability, order pipeline, margin pressure, cash collection.
   - `asset_value`: resale value, obsolescence, utilization, collateral concern.
   - `customer_behavior`: renewal, cancellation, investment timing, competitor movement.
   - `funding_cost`: interest rate, spreads, lease pricing pressure.
   - `regulation_subsidy`: legal change, subsidy expiry, compliance risk.

3. Convert into at most three high-signal checks.
   - Prefer questions that can be answered by the borrower, salesperson, bank, or documents.
   - Avoid long generic checklists.

4. Produce judgment-asset candidates only when appropriate.
   - If the insight changes a future confirmation action, output a candidate compatible with `judgment-asset-structurer`.
   - Mark source-only insights as `research_signal` and `status: candidate`.

5. Separate confidence.
   - `confirmed`: supported by strong source or multiple consistent sources.
   - `plausible`: useful for questions, not for conclusions.
   - `weak`: background only.

## Output

```markdown
## Research Summary

## Screening Implications
- ...

## Top Checks
1. Question:
   Why:
   Evidence to request:
   If yes:
   If no:

## Comment Draft

## Judgment Asset Candidates
```

## Constraints

- Do not create more than three top checks unless the user asks for a full checklist.
- Do not promote research directly into long-term memory or active scoring.
- Do not make approval/denial recommendations from macro research alone.
- Use `lease-source-validator` first when the source quality is uncertain or the topic is current.
