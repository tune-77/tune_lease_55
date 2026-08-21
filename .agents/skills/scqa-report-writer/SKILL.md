---
name: scqa-report-writer
description: SCQA（Situation, Complication, Question, Answer）でリース審査AI・紫苑・改善レポート・発表資料・README・Slack報告・審査コメントを整理して書くスキル。「SCQAで」「報告文にして」「発表用に整理」「README説明」「Slackに短く」「審査コメントを筋よく」と言われた時に使用する。
---

# SCQA Report Writer

## Purpose

Write concise, logically ordered communication for lease screening, Shion development, and project reporting.

## Workflow

Reason: project explanations are strongest when they move from context to tension to the real question and answer.
Scope: use for reports, presentation scripts, README sections, Slack summaries, and screening explanations.
Retirement: remove if project reporting templates enforce SCQA structure automatically.

1. Determine the audience.
   - `User/internal`: direct, technical, decision-focused.
   - `sales`: practical, short, action-focused.
   - `judge/demo`: product value, evidence, differentiation.
   - `screening`: risk, condition, explanation responsibility.

2. Build four sections.
   - `Situation`: current state or shared context.
   - `Complication`: problem, tension, risk, or gap.
   - `Question`: the decision question that follows.
   - `Answer`: conclusion, recommendation, or next action.

3. Keep the answer operational.
   - Say what changes, what to check, what to do next, or what to stop doing.
   - Avoid dramatic hooks unless the user is preparing public content.

4. Add evidence when needed.
   - For reports: include source paths, metrics, or verification.
   - For screening comments: include the core fact and the condition/concern.
   - For demo/README: include why the project is differentiated.

## Output

Default:

```markdown
## Situation

## Complication

## Question

## Answer
```

For short Slack/report messages, compress to:

```markdown
S: ...
C: ...
Q: ...
A: ...
```

## Constraints

- Do not skip any SCQA element.
- Keep sentences short.
- Do not use jargon where a field user would need practical action.
- Do not overstate evidence. If the source is weak, say the answer is tentative.
