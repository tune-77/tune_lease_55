# Loop Engineering Health

- Generated at: `2026-08-17T04:13:22`
- Status: `warn`
- Source coverage: 100.0%

## Improvement Loop
- Applied: 0
- Needs review: 0
- Failed: 0
- Review pressure: 0.0%

## Recursive Loop
- Canonical candidates: 7
- Ranked queue: 0
- Suppressed: 7
- Repeat issue rate: 0.0%
- Reuse rate: 100.0%
- Noise rate: 100.0%
- Churn rate: 100.0% (healthy dedup: 0, churn: 7)

## Prompt Feedback Loop
- Total: 362
- PDCA applied: 362 (100.0%)
- Previous response diffs: 128 (35.4%)

## Scoring Coefficients
- Status: `ok`
- Checked model keys: 6
- Borrower/asset weight sum: 1.0
- Quant/qual weight sum: 1.0
- Blend weight sum: 1.0
- No coefficient/model issues detected

## Guard / Safety
- Status: `warn`
- Codex queue: failed 0/1, carried over 0, aborted=False
- Preflight retries: over-budget 5 (max count 7 / limit 2)
- [warn] プリフライトのリトライ枠超過が 5 箇所（上限 2）— 人間へのバトンタッチを検討

## Outcome Loops
- Status: `ok`
- outcome_drift: 0 proposals (latest n/a)
- feedback_pattern: 53 proposals (latest 2026-08-15T03:00:24)
- judgment_divergence: 3 proposals (latest 2026-08-01T07:42:08)
- knowledge_gap: 3 proposals (latest 2026-07-03T07:21:31)
- PDCA effect: measured 1002, improved 1002, worsened 0, avg delta -0.18

## Recommendations
- 抑制の滞留(churn)が高いため、needs_review/suppressed のクールダウン固着や台帳の suppressed 再記録を確認する（健全な重複排除は含めない）
- 安全ガードに警告: 日次上限の繰り越しやプリフライトのリトライ枠超過を確認する
