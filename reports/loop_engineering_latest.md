# Loop Engineering Health

- Generated at: `2026-08-05T04:10:16`
- Status: `attention`
- Source coverage: 100.0%

## Improvement Loop
- Applied: 1
- Needs review: 0
- Failed: 0
- Review pressure: 0.0%

## Recursive Loop
- Canonical candidates: 5
- Ranked queue: 0
- Suppressed: 5
- Repeat issue rate: 0.0%
- Reuse rate: 100.0%
- Noise rate: 100.0%
- Churn rate: 80.0% (healthy dedup: 1, churn: 4)

## Prompt Feedback Loop
- Total: 345
- PDCA applied: 345 (100.0%)
- Previous response diffs: 126 (36.5%)

## Scoring Coefficients
- Status: `attention`
- Checked model keys: 6
- Borrower/asset weight sum: 1.0
- Quant/qual weight sum: 1.0
- Blend weight sum: 1.0
- [attention] /Users/kobayashiisaoryou/clawd/tune_lease_55/data/ml_rf_v4.pkl をロードできません: joblib: - got     : {'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'], 'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'], 'offsets': [0, 8, 16, 24, 32, 40, 48, 56], 'itemsize': 64} / pickle: _pickle.UnpicklingError: invalid load key, '\x09'.

## Guard / Safety
- Status: `warn`
- Codex queue: failed 0/1, carried over 0, aborted=False
- Preflight retries: over-budget 1 (max count 4 / limit 2)
- [warn] プリフライトのリトライ枠超過が 1 箇所（上限 2）— 人間へのバトンタッチを検討

## Outcome Loops
- Status: `ok`
- outcome_drift: 0 proposals (latest n/a)
- feedback_pattern: 27 proposals (latest 2026-08-05T03:00:27)
- judgment_divergence: 3 proposals (latest 2026-08-01T07:42:08)
- knowledge_gap: 3 proposals (latest 2026-07-03T07:21:31)
- PDCA effect: measured 275, improved 275, worsened 0, avg delta -0.273

## Recommendations
- スコアリング係数/モデルのヘルスチェックに重大な異常があります
- 抑制の滞留(churn)が高いため、needs_review/suppressed のクールダウン固着や台帳の suppressed 再記録を確認する（健全な重複排除は含めない）
- 安全ガードに警告: 日次上限の繰り越しやプリフライトのリトライ枠超過を確認する
