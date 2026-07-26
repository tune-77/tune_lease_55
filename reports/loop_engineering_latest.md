# Loop Engineering Health

- Generated at: `2026-07-26T04:09:33`
- Status: `attention`
- Source coverage: 100.0%

## Improvement Loop
- Applied: 0
- Needs review: 0
- Failed: 0
- Review pressure: 0.0%

## Recursive Loop
- Canonical candidates: 0
- Ranked queue: 0
- Suppressed: 0
- Repeat issue rate: 0.0%
- Reuse rate: 0.0%
- Noise rate: 0.0%
- Churn rate: 0.0% (healthy dedup: 0, churn: 0)

## Prompt Feedback Loop
- Total: 330
- PDCA applied: 330 (100.0%)
- Previous response diffs: 120 (36.4%)

## Scoring Coefficients
- Status: `attention`
- Checked model keys: 6
- Borrower/asset weight sum: 1.0
- Quant/qual weight sum: 1.0
- Blend weight sum: 1.0
- [attention] /Users/kobayashiisaoryou/clawd/tune_lease_55/data/ml_rf_v4.pkl をロードできません: joblib: - got     : {'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'], 'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'], 'offsets': [0, 8, 16, 24, 32, 40, 48, 56], 'itemsize': 64} / pickle: _pickle.UnpicklingError: invalid load key, '\x09'.

## Guard / Safety
- Status: `ok`
- Codex queue: failed 0/1, carried over 0, aborted=False
- Preflight retries: over-budget 0 (max count 2 / limit 2)
- No guard activations detected

## Outcome Loops
- Status: `ok`
- outcome_drift: 0 proposals (latest n/a)
- feedback_pattern: 6 proposals (latest 2026-07-26T03:00:13)
- judgment_divergence: 0 proposals (latest n/a)
- knowledge_gap: 3 proposals (latest 2026-07-03T07:21:31)
- PDCA effect: measured 0, improved 0, worsened 0, avg delta 0.0

## Recommendations
- スコアリング係数/モデルのヘルスチェックに重大な異常があります
