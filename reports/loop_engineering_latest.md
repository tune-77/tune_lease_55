# Loop Engineering Health

- Generated at: `2026-07-22T04:08:00`
- Status: `attention`
- Source coverage: 100.0%

## Improvement Loop
- Applied: 0
- Needs review: 36
- Failed: 0
- Review pressure: 100.0%

## Recursive Loop
- Canonical candidates: 36
- Ranked queue: 0
- Suppressed: 36
- Repeat issue rate: 0.0%
- Reuse rate: 100.0%
- Noise rate: 100.0%

## Prompt Feedback Loop
- Total: 320
- PDCA applied: 320 (100.0%)
- Previous response diffs: 117 (36.6%)

## Scoring Coefficients
- Status: `attention`
- Checked model keys: 6
- Borrower/asset weight sum: 1.0
- Quant/qual weight sum: 1.0
- Blend weight sum: 1.0
- [attention] /Users/kobayashiisaoryou/clawd/tune_lease_55/data/ml_rf_v4.pkl をロードできません: joblib: - got     : {'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'], 'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'], 'offsets': [0, 8, 16, 24, 32, 40, 48, 56], 'itemsize': 64} / pickle: _pickle.UnpicklingError: invalid load key, '\x09'.

## Recommendations
- スコアリング係数/モデルのヘルスチェックに重大な異常があります
- needs_review が多いため、低リスク候補と高リスク候補を分けて棚卸しする
- noise_rate が高いため、重複候補と抑制ルールを確認する
