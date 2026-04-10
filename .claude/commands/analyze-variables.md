# /analyze-variables — 変数重要度分析（IV/SHAP）

## 使い方
```
/analyze-variables [--iv] [--shap] [--top <N>]
```
- 引数なし: IV分析のみ実行（高速）
- `--iv`: IV（情報価値）分析を実行
- `--shap`: SHAP分析を実行（XGBoost学習あり、時間かかる）
- `--top <N>`: 上位N件を表示（デフォルト: 15）

**例:**
```
/analyze-variables
/analyze-variables --iv --top 20
/analyze-variables --shap
```

## 処理手順

### 前提チェック

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path

paths = [
    Path('past_cases.jsonl'),
    Path('../past_cases.jsonl'),
]
found = next((p for p in paths if p.exists()), None)
if found:
    import json
    cases = [json.loads(l) for l in open(found)]
    valid = [c for c in cases if c.get('final_status') in ('成約', '失注')]
    print(f'✅ データ: {found}  有効件数: {len(valid)}件')
    if len(valid) < 30:
        print(f'⚠️  件数が少ない（{len(valid)}件）。100件以上で信頼度が上がります。')
else:
    print('❌ past_cases.jsonl が見つかりません。データを登録してから実行してください。')
    exit(1)
"
```

### IV分析モード（デフォルト / `--iv`）

```bash
python3 scripts/iv_analysis.py 2>&1 | head -100
```

出力例:
```
分析対象: 127件  成約:89件  失注:38件

=== IV ランキング（成約/失注 判別力） ===
順位  特徴量                    IV値    評価
  1   equity_ratio            0.412   ★★★ 強い予測力
  2   operating_margin        0.289   ★★  中程度の予測力
  3   qualitative_tag_score   0.251   ★★  中程度の予測力
  4   lease_coverage_ratio    0.198   ★★  中程度の予測力
  5   grade                   0.156   ★★  中程度の予測力
  ...
```

### SHAP分析モード（`--shap`）

```bash
python3 scripts/shap_analysis.py 2>&1
```

- XGBoost モデルを学習（交差検証AUC も表示）
- 以下のグラフを `dashboard_images/shap/` に出力:
  - `summary_bar.png` — 変数重要度ランキング
  - `summary_beeswarm.png` — 影響方向・大きさ
  - `waterfall_*.png` — 個別案件の判定根拠

出力完了後:
```
✅ SHAP グラフ生成完了:
  - dashboard_images/shap/summary_bar.png
  - dashboard_images/shap/summary_beeswarm.png
  - dashboard_images/shap/waterfall_0.png (他N件)
```

### 結果の解釈ガイド

| IV値 | 評価 | アクション |
|------|------|-----------|
| < 0.02 | 使えない変数 | 係数から除外を検討 |
| 0.02-0.1 | 弱い予測力 | 補助的に使用 |
| 0.1-0.3 | 中程度の予測力 | 重要な説明変数 |
| > 0.3 | 強い予測力 | 係数の重み増加を検討 |

### 係数改善への連携

IV/SHAP 結果を踏まえて係数を調整したい場合:
```
/optimize-coefficients  # 自動最適化
/validate-rules         # 調整後の整合性チェック
```

## 注意事項
- `past_cases.jsonl` の `final_status` が `成約` または `失注` のデータのみ使用
- 30件未満ではIV値の信頼性が低い（目安: 100件以上）
- SHAP分析は xgboost・shap パッケージが必要（`pip install xgboost shap`）
- グラフは `/tmp/` ではなく `dashboard_images/shap/` に保存（gitignore 確認）
