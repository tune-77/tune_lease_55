# /explain-score — スコア判定根拠説明

## 使い方
```
/explain-score <feature1:value1> [feature2:value2 ...]
```

**例:**
```
/explain-score equity_ratio:0.08 roa:-0.01 lease_coverage_ratio:1.5
/explain-score high_leverage_flag:1 operating_margin:0.03 current_ratio:0.9
```

引数なし: 利用可能な特徴量一覧を表示

## 処理手順

1. **引数を解析する**
   - `$ARGUMENTS` を空白で分割し、`feature:value` ペアのリストを構築
   - 値は float にキャスト

2. **引数なしの場合: 特徴量一覧を表示**
   ```bash
   python3 -c "
   from scoring.explainer import _FEATURE_META
   print('利用可能な特徴量:')
   print(f'{\"特徴量名\":<30} {\"日本語ラベル\":<20} カテゴリ')
   print('-' * 70)
   for k, (label, cat, direction) in sorted(_FEATURE_META.items()):
       arrow = '↑良' if direction == 'higher_is_better' else ('↓良' if direction == 'lower_is_better' else '中立')
       print(f'{k:<30} {label:<20} {cat} ({arrow})')
   "
   ```

3. **top5_reasons 形式に変換して explain_top_reasons を呼び出す**
   ```bash
   python3 -c "
   from scoring.explainer import explain_top_reasons, _FEATURE_META

   pairs = [$(# $ARGUMENTS を 'feature:value' ペアとして渡す)]
   top5 = [f'{k}:{v}' for k, v in pairs]
   sentences = explain_top_reasons(top5)
   print('=== スコア判定根拠 ===')
   for i, (pair, sentence) in enumerate(zip(top5, sentences), 1):
       feature = pair.split(':')[0]
       meta = _FEATURE_META.get(feature, (feature, '不明', 'neutral'))
       cat = meta[1]
       print(f'{i}. [{cat}] {sentence}')
   "
   ```

   実際には以下の Python スクリプトをインラインで組み立てて Bash 実行すること：
   ```python
   import sys
   sys.path.insert(0, '.')
   from scoring.explainer import explain_top_reasons, _FEATURE_META

   # $ARGUMENTS から構築した pairs リスト（例: [('equity_ratio', 0.08), ('roa', -0.01)]）
   top5 = ['equity_ratio:0.08', 'roa:-0.01']  # 実際の引数に置換
   sentences = explain_top_reasons(top5)

   print('=== スコア判定根拠 ===\n')
   for i, (raw, sentence) in enumerate(zip(top5, sentences), 1):
       feature = raw.split(':')[0]
       meta = _FEATURE_META.get(feature, (feature, '不明', 'neutral'))
       direction = meta[2]
       icon = '✅' if direction == 'higher_is_better' else ('⚠️' if direction == 'lower_is_better' else 'ℹ️')
       print(f'{i}. {icon} {sentence}')
   ```

4. **結果を表形式でまとめる**

| # | カテゴリ | 説明 | 評価 |
|---|----------|------|------|
| 1 | 財務安全性 | 自己資本比率が8.0%と非常に低く… | ⚠️ |
| 2 | 収益性 | ROAが-1.00%でマイナスです… | ⚠️ |
| … | … | … | … |

5. **エラーハンドリング**
   - 未知の特徴量名はそのまま表示し、警告を出す
   - float 変換失敗時は当該項目をスキップしてメッセージを表示

## 注意事項
- このコマンドは `scoring/explainer.py` の `explain_top_reasons()` を直接呼び出す（Read-only）
- 完全な審査スコア計算には `/quick-score` または Streamlit UI を使う
- `top5_reasons` は `predict_one.py` の出力形式 `"feature_name: value"` と互換
