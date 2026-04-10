# /asset-evaluation — 物件スコア詳細評価

## 使い方
```
/asset-evaluation <物件カテゴリ> [オプション...]
```

**物件カテゴリ一覧:**
`IT機器` / `産業機械` / `車両` / `医療機器` / `建設機械` / `工作機械` / `その他`

**オプション（スペース区切りの key=value 形式）:**
- `lease_months=<月数>` — リース期間（デフォルト: 36）
- `has_buyout_option=true` — 買取オプションあり
- `vehicle_fuel_type=EV` — 燃料種別（EV / HV / gasoline）
- `is_major_maker=true` — 大手メーカー品
- `age_years=<年数>` — 製造後年数

**例:**
```
/asset-evaluation 車両 lease_months=60 vehicle_fuel_type=EV
/asset-evaluation 医療機器 is_major_maker=true lease_months=48
/asset-evaluation IT機器 lease_months=36 has_buyout_option=true
```

引数なし: カテゴリ一覧と使い方を表示

## 処理手順

1. **引数を解析する**
   - `$ARGUMENTS` の先頭トークンを物件カテゴリとして取得
   - 残りの `key=value` トークンを contract_params dict に変換
   - bool 値は `"true"` → `True` に変換、数値は `int` にキャスト

2. **引数なし or カテゴリが不明の場合: カテゴリ一覧を表示**
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, '.')
   from category_config import CATEGORY_SCORE_ITEMS
   print('利用可能な物件カテゴリ:')
   for cat in CATEGORY_SCORE_ITEMS:
       items = CATEGORY_SCORE_ITEMS[cat]
       print(f'  {cat} ({len(items)}評価項目)')
   "
   ```

3. **`calc_asset_score()` を呼び出してスコアを計算**
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, '.')
   from asset_scorer import calc_asset_score, get_recommendation
   from category_config import ASSET_WEIGHT

   category = '<カテゴリ>'  # $ARGUMENTS 先頭
   contract = {'lease_months': 36}  # $ARGUMENTS の key=value から構築

   result = calc_asset_score(category, {}, contract)
   rec = get_recommendation(result['grade'])
   w = ASSET_WEIGHT.get(category, {})

   print(f'=== 物件スコア詳細評価: {category} ===\n')
   print(f'総合スコア : {result[\"total_score\"]:.1f} 点')
   print(f'グレード   : {result[\"grade\"]}')
   print(f'ウェイト調整: {\"あり\" if result[\"weight_adjusted\"] else \"なし\"}\n')

   print('--- 評価項目別スコア ---')
   for item, detail in result['item_scores'].items():
       bar = '█' * int(detail[\"score\"] / 10) + '░' * (10 - int(detail[\"score\"] / 10))
       print(f'{item:<20} {bar} {detail[\"score\"]:5.1f}点 (ウェイト:{detail[\"weight\"]:4.1f}%)')

   print(f'\n--- 推奨リース条件 ---')
   print(f'最長リース : {rec[\"max_lease_years\"]}年')
   print(f'推奨残価率 : {rec[\"residual_value_rate\"]:.0%}')

   if result.get('warnings'):
       print(f'\n--- 警告 ---')
       for w in result['warnings']:
           print(f'  ⚠️  {w}')

   print(f'\n--- 物件/借手ウェイト比率 ---')
   aw = ASSET_WEIGHT.get(category, {\"asset_w\": 0.5, \"obligor_w\": 0.5})
   print(f'物件スコア重み: {aw[\"asset_w\"]:.0%}  借手スコア重み: {aw[\"obligor_w\"]:.0%}')
   "
   ```

4. **結果を整形して表示**

```
=== 物件スコア詳細評価: 車両 ===

総合スコア : 78.5 点
グレード   : A
ウェイト調整: あり（EV割増適用）

評価項目別スコア:
  残存価値         ██████████  85.0点 (ウェイト:35.0%)
  流動性           ████████    70.0点 (ウェイト:25.0%)
  メンテナンス性   ████████    75.0点 (ウェイト:20.0%)
  ...

推奨リース条件:
  最長リース  : 5年
  推奨残価率  : 15%
```

5. **エラーハンドリング**
   - カテゴリが `CATEGORY_SCORE_ITEMS` に存在しない場合は一覧を表示して終了
   - オプション解析失敗時は該当オプションをスキップして処理継続

## 注意事項
- `scores` 引数には空 dict `{}` を渡す（手動スコアは未入力として動的計算）
- 借手スコアの計算は行わない（物件スコアのみ）
- `weight_adjusted: True` の場合、契約条件による動的補正が適用されている
