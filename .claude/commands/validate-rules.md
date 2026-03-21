# /validate-rules — ビジネスルール整合性チェック

## 使い方
```
/validate-rules [--full]
```
- 引数なし: ウェイト合計・グレード閾値の数値検証のみ（10秒）
- `--full`: rule-validator エージェントを起動して完全検証（数分）

## 処理手順

### クイックモード（デフォルト）

1. **CATEGORY_SCORE_ITEMS ウェイト合計チェック**
   ```bash
   python3 -c "
   from category_config import CATEGORY_SCORE_ITEMS, ASSET_WEIGHT, SCORE_GRADES

   # ウェイト合計チェック
   print('=== CATEGORY_SCORE_ITEMS ウェイト合計 ===')
   all_ok = True
   for cat, items in CATEGORY_SCORE_ITEMS.items():
       total = sum(i['weight'] for i in items)
       status = '✅' if total == 100 else '❌'
       if total != 100: all_ok = False
       print(f'{status} {cat}: {total}（期待値: 100）')

   # ASSET_WEIGHT チェック
   print('\n=== ASSET_WEIGHT 合計 ===')
   for cat, w in ASSET_WEIGHT.items():
       total = w['asset_w'] + w['obligor_w']
       status = '✅' if abs(total - 1.0) < 1e-9 else '❌'
       if abs(total - 1.0) >= 1e-9: all_ok = False
       print(f'{status} {cat}: asset={w[\"asset_w\"]} + obligor={w[\"obligor_w\"]} = {total}')

   # SCORE_GRADES 順序チェック
   print('\n=== SCORE_GRADES 閾値順序 ===')
   prev = 101
   for g in SCORE_GRADES:
       status = '✅' if g['min'] < prev else '❌'
       if g['min'] >= prev: all_ok = False
       print(f'{status} {g[\"label\"]}: min={g[\"min\"]}（前: {prev}）')
       prev = g['min']

   print(f'\n{'✅ 全チェック通過' if all_ok else '❌ 問題あり（詳細は上記）'}')
   "
   ```

2. **asset_scorer の動的ウェイト正規化チェック**
   ```bash
   python3 -c "
   from asset_scorer import calc_asset_score

   test_cases = [
       ('IT機器', {}, {'lease_months': 60}),
       ('産業機械', {}, {'has_buyout_option': True}),
       ('車両', {}, {'vehicle_fuel_type': 'EV', 'lease_months': 60}),
       ('医療機器', {}, {'is_major_maker': True}),
   ]
   print('=== 動的ウェイト正規化チェック ===')
   for cat, scores, contract in test_cases:
       r = calc_asset_score(cat, scores, contract)
       total_w = sum(v['weight'] for v in r['item_scores'].values())
       status = '✅' if abs(total_w - 100) < 0.1 else '❌'
       print(f'{status} {cat} (adjusted={r[\"weight_adjusted\"]}): ウェイト合計={total_w:.1f}')
   "
   ```

3. **結果を表示して問題があれば詳細を案内**

### フルモード（--full）

`rule-validator` エージェントを起動：
```
Agent(rule-validator): rule_manager.py, coeff_definitions.py, scoring_core.py の
完全な整合性検証を実行し、衝突・デッドルール・未テスト境界値を報告してください。
```

## 注意事項
- クイックモードは `category_config.py` と `asset_scorer.py` のみを対象とする
- `rule_manager.py` の複雑なロジック検証はフルモード（または rule-validator エージェント）で行う
- このコマンドはコードを変更しない（Read-only）
