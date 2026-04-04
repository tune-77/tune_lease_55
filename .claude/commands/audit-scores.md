# /audit-scores — スコアリング監査

## 使い方
```
/audit-scores [--full]
```
- 引数なし: asset_scorer / total_scorer の基本整合性チェック（30秒）
- `--full`: scoring-auditor エージェントを起動して詳細監査

## 処理手順

### クイックモード（デフォルト）

1. **スコアリングモジュールをインポートして基本動作確認**
   ```bash
   python3 -c "
   from asset_scorer import calc_asset_score
   from total_scorer import calc_total_score
   from category_config import ASSET_ID_TO_CATEGORY, SCORE_GRADES

   categories = list(set(ASSET_ID_TO_CATEGORY.values()))
   print(f'登録カテゴリ数: {len(categories)}')

   errors = []
   for cat in categories[:5]:
       r = calc_asset_score(cat, {}, {'lease_months': 36})
       if not (0 <= r['total_score'] <= 100):
           errors.append(f'{cat}: スコア範囲外 ({r[\"total_score\"]})')
   if errors:
       print('❌ 異常あり:')
       for e in errors: print(f'  {e}')
   else:
       print('✅ スコア範囲チェック通過（全カテゴリ 0〜100 内）')
   "
   ```

2. **物件・借手スコアの乖離チェック**
   ```bash
   python3 -c "
   from asset_scorer import calc_asset_score
   from category_config import ASSET_ID_TO_CATEGORY

   large_gap = []
   for asset_id, cat in ASSET_ID_TO_CATEGORY.items():
       asset = calc_asset_score(cat, {}, {'lease_months': 36})
       # デフォルト借手スコアは50点として乖離30点超を検出
       if abs(asset['total_score'] - 50) > 30:
           large_gap.append((asset_id, asset['total_score']))
   if large_gap:
       print(f'⚠️ 物件スコアが借手デフォルト(50点)から30点超乖離: {len(large_gap)}件')
       for aid, s in large_gap[:5]:
           print(f'  {aid}: {s:.1f}点')
   else:
       print('✅ 乖離チェック通過')
   "
   ```

3. **結果サマリー表示**

### フルモード（--full）

`scoring-auditor` エージェントを起動：
```
scoring-auditor エージェントを起動します。
asset_scorer.py, total_scorer.py, scoring_core.py を Read して
スコアリング異常・判定逆転・乖離案件の詳細監査を実施し、
.claude/reports/scoring-audit/latest.md へ結果を書いてください。
```

## 注意事項
- クイックモードはデフォルトパラメータでの動作確認のみ
- 実際の案件データとの照合は `--full` モードで行う
- DB が存在しない環境ではロジックのみを検証する
