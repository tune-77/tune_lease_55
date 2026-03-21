# /quick-score — クイックスコアリング

## 使い方
```
/quick-score <業種コード> <売上高（千円）> <リース額（万円）> [物件ID]
```

**例:**
```
/quick-score 44 50000 500 vehicle
/quick-score 09 200000 2000 construction_machine
```

## 処理手順

1. **引数を解析する**
   - `$ARGUMENTS` を空白で分割: `[業種コード, 売上高, リース額, 物件ID(省略可)]`
   - 物件IDが省略された場合は `"other"` として扱う

2. **`scoring_core.py` の `quick_score()` または `calc_asset_score()` を呼び出す**
   - Bash ツールで以下を実行:
   ```python
   python3 -c "
   from asset_scorer import calc_asset_score, get_recommendation
   from category_config import ASSET_ID_TO_CATEGORY, SCORE_GRADES

   asset_id = '<物件ID>'
   category = ASSET_ID_TO_CATEGORY.get(asset_id, None)
   if category:
       result = calc_asset_score(category, {}, {'lease_months': 36})
       rec = get_recommendation(result['grade'])
       print(f'物件グレード: {result[\"grade\"]} ({result[\"total_score\"]:.1f}点)')
       print(f'推奨最長リース: {rec[\"max_lease_years\"]}年')
       print(f'warnings: {result[\"warnings\"]}')
   else:
       print(f'物件ID \"{asset_id}\" は未登録 → デフォルト50点')
   "
   ```

3. **結果を表形式で表示する**

| 項目 | 値 |
|------|-----|
| 入力業種 | <コード> |
| 入力売上高 | <千円単位で表示>千円 |
| 物件カテゴリ | <解決されたカテゴリ名> |
| 物件グレード | <S/A/B/C/D> |
| 物件スコア | <XX.X点> |
| 推奨最長リース年数 | <N年> |
| 推奨残価率 | <XX%> |
| 警告 | <warningsリスト> |

4. **エラーハンドリング**
   - 引数が不足している場合は使い方を表示
   - 物件IDが未登録の場合は `ASSET_ID_TO_CATEGORY` の一覧を表示

## 注意事項
- このコマンドは **物件スコアのみ** を計算する（借手スコアは計算しない）
- 完全な審査は Streamlit UI または Slack ウィザードで行う
- 数値は「千円」単位で入力すること（`scoring/` モジュールは「円」換算で処理）
