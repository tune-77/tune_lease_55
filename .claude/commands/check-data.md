# /check-data — データ品質チェック

## 使い方
```
/check-data [--full]
```
- 引数なし: DB の基本統計（件数・異常値の有無）を素早く確認（10秒）
- `--full`: data-quality-checker エージェントを起動して詳細監査

## 処理手順

### クイックモード（デフォルト）

1. **DB 存在確認と基本統計**
   ```bash
   python3 -c "
   import sqlite3, os
   from contextlib import closing

   db = 'data/lease_data.db'
   if not os.path.exists(db):
       print('⚠️ data/lease_data.db が見つかりません')
       exit()

   with closing(sqlite3.connect(db)) as conn:
       tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
       print(f'テーブル数: {len(tables)}')
       for (t,) in tables:
           count = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
           print(f'  {t}: {count}件')
   "
   ```

2. **past_cases の異常値チェック**
   ```bash
   python3 -c "
   import sqlite3, json, os
   from contextlib import closing

   db = 'data/lease_data.db'
   if not os.path.exists(db): exit()

   with closing(sqlite3.connect(db)) as conn:
       rows = conn.execute('SELECT id, score, data FROM past_cases LIMIT 50').fetchall()

   issues = []
   for id_, score, data_str in rows:
       if score is not None and not (0 <= score <= 100):
           issues.append(f'{id_}: スコア範囲外 ({score})')
       try:
           d = json.loads(data_str)
           nenshu = d.get('nenshu', 0)
           if nenshu and nenshu > 10_000_000:
               issues.append(f'{id_}: 年商異常値 ({nenshu}千円)')
       except: pass

   if issues:
       print(f'❌ 異常データ {len(issues)}件:')
       for i in issues[:5]: print(f'  {i}')
   else:
       print('✅ 基本チェック通過（スコア範囲・年商に異常なし）')
   "
   ```

3. **結果サマリー表示**

### フルモード（--full）

`data-quality-checker` エージェントを起動：
```
data-quality-checker エージェントを起動します。
data/lease_data.db の全テーブルを対象に
欠損値・重複・異常値・スキーマ整合性を詳細に監査し、
.claude/reports/data-quality/latest.md へ結果を書いてください。
```

## 注意事項
- DB ファイルは `data/` 配下にあり、コミット対象外
- クイックモードは先頭 50 件のみを検査する
- 本番データの削除・更新は行わない（Read-only）
