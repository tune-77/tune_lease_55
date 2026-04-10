# /case-similarity — 類似事例検索

## 使い方
```
/case-similarity [--industry <業種コード>] [--score <スコア>] [--top <N>]
```
- 引数なし: 直近登録案件を基準に類似事例を検索
- `--industry <コード>`: 業種大分類コード（例: `13` 製造業）で絞り込み
- `--score <値>`: 基準スコアを指定（0-100）
- `--top <N>`: 上位N件を表示（デフォルト: 10）

**例:**
```
/case-similarity
/case-similarity --industry 13 --score 72 --top 5
/case-similarity --score 65 --top 15
```

## 処理手順

1. **データ読み込みと基準案件の特定**
   ```bash
   python3 -c "
   import sys, json; sys.path.insert(0, '.')
   from data_cases import load_past_cases

   cases = load_past_cases()
   print(f'過去案件数: {len(cases)}件')
   if not cases:
       print('❌ 過去案件がありません。成約/失注を登録してから実行してください。')
       exit(1)
   "
   ```

2. **類似度計算ロジック（`case_network.py` の `_similarity()` を活用）**

   類似度スコア = 業種一致(0.4) + スコア近接(0-0.4) + 競合他社名一致(0.2)

   ```bash
   python3 -c "
   import sys, json; sys.path.insert(0, '.')
   from components.case_network import _similarity, _score_of
   from data_cases import load_past_cases

   cases = load_past_cases()

   # 基準案件: 引数で指定、または直近案件
   industry = '<--industry 引数>'   # 未指定時は None
   base_score = <--score 引数>       # 未指定時は None
   top_n = <--top 引数>              # デフォルト 10

   # 基準案件を構築
   base = {
       'industry_major': industry,
       'score': base_score,
   }

   # 類似度でランク付け
   ranked = []
   for c in cases:
       sim = _similarity(base, c)
       s = _score_of(c)
       ranked.append((sim, c, s))

   ranked.sort(key=lambda x: -x[0])

   print(f'=== 類似事例ランキング ===')
   print(f'基準: 業種={industry or \"指定なし\"}  スコア={base_score or \"指定なし\"}\n')
   print(f'{\"順位\":<4} {\"類似度\":<8} {\"スコア\":<8} {\"グレード\":<6} {\"業種\":<10} {\"最終結果\":<8}')
   print('-' * 60)

   for i, (sim, c, s) in enumerate(ranked[:top_n], 1):
       ind = (c.get('industry_major') or '')[:8]
       status = c.get('final_status', '不明')
       grade = (c.get('result') or {}).get('grade', '?')
       status_icon = '✅' if status == '成約' else ('❌' if status == '失注' else '⚪')
       print(f'{i:<4} {sim:.3f}   {s:6.1f}点  {grade:<6} {ind:<10} {status_icon} {status}')

   # 集計
   top = ranked[:top_n]
   wins = sum(1 for _, c, _ in top if c.get('final_status') == '成約')
   print(f'\n上位{top_n}件: 成約 {wins}件 / 失注 {top_n - wins}件 （成約率 {wins/top_n:.0%}）')
   "
   ```

3. **結果サマリー表示**

```
=== 類似事例ランキング ===
基準: 業種=13 製造業  スコア=72.0

順位 類似度   スコア   グレード 業種       最終結果
------------------------------------------------------------
1    0.800    74.5点  B       13 製造業   ✅ 成約
2    0.750    69.3点  B       13 製造業   ✅ 成約
3    0.620    81.2点  A       13 製造業   ✅ 成約
4    0.600    55.8点  C       13 製造業   ❌ 失注
5    0.550    78.0点  A       13 製造業   ✅ 成約

上位5件: 成約 4件 / 失注 1件 （成約率 80%）
```

4. **インサイト提示**
   - 類似案件の成約率をコメントとして出力
   - スコアが大きく乖離する場合は注意を促す
   - 成約率が低い場合はリスク要因を過去案件から列挙

## 注意事項
- 類似度は「業種 + スコア + 競合他社名」の3要素のみ（簡易版）
- 詳細なネットワーク可視化は Streamlit UI の「案件ネットワーク」ページ
- 過去案件が30件未満の場合、結果の信頼性が低い
- `final_status` が未設定の案件は類似度計算の対象外
