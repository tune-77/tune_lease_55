# Introspection Report

- Generated at: `2026-06-19T13:05:30`
- Status: `attention`
- Window: 2026-06-19 / 7 days

## Metrics
- Daily notes found: 3
- Reflection days: 3
- Action days: 2
- Boredom hits: 10
- Promotable items: 2
- Prompt previous diff rate: 6.0%

## Findings
- `warn` 内省が次の行動に変換されていない: 振り返りらしき記述に対して、次に確認・修正する行動が少ない。
- `attention` 退屈・停滞シグナルが出ている: ユーザーまたはログに退屈化の兆候がある。数値集計だけでなく、何を変えるかを明示する必要がある。
- `warn` 再帰的自己改善レポートが欠けている: recursive_self_improvement_latest.json がないため、改善結果が次の候補に戻っているか確認できない。
- `info` ループ健全性に警告がある: loop_engineering status=warn。内省レポートでも同じ警告を拾う。
- `info` 応答変化率が低い: prompt feedback の previous_diff_rate=6.0% 。PDCAが形式化している可能性がある。

## Repeated Terms
- 影響: 21
- 次の行動: 21
- memory: 17
- Obsidian: 13
- AUC: 13
- json: 12
- Vault: 10
- Lease: 8

## Next Actions
- 各内省項目に、確認日または実装対象ファイルを1つ紐づける
- 観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える
- 日次改善パイプライン後に recursive_self_improvement_latest.json の生成を確認する
