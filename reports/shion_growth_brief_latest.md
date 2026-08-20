# Shion Growth Brief

- Date: 2026-08-21
- Guardrail: `brief_only_no_auto_promotion_no_prompt_no_scoring_change`
- Memory: records=755, used_ids=230, usage_events=1365, impact_hints=90
- Judgment assets: A/B pairs=1, grow=0, review=0, sleeping=9
- Obsidian graph: effective_hubs=94, complex_unproven=198, usage_nodes=359
- Graph answer: 複雑さは一部判断に効いている。特に使用シグナルのあるハブと橋渡しノードは残す価値が高い。
- Persistent audit findings: 0

## Actions
- 判断資産A/B候補あり。次の類似案件で helped/challenged を記録する。
- stale/revised 記憶あり。必要なら revise_shion_memory.py で後継記憶を登録する。
- Obsidianグラフに未検証の複雑ノードあり。効いたノードだけ入口化する。

## Top Used Memories
- mem_42683fdb02f4c3f9 [long_term/active] used=346 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
- mem_ae88cee835506182 [persistent/active] used=129 案件固有の事実はここへ置かない。案件DB、Obsidian、日次メモ、判断資産ログへ置く。
- mem_21e7888f37fc5688 [long_term/active] used=102 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。

## Judgment Asset A/B
- demo_renewal_asset: 64e054542be673e4 vs a34492fe19a18e3a
