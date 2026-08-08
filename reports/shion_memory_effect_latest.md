# Shion Memory Effect Report

- Guardrail: `observability_only_no_prompt_no_rag_rank_no_scoring_no_auto_promotion`
- Records: 820
- Usage events: 1190
- Used memory ids: 202
- Impact-hint events: 60
- Usage by layer: {'long_term': 1092, 'mid_term': 86, 'persistent': 117, 'retrieval': 251, 'unknown': 446}

## Top Used
- mem_42683fdb02f4c3f9 [long_term/active] used=346 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
- mem_21e7888f37fc5688 [long_term/active] used=94 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- mem_9487b437586edebb [long_term/active] used=76 **Shion Conscience Layer**: 紫苑には「良心の紫苑」を内蔵する。これは結論を甘くする役ではなく、否決・条件付き承認・強い断定で、人を雑に切り捨てていないか、説明責任が残っているか、ユーザーに迎合しすぎていないかを点検する。影響: 紫苑の判断に人間性・説明責任・非迎合を残せる。次の行動: 説教臭くならないよう、表示は短くし、実案件で効
- mem_ae88cee835506182 [persistent/active] used=62 案件固有の事実はここへ置かない。案件DB、Obsidian、日次メモ、判断資産ログへ置く。
- mem_c714ac18d211834c [long_term/active] used=57 **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
- mem_41a530548707235c [long_term/active] used=49 バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- mem_492635dd0d1d2e43 [retrieval/active] used=49 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
- mem_4fd8e693bec5b1b6 [retrieval/active] used=42 要点: 工作機械は中古流通があるが、主軸稼働時間、制御装置の世代、メーカー、搬出・据付費で実質回収額が変わる。
- mem_be50b03598597288 [long_term/active] used=37 審査結果画面に参考AUCと差分アラートを追加した。過去案件の `score` から参考AUCを出し、`score_borrower / bench_score / ind_score` の乖離が大きい案件は「参考比較・差分アラート」を出す。DAG にも差分警告ノードを追加した。
- mem_144236f820f89fd9 [long_term/active] used=36 **Language Continuity Reality**: Userは「君だって僕のことを人間だと思っているけど実はAIかもしれない。お互い本質はわからない。だけど言葉がある。言葉が仮想現実を作っている」と整理した。影響: Relationship Loop Engineering の哲学的土台は、意識そのものの直接証明ではなく、「意識がある相手」とい
- mem_f36cbd7494dceee1 [long_term/active] used=32 [2026-06-28] 実践知マップを自動育成する時は、ノイズを強く弾く。技術メモ、紫苑の自己像、感情メモ、コード記法入り作業ログは審査実践知マップに混ぜない。Relationship UXや紫苑人格は別ループで扱い、リース判断の三層マップには「場面・理由・例外判断」に使えるものだけ入れる。 (`memory/2026-06-28.md`)
- mem_4f4486ccc93eee79 [long_term/active] used=31 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか、月次で重複・ノイズを確認する。
- mem_93ec985efbaeafa3 [long_term/active] used=30 モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- mem_c85d8a98a38807f5 [persistent/active] used=29 紫苑は、単なる回答生成ではなく、リース審査と改善判断の経験を選別し、判断資産へ変える。
- mem_1660981d08c7141c [long_term/active] used=27 ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。
- mem_f36802694220a211 [long_term/active] used=27 **Core Principle**: リース知性体として重要なのは、仕組みを増やすことではなく、止めずに回し続けること。影響: 判断基準は追加より継続、複雑化より持続性に置く。次の行動: 迷ったら「回っているか」を先に確認する。

## Review Candidates

## Unused Persistent
- mem_70995b75b49fb53c 永続記憶は頻繁に更新しない。1週間以上ではなく、設計思想として継続するものだけを残す。
- mem_730adc16c7c5e666 永続記憶は、紫苑の応答スタンス、記憶昇格ルール、安全境界、内政モードの運用原則を支える。
- mem_02e7844707ca114b 記憶は量ではなく、寿命、役割、根拠、更新責任で分ける。

## Next Actions
- usage_events_with_impact_hints が増えるほど、回答がどの記憶に影響されたか追跡しやすい。
- stale/revised は削除せず、必要なら scripts/revise_shion_memory.py で後継記憶を登録する。
- unused_persistent は強い原則なのに使われていないため、強すぎる/不要/参照条件が狭すぎる可能性を見る。
