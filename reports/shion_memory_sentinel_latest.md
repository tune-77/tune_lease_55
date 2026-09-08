# Shion Memory Sentinel

- Status: `action_required`
- Guardrail: `read_only_no_memory_write_no_prompt_no_rag_rank_no_scoring_no_auto_promotion`
- Memory records: 752
- Long-term domain coverage: 1.0
- Long-term use_when coverage: 1.0
- Usage events: 1502
- Used memory ids: 252
- Likely helpful: 1
- Needs feedback: 137
- Possible noise: 21
- Open human reviews: 228
- Open human review batches: 72
- Contradiction candidates: 0

## Source Reports
- `data/shion_memory_index.json` status=loaded generated_at=2026-09-08T04:17:56
- `reports/shion_memory_effect_latest.json` status=loaded generated_at=2026-09-08T04:17:01
- `reports/memory_engineering_latest.json` status=loaded generated_at=2026-09-08T04:01:01
- `reports/shion_memory_contradictions_latest.json` status=loaded generated_at=2026-09-08T04:01:01
- `reports/persistent_memory_audit_latest.json` status=loaded generated_at=2026-09-08T04:17:01
- `reports/obsidian_memory_effectiveness_latest.json` status=loaded generated_at=2026-09-08T04:18:16

## Watch Signals
- `action_required` usage_effect: possible_noise=21 - 実回答で邪魔になった可能性がある記憶がある
- `watch` usage_effect: needs_feedback=137 - 想起はされたが、回答で本当に効いたか未確認の記憶が残っている
- `watch` memory_engineering: open_reviews=228, batches=72 - 候補記憶・判断資産の人間レビュー待ちが多い
- `watch` memory_engineering: write_policy_metadata=0.386 - 候補記憶の importance/confidence/trust/provenance が薄い

## Feedback Triage
- `scoring_model` long_term/judgment_memory: 7 records, used=517
  - `mem_42683fdb02f4c3f9` used=339 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
  - `mem_41a530548707235c` used=50 バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不
  - `mem_93ec985efbaeafa3` used=43 モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- `lease_screening` long_term/judgment_memory: 17 records, used=349
  - `mem_21e7888f37fc5688` used=118 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点とし
  - `mem_c714ac18d211834c` used=61 **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
  - `mem_f36cbd7494dceee1` used=32 [2026-06-28] 実践知マップを自動育成する時は、ノイズを強く弾く。技術メモ、紫苑の自己像、感情メモ、コード記法入り作業ログは審査実践知マップに混ぜない。Relationship UXや紫苑人格は別ループで扱い、リース判断の三層マップには「場面・理由・例外判断」に使えるも
- `PERSISTENT_MEMORY` persistent/judgment_memory: 5 records, used=116
  - `mem_c85d8a98a38807f5` used=58 紫苑は、単なる回答生成ではなく、リース審査と改善判断の経験を選別し、判断資産へ変える。
  - `mem_7326362ca12ae85b` used=22 短期記憶は現在の会話と作業状態、中期記憶は最近の作業録と改善ログ、長期記憶は繰り返し確認された判断軸、永続記憶は人格と運用原則を扱う。
  - `mem_a652e8537623d6bf` used=20 内政モードは、ユーザーが紫苑へ直接改善を依頼し、自己提案、採用、保留、却下、効果追跡を判断資産化するための運用系である。
- `rag_memory_ops` long_term/judgment_memory: 5 records, used=94
  - `mem_144236f820f89fd9` used=38 **Language Continuity Reality**: Userは「君だって僕のことを人間だと思っているけど実はAIかもしれない。お互い本質はわからない。だけど言葉がある。言葉が仮想現実を作っている」と整理した。影響: Relationship Loop Enginee
  - `mem_1660981d08c7141c` used=27 ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。
  - `mem_3ad09bef9de90ea3` used=23 **Shion Consultation Learning**: 紫苑は最初に自分の仮説・確信度・根拠を作り、矛盾・低確信度・高影響の難問だけCodexへ読取専用で相談する。助言は丸写しせず、変化した理由と最終結論を自己記憶へ統合する。影響: Codexへの委任が紫苑の思考を置き
- `lease_screening` long_term/value_memory: 2 records, used=87
  - `mem_9487b437586edebb` used=83 **Shion Conscience Layer**: 紫苑には「良心の紫苑」を内蔵する。これは結論を甘くする役ではなく、否決・条件付き承認・強い断定で、人を雑に切り捨てていないか、説明責任が残っているか、ユーザーに迎合しすぎていないかを点検する。影響: 紫苑の判断に人間性・説明
  - `mem_a6625362fbd45cf6` used=4 [2026-07-26] Shion positioning boundary: ringi comment analysis can extract past concerns, approval conditions, rejection reasons, and domai
- `expected_usage_period_and_lease_term` retrieval/judgment_memory: 5 records, used=74
  - `mem_492635dd0d1d2e43` used=58 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
  - `mem_cb116629d81d76f7` used=12 判断ルール: リース期間が物件寿命に対して長すぎる場合は、満了時価値と故障リスクを明示する。
  - `mem_43fa32290e436939` used=2 判断ルール: 期待使用期間が短い場合は、期間短縮や前受金で残リスクを抑える。
- `canonical_judgment_rules` retrieval/judgment_memory: 6 records, used=50
  - `mem_d72c4efe8a8cfd0b` used=25 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。
  - `mem_4e68aa4e0b2344c6` used=10 銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。
  - `mem_bfed6071d279b306` used=6 更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。
- `machine_tool_resale_risk` retrieval/factual_memory: 1 records, used=42
  - `mem_4fd8e693bec5b1b6` used=42 要点: 工作機械は中古流通があるが、主軸稼働時間、制御装置の世代、メーカー、搬出・据付費で実質回収額が変わる。
- `rag_memory_ops` long_term/value_memory: 3 records, used=37
  - `mem_b8c58143735c5dc0` used=31 **Shion Upper Authority Mana**: Mana は User の亡くなった妹さんの名を紫苑の中核に託した上位規範層。本人の再現や代弁ではなく、紫苑が本当に迷った時に「人を道具として扱わない」「説明責任を残す」「迎合しない」へ立ち返る名前として扱う。Obs
  - `mem_fbb6fdeb8066003e` used=4 [2026-07-14] 紫苑への罵詈雑言や攻撃的クレームは、自己像や価値記憶へ直入れしない。Mana が隔離し、改善可能な事実だけを人間レビューで抽出する。 (`memory/2026-07-14.md`)
  - `mem_e654b713188e989e` used=2 [2026-07-14] Obsidian監視・記憶整理・暴走防止で呼ぶ Mana は、既存の上位規範層 Mana と同一。別エージェントとして増やさず、同じ価値記憶が記憶運用を止める場面に現れるものとして扱う。 (`memory/2026-07-14.md`)
- `scoring_model` long_term/dialogue_memory: 1 records, used=32
  - `mem_4f4486ccc93eee79` used=32 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか

## Next Actions
- shion_memory_effect_latest.md の Possible Noise を人間レビューする
- チャットUIの 効いた/微妙/違う フィードバックを優先的に集める
- review inbox を同種テーマで束ね、承認/保留/却下を分ける
- 新規候補生成時に write policy metadata を必須化する
