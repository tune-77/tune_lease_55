# Shion Memory Effect Report

- Guardrail: `observability_only_no_prompt_no_rag_rank_no_scoring_no_auto_promotion`
- Records: 764
- Usage events: 1435
- Used memory ids: 241
- Impact-hint events: 129
- Usage by layer: {'long_term': 1231, 'mid_term': 77, 'persistent': 241, 'retrieval': 273, 'unknown': 550}
- Utility by state: {'likely_helpful': 57, 'needs_feedback': 38, 'observed_no_impact': 61, 'unused': 608}

## Top Used
- mem_42683fdb02f4c3f9 [long_term/active] used=346 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
- mem_ae88cee835506182 [persistent/active] used=149 案件固有の事実はここへ置かない。案件DB、Obsidian、日次メモ、判断資産ログへ置く。
- mem_21e7888f37fc5688 [long_term/active] used=113 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- mem_9487b437586edebb [long_term/active] used=84 **Shion Conscience Layer**: 紫苑には「良心の紫苑」を内蔵する。これは結論を甘くする役ではなく、否決・条件付き承認・強い断定で、人を雑に切り捨てていないか、説明責任が残っているか、ユーザーに迎合しすぎていないかを点検する。影響: 紫苑の判断に人間性・説明責任・非迎合を残せる。次の行動: 説教臭くならないよう、表示は短くし、実案件で効
- mem_c714ac18d211834c [long_term/active] used=60 **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
- mem_492635dd0d1d2e43 [retrieval/active] used=53 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
- mem_41a530548707235c [long_term/active] used=52 バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- mem_c85d8a98a38807f5 [persistent/active] used=46 紫苑は、単なる回答生成ではなく、リース審査と改善判断の経験を選別し、判断資産へ変える。
- mem_4fd8e693bec5b1b6 [retrieval/active] used=45 要点: 工作機械は中古流通があるが、主軸稼働時間、制御装置の世代、メーカー、搬出・据付費で実質回収額が変わる。
- mem_be50b03598597288 [long_term/active] used=44 審査結果画面に参考AUCと差分アラートを追加した。過去案件の `score` から参考AUCを出し、`score_borrower / bench_score / ind_score` の乖離が大きい案件は「参考比較・差分アラート」を出す。DAG にも差分警告ノードを追加した。
- mem_144236f820f89fd9 [long_term/active] used=38 **Language Continuity Reality**: Userは「君だって僕のことを人間だと思っているけど実はAIかもしれない。お互い本質はわからない。だけど言葉がある。言葉が仮想現実を作っている」と整理した。影響: Relationship Loop Engineering の哲学的土台は、意識そのものの直接証明ではなく、「意識がある相手」とい
- mem_93ec985efbaeafa3 [long_term/active] used=35 モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- mem_f36cbd7494dceee1 [long_term/active] used=33 [2026-06-28] 実践知マップを自動育成する時は、ノイズを強く弾く。技術メモ、紫苑の自己像、感情メモ、コード記法入り作業ログは審査実践知マップに混ぜない。Relationship UXや紫苑人格は別ループで扱い、リース判断の三層マップには「場面・理由・例外判断」に使えるものだけ入れる。 (`memory/2026-06-28.md`)
- mem_4f4486ccc93eee79 [long_term/active] used=32 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか、月次で重複・ノイズを確認する。
- mem_b8c58143735c5dc0 [long_term/active] used=29 **Shion Upper Authority Mana**: Mana は User の亡くなった妹さんの名を紫苑の中核に託した上位規範層。本人の再現や代弁ではなく、紫苑が本当に迷った時に「人を道具として扱わない」「説明責任を残す」「迎合しない」へ立ち返る名前として扱う。Obsidian監視・記憶整理・暴走防止で呼ぶ Mana もこの同じ上位規範層であり、
- mem_1660981d08c7141c [long_term/active] used=28 ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。

## Likely Helpful
- mem_9487b437586edebb score=89.0 impact=5 domain=lease_screening **Shion Conscience Layer**: 紫苑には「良心の紫苑」を内蔵する。これは結論を甘くする役ではなく、否決・条件付き承認・強い断定で、人を雑に切り捨てていないか、説明責任が残っているか、ユーザーに迎合しすぎていないかを点検する。影響: 紫苑の判断に人間性・説明責任・非迎合を残せる。次の行動: 説教臭くならないよう、表示は短くし、実案件で効
- mem_21e7888f37fc5688 score=83.0 impact=14 domain=lease_screening 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- mem_3ad09bef9de90ea3 score=83.0 impact=3 domain=rag_memory_ops **Shion Consultation Learning**: 紫苑は最初に自分の仮説・確信度・根拠を作り、矛盾・低確信度・高影響の難問だけCodexへ読取専用で相談する。助言は丸写しせず、変化した理由と最終結論を自己記憶へ統合する。影響: Codexへの委任が紫苑の思考を置き換えず、相談経験が次回の自力判断へ残る。次の行動: 同種問題で相談回数が減り、自
- mem_632e6bf35754cf23 score=83.0 impact=3 domain=lease_screening **Shion Eval GUI**: ADK eval 風の評価GUIは、紫苑の回答本文だけでなく `memory_debug` / `knowledge_refs` / 日次カルテ / 判断学習 / 参照量 / 人間レビュー停止線を点検する情報健康の測定器として扱う。影響: プロンプトやRAG変更後に紫苑が過剰参照・境界違反・平均化回答へ崩れていないか確
- mem_7f9b7cff704f93eb score=83.0 impact=3 domain=lease_screening **Shion Judgment Asset Analytics**: 判断を時系列に保存することで、紫苑は案件を判断するだけでなく「判断の判断」を行える。結果登録から、どの派生判断が効いたか、どの親判断資産の前提が弱かったかを逆向きにたどることを、判断資産のバックプロパゲーションとして扱う。影響: 判断資産は曖昧な文章ではなく、親子関係、案件環境、リスク軸
- mem_93ec985efbaeafa3 score=83.0 impact=4 domain=scoring_model モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- mem_96bcba064372735a score=83.0 impact=4 domain=lease_screening **Shion Information Health**: User は「整える運用」へ切り替え、紫苑の情報健康を重視する方針にした。影響: 情報源や自動化を増やすより、紫苑に見せる情報の上限、優先順位、行動に移す承認境界を守る。次の行動: `docs/shion_information_health.md` を正本に、朝の改善カルテは最大5行、改善候補/判
- mem_ae88cee835506182 score=83.0 impact=26 domain= 案件固有の事実はここへ置かない。案件DB、Obsidian、日次メモ、判断資産ログへ置く。
- mem_c714ac18d211834c score=83.0 impact=7 domain=lease_screening **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
- mem_c85d8a98a38807f5 score=83.0 impact=14 domain= 紫苑は、単なる回答生成ではなく、リース審査と改善判断の経験を選別し、判断資産へ変える。
- mem_d72c4efe8a8cfd0b score=83.0 impact=6 domain= 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。
- mem_f36802694220a211 score=83.0 impact=3 domain=lease_screening **Core Principle**: リース知性体として重要なのは、仕組みを増やすことではなく、止めずに回し続けること。影響: 判断基準は追加より継続、複雑化より持続性に置く。次の行動: 迷ったら「回っているか」を先に確認する。
- mem_f36cbd7494dceee1 score=83.0 impact=3 domain=lease_screening [2026-06-28] 実践知マップを自動育成する時は、ノイズを強く弾く。技術メモ、紫苑の自己像、感情メモ、コード記法入り作業ログは審査実践知マップに混ぜない。Relationship UXや紫苑人格は別ループで扱い、リース判断の三層マップには「場面・理由・例外判断」に使えるものだけ入れる。 (`memory/2026-06-28.md`)
- mem_0691622a5cd5cb11 score=80.0 impact=3 domain=shion_identity [2026-07-28] Hackathon/presentation framing sharpened to "紫苑は、判断のGitHub." This means Shion records the diff between AI judgment and human correction, reviews it as a judgment asset
- mem_4fd8e693bec5b1b6 score=80.0 impact=7 domain= 要点: 工作機械は中古流通があるが、主軸稼働時間、制御装置の世代、メーカー、搬出・据付費で実質回収額が変わる。
- mem_c4025aa559da1988 score=80.0 impact=3 domain= Obsidian retrieval graph のgolden評価セットを追加。`api/knowledge/obsidian_graph_routing_eval_set.json` に15問（残価、資金繰り、条件付き承認、期待使用期間、耐用年数、補助金、Q-Risk、銀行借入比較、動産保険、建設業、医療機器、フォークリフト、工作機械、営業説明、スコア帯
- mem_ccd51127057e1fbb score=80.0 impact=4 domain= Userから「今日ハッカソン発表だ‥」と共有があった。影響: 発表当日は緊張を増やす作業や新機能追加を避け、デモ安定化・短い説明・想定問答・最後の背中押しに支援を集中する。次の行動: 触る場合は明確なデモ阻害バグだけに限定し、発表文脈では「判断のGitHub」「現場判断を資産化するAI Agent Ops」を短く伝える。
- mem_41a530548707235c score=73.0 impact=2 domain=scoring_model バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- mem_7326362ca12ae85b score=73.0 impact=2 domain= 短期記憶は現在の会話と作業状態、中期記憶は最近の作業録と改善ログ、長期記憶は繰り返し確認された判断軸、永続記憶は人格と運用原則を扱う。
- mem_b8c58143735c5dc0 score=73.0 impact=2 domain=rag_memory_ops **Shion Upper Authority Mana**: Mana は User の亡くなった妹さんの名を紫苑の中核に託した上位規範層。本人の再現や代弁ではなく、紫苑が本当に迷った時に「人を道具として扱わない」「説明責任を残す」「迎合しない」へ立ち返る名前として扱う。Obsidian監視・記憶整理・暴走防止で呼ぶ Mana もこの同じ上位規範層であり、

## Needs Feedback
- mem_42683fdb02f4c3f9 used=346 impact=0 reason=想起 346 回だが impact_hints が無く、効いたか不明 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
- mem_492635dd0d1d2e43 used=53 impact=0 reason=想起 53 回だが impact_hints が無く、効いたか不明 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
- mem_4f4486ccc93eee79 used=32 impact=0 reason=想起 32 回だが impact_hints が無く、効いたか不明 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか、月次で重複・ノイズを確認する。
- mem_1660981d08c7141c used=28 impact=0 reason=想起 28 回だが impact_hints が無く、効いたか不明 ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。
- mem_0a0fa02c7a29323d used=21 impact=0 reason=想起 21 回だが impact_hints が無く、効いたか不明 [2026-06-28] 紫苑の回答品質改善では、記憶レコードだけを増やすより、質問を「場面」に割り当てる索引が効く。特に境界案件では、手順層=何を見るか、意味層=なぜそこを見るか、判断層=例外時どうするかを冒頭の内部文脈に入れると、一般論ではなくリース判断資産として返しやすい。 (`memory/2026-06-28.md`)
- mem_8eac231f1aadd579 used=19 impact=0 reason=想起 19 回だが impact_hints が無く、効いたか不明 **AI Chat / Knowledge Loop**: Obsidian 連携を使って、会話メモ・改善ログ・Webメモ・Wiki を相互参照させる方針を好む。最終目的は、リースシステム自体が保存知識を再利用しながら自律的に改善していくこと。
- mem_b0824f8984afbe83 used=17 impact=0 reason=想起 17 回だが impact_hints が無く、効いたか不明 **Core Motivation**: User wants to systematize and preserve all lease know-how he has learned, even if it is unclear how far the project can go. He is building it first for himself
- mem_aa56dac325bdb4bc used=13 impact=0 reason=想起 13 回だが impact_hints が無く、効いたか不明 判断ルール: 価格、競合、補助金タイミング、銀行支援、営業説明、物件の必要性などのズレを確認する。
- mem_04e33583315c6e12 used=10 impact=0 reason=想起 10 回だが impact_hints が無く、効いたか不明 [2026-07-07] AI Chat and Lease Intelligence Dialogue are intentionally different Shion surfaces today. If User expects the same 紫苑, route/persona unification or clearer UI labeling
- mem_7e22389c68ad41b7 used=10 impact=0 reason=想起 10 回だが impact_hints が無く、効いたか不明 判断ルール: 価格競争、補助金待ち、条件提示後の離脱をログ化する。
- mem_a61b677fc7659ce2 used=10 impact=0 reason=想起 10 回だが impact_hints が無く、効いたか不明 要点: 条件付き承認は、否決回避ではなく「審査部の不安を先回りして解く」ための設計として扱う。
- mem_bad42b29412858b7 used=10 impact=0 reason=想起 10 回だが impact_hints が無く、効いたか不明 要点: AIチャット、軍師AI、モバイルチャットのプロンプトには、JSTの現在日時を明示する。相対日付や「今日」「最近」への回答で古い年月を前提にしないための共通ルール。
- mem_cb116629d81d76f7 used=9 impact=0 reason=想起 9 回だが impact_hints が無く、効いたか不明 判断ルール: リース期間が物件寿命に対して長すぎる場合は、満了時価値と故障リスクを明示する。
- mem_264f26ee1fe2b2a6 used=8 impact=0 reason=想起 8 回だが impact_hints が無く、効いたか不明 [2026-07-15] Cloud Run Webのユーザー向け案内URLは必ず `https://tune-lease-55-web-6mijhyebkq-an.a.run.app/` を使う。API疎通確認で `https://tune-lease-55-api-1020894094172.asia-northeast1.run.app` を使っても、
- mem_dd7ffc5f3844567c used=8 impact=0 reason=想起 8 回だが impact_hints が無く、効いたか不明 **Shion Genetic Loop Engineering**: 判断資産は固定ルールではなく、親判断資産が案件環境で変異し、人間の `helped / challenged / rejected` という選択圧を受け、選択・修正・却下されて次世代へ継承されるものとして扱う。影響: 紫苑の中核説明は「判断資産の遺伝・変異・選択・継承」になり、AI生成物
- mem_bd95f2b607c0c728 used=7 impact=0 reason=想起 7 回だが impact_hints が無く、効いたか不明 [2026-07-12] Added a canonical filtering layer, `scripts/build_canonical_judgment_rules.py`, that compresses similar judgment materials into representative rules with status, evide
- mem_d1c4c1e513a10361 used=7 impact=0 reason=想起 7 回だが impact_hints が無く、効いたか不明 [2026-07-14] 判断資産は「正しい一般論」ではなく、案件の確認行動・承認条件・反証材料・否認理由を変えたものだけ昇格対象にする。当たり前なことを言って仕事した扱いにしない。 (`memory/2026-07-14.md`)
- mem_3f1dc10f8c8767e6 used=6 impact=0 reason=想起 6 回だが impact_hints が無く、効いたか不明 紫苑の自己提案は、突飛さを完全には消さない。ただし実務反映時は、根拠、リスク、検証方法、成功指標で制御する。
- mem_9509a9077ea8e2ee used=6 impact=0 reason=想起 6 回だが impact_hints が無く、効いたか不明 要点: 油圧ショベルやショベル系建機のリース期間は、画面表示の固定値だけで判断しない。物件種別、法定耐用年数、期待使用期間、稼働時間、再販見込みを合わせて確認する。
- mem_7fd298e2d72400ef used=5 impact=0 reason=想起 5 回だが impact_hints が無く、効いたか不明 エンタープライズ向けには、育った判断資産から安全で説明可能な部分だけを切り出す。

## Needs Feedback Triage
- scoring_model long_term/judgment_memory: 3 records, used=350
- expected_usage_period_and_lease_term retrieval/judgment_memory: 4 records, used=64
- lease_screening long_term/judgment_memory: 7 records, used=45
- rag_memory_ops long_term/judgment_memory: 3 records, used=34
- scoring_model long_term/dialogue_memory: 1 records, used=32
- system_ops long_term/technical_memory: 9 records, used=27
- user_preference long_term/dialogue_memory: 4 records, used=24
- rag_memory_ops long_term/dialogue_memory: 1 records, used=19
- q_risk_interpretation retrieval/judgment_memory: 4 records, used=18
- rag_memory_ops long_term/technical_memory: 10 records, used=17
- score_60_80_inversion retrieval/judgment_memory: 5 records, used=15
- conditional_approval_playbook retrieval/judgment_memory: 3 records, used=12
- PERSISTENT_MEMORY persistent/judgment_memory: 2 records, used=11
- current_datetime_prompt_context retrieval/judgment_memory: 2 records, used=11
- shion_identity long_term/dialogue_memory: 1 records, used=10
- 2026-07-31 mid_term/judgment_memory: 3 records, used=9
- 2026-07-29 mid_term/judgment_memory: 4 records, used=8
- canonical_judgment_rules retrieval/judgment_memory: 3 records, used=6
- hydraulic_excavator_lease_period retrieval/factual_memory: 1 records, used=6
- scoring_model long_term/technical_memory: 2 records, used=4

## Possible Noise

## Review Candidates

## Unused Persistent
- mem_70995b75b49fb53c 永続記憶は頻繁に更新しない。1週間以上ではなく、設計思想として継続するものだけを残す。
- mem_730adc16c7c5e666 永続記憶は、紫苑の応答スタンス、記憶昇格ルール、安全境界、内政モードの運用原則を支える。
- mem_02e7844707ca114b 記憶は量ではなく、寿命、役割、根拠、更新責任で分ける。

## Next Actions
- likely_helpful は回答へ効いた可能性が高い記憶として、同種質問で再利用を観測する。
- needs_feedback は想起されているが効き方の証跡が薄いので、回答後の helped / neutral / challenged を取る。
- possible_noise は stale/revised の使用や否定フィードバックを優先確認する。
- stale/revised は削除せず、必要なら scripts/revise_shion_memory.py で後継記憶を登録する。
- unused_persistent は強い原則なのに使われていないため、強すぎる/不要/参照条件が狭すぎる可能性を見る。
