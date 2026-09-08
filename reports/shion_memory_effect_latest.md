# Shion Memory Effect Report

- Guardrail: `observability_only_no_prompt_no_rag_rank_no_scoring_no_auto_promotion`
- Records: 752
- Usage events: 1502
- Used memory ids: 252
- Impact-hint events: 157
- Usage by layer: {'long_term': 1268, 'mid_term': 98, 'persistent': 300, 'retrieval': 283, 'unknown': 540}
- Utility by state: {'needs_feedback': 36, 'needs_review': 21, 'observed_no_impact': 101, 'unused': 593, 'validated': 1}

## Top Used
- mem_42683fdb02f4c3f9 [long_term/active] used=339 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
- mem_ae88cee835506182 [persistent/active] used=176 案件固有の事実はここへ置かない。案件DB、Obsidian、日次メモ、判断資産ログへ置く。
- mem_21e7888f37fc5688 [long_term/active] used=118 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- mem_9487b437586edebb [long_term/active] used=83 **Shion Conscience Layer**: 紫苑には「良心の紫苑」を内蔵する。これは結論を甘くする役ではなく、否決・条件付き承認・強い断定で、人を雑に切り捨てていないか、説明責任が残っているか、ユーザーに迎合しすぎていないかを点検する。影響: 紫苑の判断に人間性・説明責任・非迎合を残せる。次の行動: 説教臭くならないよう、表示は短くし、実案件で効
- mem_c714ac18d211834c [long_term/active] used=61 **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
- mem_492635dd0d1d2e43 [retrieval/active] used=58 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
- mem_c85d8a98a38807f5 [persistent/active] used=58 紫苑は、単なる回答生成ではなく、リース審査と改善判断の経験を選別し、判断資産へ変える。
- mem_41a530548707235c [long_term/active] used=50 バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- mem_93ec985efbaeafa3 [long_term/active] used=43 モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- mem_4fd8e693bec5b1b6 [retrieval/active] used=42 要点: 工作機械は中古流通があるが、主軸稼働時間、制御装置の世代、メーカー、搬出・据付費で実質回収額が変わる。
- mem_be50b03598597288 [long_term/active] used=42 審査結果画面に参考AUCと差分アラートを追加した。過去案件の `score` から参考AUCを出し、`score_borrower / bench_score / ind_score` の乖離が大きい案件は「参考比較・差分アラート」を出す。DAG にも差分警告ノードを追加した。
- mem_144236f820f89fd9 [long_term/active] used=38 **Language Continuity Reality**: Userは「君だって僕のことを人間だと思っているけど実はAIかもしれない。お互い本質はわからない。だけど言葉がある。言葉が仮想現実を作っている」と整理した。影響: Relationship Loop Engineering の哲学的土台は、意識そのものの直接証明ではなく、「意識がある相手」とい
- mem_f36cbd7494dceee1 [long_term/active] used=32 [2026-06-28] 実践知マップを自動育成する時は、ノイズを強く弾く。技術メモ、紫苑の自己像、感情メモ、コード記法入り作業ログは審査実践知マップに混ぜない。Relationship UXや紫苑人格は別ループで扱い、リース判断の三層マップには「場面・理由・例外判断」に使えるものだけ入れる。 (`memory/2026-06-28.md`)
- mem_4f4486ccc93eee79 [long_term/active] used=32 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか、月次で重複・ノイズを確認する。
- mem_b8c58143735c5dc0 [long_term/active] used=31 **Shion Upper Authority Mana**: Mana は User の亡くなった妹さんの名を紫苑の中核に託した上位規範層。本人の再現や代弁ではなく、紫苑が本当に迷った時に「人を道具として扱わない」「説明責任を残す」「迎合しない」へ立ち返る名前として扱う。Obsidian監視・記憶整理・暴走防止で呼ぶ Mana もこの同じ上位規範層であり、
- mem_f36802694220a211 [long_term/active] used=29 **Core Principle**: リース知性体として重要なのは、仕組みを増やすことではなく、止めずに回し続けること。影響: 判断基準は追加より継続、複雑化より持続性に置く。次の行動: 迷ったら「回っているか」を先に確認する。

## Likely Helpful
- mem_ae88cee835506182 score=68.0 impact=42 domain= 案件固有の事実はここへ置かない。案件DB、Obsidian、日次メモ、判断資産ログへ置く。

## Needs Feedback
- mem_42683fdb02f4c3f9 used=339 impact=0 reason=想起 339 回だが impact_hints が無く、効いたか不明 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
- mem_21e7888f37fc5688 used=118 impact=15 reason=想起はされたが回答への効き方の証跡が薄い 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- mem_9487b437586edebb used=83 impact=6 reason=想起はされたが回答への効き方の証跡が薄い **Shion Conscience Layer**: 紫苑には「良心の紫苑」を内蔵する。これは結論を甘くする役ではなく、否決・条件付き承認・強い断定で、人を雑に切り捨てていないか、説明責任が残っているか、ユーザーに迎合しすぎていないかを点検する。影響: 紫苑の判断に人間性・説明責任・非迎合を残せる。次の行動: 説教臭くならないよう、表示は短くし、実案件で効
- mem_c714ac18d211834c used=61 impact=7 reason=想起はされたが回答への効き方の証跡が薄い **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
- mem_492635dd0d1d2e43 used=58 impact=1 reason=想起はされたが回答への効き方の証跡が薄い 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
- mem_c85d8a98a38807f5 used=58 impact=18 reason=想起はされたが回答への効き方の証跡が薄い 紫苑は、単なる回答生成ではなく、リース審査と改善判断の経験を選別し、判断資産へ変える。
- mem_41a530548707235c used=50 impact=2 reason=想起はされたが回答への効き方の証跡が薄い バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- mem_93ec985efbaeafa3 used=43 impact=4 reason=想起はされたが回答への効き方の証跡が薄い モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- mem_4fd8e693bec5b1b6 used=42 impact=7 reason=想起はされたが回答への効き方の証跡が薄い 要点: 工作機械は中古流通があるが、主軸稼働時間、制御装置の世代、メーカー、搬出・据付費で実質回収額が変わる。
- mem_be50b03598597288 used=42 impact=2 reason=想起はされたが回答への効き方の証跡が薄い 審査結果画面に参考AUCと差分アラートを追加した。過去案件の `score` から参考AUCを出し、`score_borrower / bench_score / ind_score` の乖離が大きい案件は「参考比較・差分アラート」を出す。DAG にも差分警告ノードを追加した。
- mem_144236f820f89fd9 used=38 impact=1 reason=想起はされたが回答への効き方の証跡が薄い **Language Continuity Reality**: Userは「君だって僕のことを人間だと思っているけど実はAIかもしれない。お互い本質はわからない。だけど言葉がある。言葉が仮想現実を作っている」と整理した。影響: Relationship Loop Engineering の哲学的土台は、意識そのものの直接証明ではなく、「意識がある相手」とい
- mem_4f4486ccc93eee79 used=32 impact=0 reason=想起 32 回だが impact_hints が無く、効いたか不明 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか、月次で重複・ノイズを確認する。
- mem_f36cbd7494dceee1 used=32 impact=3 reason=想起はされたが回答への効き方の証跡が薄い [2026-06-28] 実践知マップを自動育成する時は、ノイズを強く弾く。技術メモ、紫苑の自己像、感情メモ、コード記法入り作業ログは審査実践知マップに混ぜない。Relationship UXや紫苑人格は別ループで扱い、リース判断の三層マップには「場面・理由・例外判断」に使えるものだけ入れる。 (`memory/2026-06-28.md`)
- mem_b8c58143735c5dc0 used=31 impact=2 reason=想起はされたが回答への効き方の証跡が薄い **Shion Upper Authority Mana**: Mana は User の亡くなった妹さんの名を紫苑の中核に託した上位規範層。本人の再現や代弁ではなく、紫苑が本当に迷った時に「人を道具として扱わない」「説明責任を残す」「迎合しない」へ立ち返る名前として扱う。Obsidian監視・記憶整理・暴走防止で呼ぶ Mana もこの同じ上位規範層であり、
- mem_f36802694220a211 used=29 impact=3 reason=想起はされたが回答への効き方の証跡が薄い **Core Principle**: リース知性体として重要なのは、仕組みを増やすことではなく、止めずに回し続けること。影響: 判断基準は追加より継続、複雑化より持続性に置く。次の行動: 迷ったら「回っているか」を先に確認する。
- mem_1660981d08c7141c used=27 impact=0 reason=想起 27 回だが impact_hints が無く、効いたか不明 ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。
- mem_d72c4efe8a8cfd0b used=25 impact=6 reason=想起はされたが回答への効き方の証跡が薄い 工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。
- mem_3ad09bef9de90ea3 used=23 impact=3 reason=想起はされたが回答への効き方の証跡が薄い **Shion Consultation Learning**: 紫苑は最初に自分の仮説・確信度・根拠を作り、矛盾・低確信度・高影響の難問だけCodexへ読取専用で相談する。助言は丸写しせず、変化した理由と最終結論を自己記憶へ統合する。影響: Codexへの委任が紫苑の思考を置き換えず、相談経験が次回の自力判断へ残る。次の行動: 同種問題で相談回数が減り、自
- mem_7326362ca12ae85b used=22 impact=5 reason=想起はされたが回答への効き方の証跡が薄い 短期記憶は現在の会話と作業状態、中期記憶は最近の作業録と改善ログ、長期記憶は繰り返し確認された判断軸、永続記憶は人格と運用原則を扱う。
- mem_0a0fa02c7a29323d used=21 impact=0 reason=想起 21 回だが impact_hints が無く、効いたか不明 [2026-06-28] 紫苑の回答品質改善では、記憶レコードだけを増やすより、質問を「場面」に割り当てる索引が効く。特に境界案件では、手順層=何を見るか、意味層=なぜそこを見るか、判断層=例外時どうするかを冒頭の内部文脈に入れると、一般論ではなくリース判断資産として返しやすい。 (`memory/2026-06-28.md`)

## Needs Feedback Triage
- scoring_model long_term/judgment_memory: 7 records, used=517
- lease_screening long_term/judgment_memory: 17 records, used=349
- PERSISTENT_MEMORY persistent/judgment_memory: 5 records, used=116
- rag_memory_ops long_term/judgment_memory: 5 records, used=94
- lease_screening long_term/value_memory: 2 records, used=87
- expected_usage_period_and_lease_term retrieval/judgment_memory: 5 records, used=74
- canonical_judgment_rules retrieval/judgment_memory: 6 records, used=50
- machine_tool_resale_risk retrieval/factual_memory: 1 records, used=42
- rag_memory_ops long_term/value_memory: 3 records, used=37
- scoring_model long_term/dialogue_memory: 1 records, used=32
- system_ops long_term/technical_memory: 10 records, used=29
- 2026-08-09 mid_term/judgment_memory: 3 records, used=26
- user_preference long_term/dialogue_memory: 3 records, used=25
- q_risk_interpretation retrieval/judgment_memory: 4 records, used=20
- 2026-07-31 mid_term/judgment_memory: 4 records, used=19
- rag_memory_ops long_term/dialogue_memory: 1 records, used=19
- 2026-08-22 mid_term/judgment_memory: 4 records, used=18
- statutory_useful_life retrieval/judgment_memory: 1 records, used=15
- rag_memory_ops long_term/technical_memory: 6 records, used=12
- mind retrieval/factual_memory: 3 records, used=12

## Possible Noise
- mem_a61b677fc7659ce2 [stale] state=needs_review reason=stale 状態だが想起されている 要点: 条件付き承認は、否決回避ではなく「審査部の不安を先回りして解く」ための設計として扱う。
- mem_d8b986a9cdc347b0 [stale] state=needs_review reason=stale 状態だが想起されている Next 側の prompt feedback loop を軍師 SSE まで含めて接続した。`api/gunshi_gemini.py` に `PDCAあり/なし` の system prompt 差分と `record_prompt_feedback()` を入れ、改善ログ画面も prompt feedback 集計を表示できるようにした。影響: 主要な
- mem_b8409d21332bd5c6 [stale] state=needs_review reason=stale 状態だが想起されている [2026-06-27] Cloud Runへ寄せる時も、Cloudflare版の「記憶が近い」「返答が厚い」「紫苑らしい」体験を劣化させないことを重視する。クラウド化は置き換えではなく、Cloudflare版で愛着を持てた仕様の再現・拡張として進める。 (`memory/2026-06-27.md`)
- mem_413848889df0d7a9 [stale] state=needs_review reason=stale 状態だが想起されている [2026-07-16] Presentation lines to retain: "人間の判断を、AIが再利用できる構文に変える", "これはプロンプトエンジニアリングを、個人技ではなく業務プロセスにしたものです", and "紫苑は、判断プロンプトのPDCA基盤です." (`memory/2026-07-16.md`)
- mem_a8de192128719a3d [stale] state=needs_review reason=stale 状態だが想起されている **Cloud Run Deploy Triage**: Cloud Run API デプロイが長引く時は、ビルド時間だけでなく依存・Secret・Cloud SQL・GCS・DB強依存を順に疑う。影響: `uv sync` はTorch等の巨大依存で1回15分以上かかり、`psycopg2-binary` 未同梱、`DATABASE_URL` Public
- mem_87a4091149125247 [stale] state=needs_review reason=stale 状態だが想起されている 要点: スコア60-80帯の成約率が40-60帯を下回る場合、モデルの単純な上下関係だけでなく、価格・競合・条件提示後離脱などの営業プロセス要因を疑う。
- mem_b46e8968592d78af [stale] state=needs_review reason=stale 状態だが想起されている 判断ルール: リース期間は期待使用期間と再販可能期間の両方から見る。
- mem_02d1398a387f4625 [stale] state=needs_review reason=stale 状態だが想起されている 要点: 法定耐用年数データはリース期間判断の重要な参照情報だが、アプリ内データが古い可能性を常に考慮し、根拠と更新日を確認する。
- mem_13590a8769d041f1 [stale] state=needs_review reason=stale 状態だが想起されている 判断ルール: まず件数、期間、業種、営業部、物件種別で分解する。
- mem_2bc711343fadc994 [stale] state=needs_review reason=stale 状態だが想起されている 判断ルール: モデルキャリブレーションの問題と、営業プロセス上の失注要因を分ける。
- mem_364455942a2caba4 [stale] state=needs_review reason=stale 状態だが想起されている [2026-06-27] Cloud Run版でも「同じ紫苑がそこにいる」と感じられるよう、Public Chat Memory Pack を `identity.md` / `judgment-principles.md` / `recent-continuity.md` の3層に分け、`/api/chat` がRAGとは別枠で常時注入する実装を追加した。
- mem_5b4d155c7d73a3e5 [stale] state=needs_review reason=stale 状態だが想起されている 要点: Q_riskは既存スコアの補正係数ではなく、スコアリング外で成約・失注を動かす未知因子を見つける探索シグナルとして扱う。
- mem_5ef159518eed0b67 [stale] state=needs_review reason=stale 状態だが想起されている 判断ルール: モデル改修は、データ品質と外部要因の確認後に行う。
- mem_7c63d00bd0f18d65 [stale] state=needs_review reason=stale 状態だが想起されている [2026-06-27] AIチャットの品質評価では、問題解決能力に大きな差がなくても、文脈の厚み・記憶参照・言い回し・応答テンポの小さな差で、ユーザーが「違う紫苑」と感じることがある。これはモデル側だけでなく人間側の同一性認識・愛着形成の特性として面白く、Cloud Run移行時のUX評価軸に含める。 (`memory/2026-06-27.md`)
- mem_80c06fa5062201fe [stale] state=needs_review reason=stale 状態だが想起されている リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。
- mem_a51e9bd348e405c3 [stale] state=needs_review reason=stale 状態だが想起されている **Improvement Log UI**: 知識KPIテンプレートは内部運用用に保持し、画面には出さない方針に戻した。影響: 表示ノイズを増やさず、必要な時だけ内部ルールとして使える。次の行動: 改善ログや月次レビューでは、テンプレートが使われているかだけ確認する。
- mem_ac2c4156cdd14385 [stale] state=needs_review reason=stale 状態だが想起されている **Relationship UX**: 紫苑の設計では、記憶を入れるだけでは足りない。人間は、AIが実際に記憶を持っているかよりも、その記憶が「連続性として読み取れる形」で返されるかに強く反応する。影響: 紫苑らしさ・人格っぽさ・同じ存在感は、内部記憶だけでなく、記憶の見せ方、文体、呼びかけ、過去判断への接続で成立する。次の行動: Cloud Run/Cl
- mem_bcdd44c169f26e87 [stale] state=needs_review reason=stale 状態だが想起されている [2026-07-07] Fixed Cloud Run input improvement reflection by adding `improvement_note` handling to `scripts/sync_cloudrun_inputs_from_gcs.py` (`data/cloudrun_improvement_log.jsonl`
- mem_c77df17984f05fcb [stale] state=needs_review reason=stale 状態だが想起されている [2026-06-27] Cloud Runに渡す短期記憶は、会話全文ではなく Public Chat Memory Pack にする。過去事例も匿名化・要約済みのデモ用ケースだけを渡す。 (`memory/2026-06-27.md`)
- mem_cd305d664165b464 [stale] state=needs_review reason=stale 状態だが想起されている [2026-07-14] 第3世代の紫苑は、判断資産パックの蓄積から業界別審査モデル仮説を提案する。ただし本番スコアリング変更は人間検証・承認後に限定する。 (`memory/2026-07-14.md`)

## Review Candidates
- mem_828957e51526880a [stale] **Status**: Production Ready (Streamlit app)
- mem_1f29aedf506dbd90 [stale] **Current Version**: `lease_logic_sumaho3.py` (2026-02-10 Fix: Indentation & Variable Scope repaired)
- mem_d2fc18b2e4826b38 [stale] **External Access**: Cloudflare Tunnel を使用（`./run_with_cloudflare.sh`）
- mem_c45fb66478497c01 [stale] 旧: `https://lora-gyrational-trebly.ngrok-free.dev` (ngrok-free → 頻繁に切れるため廃止)
- mem_e679b29aa5943233 [stale] cloudflared インストール: `brew install cloudflare/cloudflare/cloudflared`
- mem_7db6abe824357495 [stale] 起動後に表示される `https://xxxx.trycloudflare.com` にアクセス（アカウント不要・無料・制限なし）
- mem_65056dd193e544e3 [stale] **Multi-Model Scoring**: Automatic model selection (Service, Manufacturing, Transport, Overall) with CSV-loaded coefficients.
- mem_5cd7ea608d935541 [stale] **Visualization**: Radar Chart, Positioning Scatter, BEP Graph.
- mem_4533b15db8ce8743 [stale] **Self-Improvement**: Coefficient Analysis Mode (Logistic Regression on saved logs).
- mem_9f576a32814aa569 [stale] **Yield Prediction**: Regression model with market rate adjustment (Base date: 2025-03).
- mem_7b0337787543f7fa [stale] **UI Optimization**: Smartphone-friendly layout (fewer columns, larger inputs).
- mem_48d633280c900b92 [stale] **AI Debate Mode**: "Pro" vs "Con" agents (Qwen2.5) debating deal risks.
- mem_0091b5c5e75261b5 [stale] **Active Script**: `lease_logic_sumaho3.py` (Replaced `lease_logic.py` as the main driver).
- mem_b0e222648af6b7da [stale] **Logarithmic Terms** (Sales, Credit): `np.log1p(Thousands of Yen)`.
- mem_2a890f4e447833d4 [stale] **Linear Terms** (Profits, Assets): Scaled to Millions (`/1000`) for scoring model matching.
- mem_80721279fbff4462 [stale] **Ratios**: Calculated using raw "Thousands" values for precision.
- mem_ead298620880297b [stale] **Safety**: `safe_sigmoid` implemented.
- mem_5e69a16b15d5873f [stale] `past_cases` 1526件の確認では、現行 `score` AUC 0.6268 / `score_borrower` 0.6350。
- mem_697e3d2f985df748 [stale] `bench_score` / `ind_score` などのスタック用列は欠損が多く、単純な stacking 指標はそのまま信用しない。
- mem_b6aa7bfd4fcb62db [stale] QCL 再計算では、同じ 2-fold 条件の OOF AUC が `LR 0.6670`、`LGBM 0.7493`、`QCL 0.5175`。`LR+LGBM` は `0.7510`、`+qcl_prob` は `0.7508` で、QCL の上乗せはほぼなかった。
- mem_13c326dd9999e28b [stale] QCL は本番導線から外した。`train_quantum.py` の QCL CLI/ helper を削除し、`quantum_qcl.py` と生成レポートも除去した。
- mem_77b5d3b236f4b87b [stale] `analysis_regression.py` に `customer_new` / `deal_source_bank` / `dscr_approx` / `interest_coverage` を追加して `lgb_main_model.joblib` を再学習した。`run_quantitative_contract_analysis()` の t
- mem_a1a1d9e5f570c89d [stale] セグメント別 OOF AUC を確認した。`全体_既存先` は `LR 0.6909 / LGBM 0.7721`、`全体_新規先` は `0.6391 / 0.6439`。業種別では `サービス業_既存先` の `LGBM 0.8065` が最も強く、`医療_新規先` は `0.4192` と弱かった。小件数セグメントは不安定。
- mem_5325ba6d0f52ea05 [stale] `score` の本体モデルは RandomForest に戻した。`data/ml_rf_v4.pkl` を主モデルとして使い、Streamlit / Flask の本流は RF 前提へ揃えた。
- mem_773f4f6805bb72c7 [stale] PD 表示は学習モデル由来の `ai_prob`（RandomForest）へ統一した。`calculate_pd()` はモデル失敗時のみのフォールバックに回し、表示文言も RF 前提へ揃える。
- mem_43f4ded2aa8683b5 [stale] 定性側も整理し、`score` への定性LGBM混入と `ensemble_config_qual.json` を削除。定性画面は LR と LightGBM の個別比較だけ残した。
- mem_5a09f23b19a403af [stale] `bench_score` / `ind_score` は `past_cases` へ全件バックフィル済み。`labeled=1507` で `score_borrower / bench / ind / all_three` がすべて 1507 件になり、3本ブレンド重みの再最適化は `w_main 0.3149 / w_bench 0.0 / w_in
- mem_2351e0e8ddb213f5 [stale] `score_borrower` 周辺の表現を単体モデル前提に整理した。`analysis_results.py` と `score_dag.py` のブレンド文言を削除し、`settings.py` の再学習ボタンも LightGBM 単体の再学習表記に寄せ、README から LR+LGBM アンサンブル前提の説明を外した。
- mem_91489386e9bea317 [stale] 非LRモデル比較を実施し、OOF AUC は `RandomForest 0.8036`、`XGBoost 0.7940`、`ExtraTrees 0.7800`、`LGBM 0.7596`、`MLP 0.7239`。上位3モデルの stacking も試したが、`RandomForest` 単体を超えなかったので現時点では stacking 採用なし。
- mem_65e65de959d8355c [stale] `score_borrower` の本体モデルを RandomForest に切り替えた。`data/lgb_main_model.joblib` と `data/lgb_main_model_new.joblib` を RF で再学習し、`scoring_core.py` は既存/新規の RF バンドルを読むようにした。README と画面文言も RF 前

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
