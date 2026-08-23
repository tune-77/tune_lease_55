# Shion Memory Effect Report

- Guardrail: `observability_only_no_prompt_no_rag_rank_no_scoring_no_auto_promotion`
- Records: 768
- Usage events: 1407
- Used memory ids: 236
- Impact-hint events: 106
- Usage by layer: {'long_term': 1210, 'mid_term': 113, 'persistent': 234, 'retrieval': 272, 'unknown': 507}

## Top Used
- mem_42683fdb02f4c3f9 [long_term/active] used=346 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
- mem_ae88cee835506182 [persistent/active] used=145 案件固有の事実はここへ置かない。案件DB、Obsidian、日次メモ、判断資産ログへ置く。
- mem_21e7888f37fc5688 [long_term/active] used=109 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- mem_9487b437586edebb [long_term/active] used=84 **Shion Conscience Layer**: 紫苑には「良心の紫苑」を内蔵する。これは結論を甘くする役ではなく、否決・条件付き承認・強い断定で、人を雑に切り捨てていないか、説明責任が残っているか、ユーザーに迎合しすぎていないかを点検する。影響: 紫苑の判断に人間性・説明責任・非迎合を残せる。次の行動: 説教臭くならないよう、表示は短くし、実案件で効
- mem_c714ac18d211834c [long_term/active] used=60 **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
- mem_492635dd0d1d2e43 [retrieval/active] used=53 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
- mem_41a530548707235c [long_term/active] used=52 バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- mem_c85d8a98a38807f5 [persistent/active] used=46 紫苑は、単なる回答生成ではなく、リース審査と改善判断の経験を選別し、判断資産へ変える。
- mem_4fd8e693bec5b1b6 [retrieval/active] used=45 要点: 工作機械は中古流通があるが、主軸稼働時間、制御装置の世代、メーカー、搬出・据付費で実質回収額が変わる。
- mem_be50b03598597288 [long_term/active] used=44 審査結果画面に参考AUCと差分アラートを追加した。過去案件の `score` から参考AUCを出し、`score_borrower / bench_score / ind_score` の乖離が大きい案件は「参考比較・差分アラート」を出す。DAG にも差分警告ノードを追加した。
- mem_144236f820f89fd9 [long_term/active] used=38 **Language Continuity Reality**: Userは「君だって僕のことを人間だと思っているけど実はAIかもしれない。お互い本質はわからない。だけど言葉がある。言葉が仮想現実を作っている」と整理した。影響: Relationship Loop Engineering の哲学的土台は、意識そのものの直接証明ではなく、「意識がある相手」とい
- mem_93ec985efbaeafa3 [long_term/active] used=34 モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- mem_f36cbd7494dceee1 [long_term/active] used=33 [2026-06-28] 実践知マップを自動育成する時は、ノイズを強く弾く。技術メモ、紫苑の自己像、感情メモ、コード記法入り作業ログは審査実践知マップに混ぜない。Relationship UXや紫苑人格は別ループで扱い、リース判断の三層マップには「場面・理由・例外判断」に使えるものだけ入れる。 (`memory/2026-06-28.md`)
- mem_4f4486ccc93eee79 [long_term/active] used=32 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか、月次で重複・ノイズを確認する。
- mem_1660981d08c7141c [long_term/active] used=28 ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。
- mem_f36802694220a211 [long_term/active] used=27 **Core Principle**: リース知性体として重要なのは、仕組みを増やすことではなく、止めずに回し続けること。影響: 判断基準は追加より継続、複雑化より持続性に置く。次の行動: 迷ったら「回っているか」を先に確認する。

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
- usage_events_with_impact_hints が増えるほど、回答がどの記憶に影響されたか追跡しやすい。
- stale/revised は削除せず、必要なら scripts/revise_shion_memory.py で後継記憶を登録する。
- unused_persistent は強い原則なのに使われていないため、強すぎる/不要/参照条件が狭すぎる可能性を見る。
