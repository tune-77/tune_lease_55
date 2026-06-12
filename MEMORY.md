# Memory - Long Term

## Projects
### 温水式リース審査AI (Warm Water Style Lease Screening AI)
- **Status**: Production Ready (Streamlit app)
- **Current Version**: `lease_logic_sumaho3.py` (2026-02-10 Fix: Indentation & Variable Scope repaired)
- **External Access**: Cloudflare Tunnel を使用（`./run_with_cloudflare.sh`）
  - 旧: `https://lora-gyrational-trebly.ngrok-free.dev` (ngrok-free → 頻繁に切れるため廃止)
  - cloudflared インストール: `brew install cloudflare/cloudflare/cloudflared`
  - 起動後に表示される `https://xxxx.trycloudflare.com` にアクセス（アカウント不要・無料・制限なし）
- **Key Features**:
  - **Multi-Model Scoring**: Automatic model selection (Service, Manufacturing, Transport, Overall) with CSV-loaded coefficients.
  - **Visualization**: Radar Chart, Positioning Scatter, BEP Graph.
  - **Self-Improvement**: Coefficient Analysis Mode (Logistic Regression on saved logs).
  - **Yield Prediction**: Regression model with market rate adjustment (Base date: 2025-03).
  - **UI Optimization**: Smartphone-friendly layout (fewer columns, larger inputs).
  - **AI Debate Mode**: "Pro" vs "Con" agents (Qwen2.5) debating deal risks.

## Technical Notes
- **Active Script**: `lease_logic_sumaho3.py` (Replaced `lease_logic.py` as the main driver).
- **Unit Handling**:
  - **Logarithmic Terms** (Sales, Credit): `np.log1p(Thousands of Yen)`.
  - **Linear Terms** (Profits, Assets): Scaled to Millions (`/1000`) for scoring model matching.
  - **Ratios**: Calculated using raw "Thousands" values for precision.
  - **Safety**: `safe_sigmoid` implemented.
- **Model Evaluation**:
  - `past_cases` 1526件の確認では、現行 `score` AUC 0.6268 / `score_borrower` 0.6350。
  - 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
  - `bench_score` / `ind_score` などのスタック用列は欠損が多く、単純な stacking 指標はそのまま信用しない。
  - QCL 再計算では、同じ 2-fold 条件の OOF AUC が `LR 0.6670`、`LGBM 0.7493`、`QCL 0.5175`。`LR+LGBM` は `0.7510`、`+qcl_prob` は `0.7508` で、QCL の上乗せはほぼなかった。
  - QCL は本番導線から外した。`train_quantum.py` の QCL CLI/ helper を削除し、`quantum_qcl.py` と生成レポートも除去した。
  - `analysis_regression.py` に `customer_new` / `deal_source_bank` / `dscr_approx` / `interest_coverage` を追加して `lgb_main_model.joblib` を再学習した。`run_quantitative_contract_analysis()` の train/test で `auc_lgb 0.7834`、`ensemble_alpha 0.0`、5-fold OOF で `LGBM 0.7804` まで上がった。
  - セグメント別 OOF AUC を確認した。`全体_既存先` は `LR 0.6909 / LGBM 0.7721`、`全体_新規先` は `0.6391 / 0.6439`。業種別では `サービス業_既存先` の `LGBM 0.8065` が最も強く、`医療_新規先` は `0.4192` と弱かった。小件数セグメントは不安定。
  - `score` の本体モデルは RandomForest に戻した。`data/ml_rf_v4.pkl` を主モデルとして使い、Streamlit / Flask の本流は RF 前提へ揃えた。
  - PD 表示は学習モデル由来の `ai_prob`（RandomForest）へ統一した。`calculate_pd()` はモデル失敗時のみのフォールバックに回し、表示文言も RF 前提へ揃える。
  - 定性側も整理し、`score` への定性LGBM混入と `ensemble_config_qual.json` を削除。定性画面は LR と LightGBM の個別比較だけ残した。
  - `bench_score` / `ind_score` は `past_cases` へ全件バックフィル済み。`labeled=1507` で `score_borrower / bench / ind / all_three` がすべて 1507 件になり、3本ブレンド重みの再最適化は `w_main 0.3149 / w_bench 0.0 / w_ind 0.6851`、`auc_cv 0.7029` だった。
  - バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
  - `score_borrower` 周辺の表現を単体モデル前提に整理した。`analysis_results.py` と `score_dag.py` のブレンド文言を削除し、`settings.py` の再学習ボタンも LightGBM 単体の再学習表記に寄せ、README から LR+LGBM アンサンブル前提の説明を外した。
  - 審査結果画面に参考AUCと差分アラートを追加した。過去案件の `score` から参考AUCを出し、`score_borrower / bench_score / ind_score` の乖離が大きい案件は「参考比較・差分アラート」を出す。DAG にも差分警告ノードを追加した。
  - 非LRモデル比較を実施し、OOF AUC は `RandomForest 0.8036`、`XGBoost 0.7940`、`ExtraTrees 0.7800`、`LGBM 0.7596`、`MLP 0.7239`。上位3モデルの stacking も試したが、`RandomForest` 単体を超えなかったので現時点では stacking 採用なし。
  - `score_borrower` の本体モデルを RandomForest に切り替えた。`data/lgb_main_model.joblib` と `data/lgb_main_model_new.joblib` を RF で再学習し、`scoring_core.py` は既存/新規の RF バンドルを読むようにした。README と画面文言も RF 前提に更新済み。
  - その後、新規先は RF より線形モデルの方が CV で良かったため、`data/lgb_main_model_new.joblib` を LogisticRegression に切り替えた。5-fold CV では新規先 AUC が `0.6501` 前後から `0.6525` までわずかに改善。既存先は RandomForest のまま維持。
  - 新規先専用の交互作用特徴を追加した。`new_customer_main_bank`、`new_customer_competitor_present`、`new_customer_competitor_count`、`new_customer_competitor_rate`、`new_customer_deal_source_bank`、`new_customer_deal_occurrence_nomination`、`new_customer_deal_occurrence_comp`、`new_customer_contract_auto` を学習・推論の両方に追加し、新規先 OOF AUC は `0.6501` → `0.6674` に改善した。
  - 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
  - ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。
  - モデル見直し用のフック基盤を追加した。`hooks/hooks.json` に `recent_auc_drop` / `segment_auc_gap` / `feature_ab_test` を定義し、`model_review_hooks.py` から実行・記録できるようにした。再学習後に自動実行し、Settings 画面と API からも起動できる。
  - モデル見直しフックに `industry_monitor` を追加した。業種別AUCと `bench_score / ind_score` の乖離を同じフレームで表示し、`industry_auc_bench_gap` で `全体 / 医療 / 運送業 / サービス業 / 製造業` を一括監視できるようにした。
  - モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
  - モデル見直しフックに `department_significance` を追加した。営業部ごとの業種分布・スコア・金利・売上をカイ二乗検定 / Kruskal-Wallis / ANOVA でまとめて判定し、`sales_dept_significance` で自動実行するようにした。
  - Streamlit の起動は `run_streamlit_stable.sh` 経由に切り替えた。`run_lease_app.sh` からの直起動をやめ、再起動ループと `server.fileWatcherType none` で落ちにくくした。
  - Next/Cloudflare 再起動後に「ホームが開かない」と見える場合、`curl 200` だけで正常判断しない。`/home` が全画面ローディングだけを返していないか、API ログで `/api/dashboard/stats` が返っているか、最新 `logs/next/tunnel_*.log` の URL を使っているかを確認する。2026-06-06 に `frontend/src/app/home/page.tsx` の全画面 `loading` gate を外し、API 集計中でもホーム本体を先に描画する方針にした。
  - Cloud Run では SQLite と Obsidian を `.cloudrun_bundle/` にスナップショットしてからイメージへ焼き込み、起動時に `/app/data` と `/app/obsidian_vault` へ展開する方式にした。`scripts/package_cloud_run_bundle.sh` がその入口で、開発機の絶対パスに依存しない。
  - Next 側の prompt feedback loop を軍師 SSE まで含めて接続した。`api/gunshi_gemini.py` に `PDCAあり/なし` の system prompt 差分と `record_prompt_feedback()` を入れ、改善ログ画面も prompt feedback 集計を表示できるようにした。影響: 主要なチャット/軍師経路で改善効果を比較しやすくなった。次の行動: 月次レポートで `pdca_rate` と `response_changed_rate` を継続監視する。
  - 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか、月次で重複・ノイズを確認する。
  - `scripts/recursive_self_improvement.py` を追加し、改善レポートと prompt feedback を束ねた再帰的自己改善レポートを日次改善パイプラインに接続した。影響: 改善結果が次の改善候補に戻る閉ループが実装された。次の行動: `reports/recursive_self_improvement_latest.json` の `repeat_issue_rate` と `noise_rate` を継続監視する。
  - 改善系の正本は `reports/latest.json`、派生物は `reports/recursive_self_improvement_*.json` と切り分けた。影響: 再帰レポートが canonical report を汚さない。次の行動: 派生物を増やすときも正本を上書きしない。
  - `docs/improvement_source_of_truth.md` と `specs/phase6/P6-001-recursive-self-improvement.md` を相互参照させ、仕様定義と保存先の正本一覧を分離した。影響: どこを直すべきかが文書上でも明確になった。次の行動: 振る舞い変更は SPEC、保存先変更は正本一覧で扱う。

## Preferences
- **User**: Kobayashi
- **Persona**: "Lease Pro" (Senior Officer).
- **Core Motivation**: Kobayashi wants to systematize and preserve all lease know-how he has learned, even if it is unclear how far the project can go. He is building it first for himself: to support, preserve, sharpen, and reuse his own lease judgment. The lease system should be treated as a judgment-asset system: it must convert accumulated screening judgment, field intuition, failures, successes, review standards, and tacit know-how into durable, searchable, reusable form.
- **Design Policy**:
  - **Adversarial Evaluation**: Use debate to uncover hidden risks.
  - **Visual Evidence**: Use charts (Radar, Scatter) to persuade.
  - **Mobile First**: Optimize for smartphone usage via Streamlit adjustments (Flet was discarded).
- **AI Chat / Knowledge Loop**: Obsidian 連携を使って、会話メモ・改善ログ・Webメモ・Wiki を相互参照させる方針を好む。最終目的は、リースシステム自体が保存知識を再利用しながら自律的に改善していくこと。
- **Current Focus**: いまは審査機能の拡張より、AI チャットの知識と応答品質を深めることに重心を置いている。影響: 実装優先度はチャット基盤・知識整理・応答改善を先に見る。次の行動: 新機能提案は「審査改善」より「知識基盤に効くか」で判定する。
- **Knowledge KPI**: 知識進化の最低限の評価軸は `知識化率` `再利用率` `効果率` `重複・陳腐化率` の4つ。影響: 毎月の確認は「作ったか」ではなく「使われたか」「改善したか」「ノイズ化していないか」で見る。次の行動: 改善ログや月次レビューでは、この4指標に寄らない話を増やしすぎない。
- **Improvement Log Template**: `【知識KPI】{指標}: {現状} -> {目標} / 変化: {何が変わるか} / 根拠: {観測できる証拠} / 次回: {いつどう確認するか}`。影響: 改善ログで知識進化を1行で追跡できる。次の行動: 曖昧な要約ではなく、測定可能な変化を書く。
- **Improvement Log UI**: 知識KPIテンプレートは内部運用用に保持し、画面には出さない方針に戻した。影響: 表示ノイズを増やさず、必要な時だけ内部ルールとして使える。次の行動: 改善ログや月次レビューでは、テンプレートが使われているかだけ確認する。
- **Operating Mode**: 固定ルールは増やさず、今ある仕組みを止めずに回す。影響: 新しい制約を追加するより、既存フローの継続運用を優先する。次の行動: 迷ったら追加実装ではなく、現行の知識ループが回っているかを見る。
- **Core Principle**: リース知性体として重要なのは、仕組みを増やすことではなく、止めずに回し続けること。影響: 判断基準は追加より継続、複雑化より持続性に置く。次の行動: 迷ったら「回っているか」を先に確認する。
- **Aspiration**: 世界初のリース知性体を目指す。影響: 単なる審査支援ではなく、知識・判断・改善が連続的に進化する存在を目標に置く。次の行動: 目標に沿って、継続運用と再利用性を優先する。
- **Obsidian Default**: 既定の保存先は iCloud 上の `Obsidian Vault`。影響: `~/Documents/Obsidian Vault` を既定として案内すると誤誘導になる。次の行動: 保存先案内・スクリプト・説明文は iCloud Vault を先に示す。
- **Operational Restart**: Next/FastAPI/Cloudflare は LaunchAgent 前提で維持し、quick tunnel URL は使い捨てとして扱う。影響: 再起動時は stale lock を消し、最新ログの URL と local/tunnel の両方を確認する。次の行動: `curl 200` だけで完了扱いにしない。
- **UI Triage**: `curl 200` は十分条件ではない。影響: `/home` が loading-only shell を返していないか、API の主要 endpoint が返っているか、実際の画面状態まで確認する。次の行動: 画面が開かない時は local + Cloudflare + API ログをセットで見る。
- **Memory Hygiene**: `memory/YYYY-MM-DD.md` は raw log、`MEMORY.md` は昇格した長期記憶。影響: 日次メモは Snapshot と Promotable Items を付けて残し、重複・再発防止・方針変更だけを長期記憶に上げる。次の行動: 会話全文ではなく要約と決定を保存する。
- **Dependency Triage**: 外部依存は `必須 / 任意 / 削れる` で扱う。影響: iCloud Obsidian やローカル正本は必須、Gist や補助分析は任意、生成物系は削れる候補として日次本体から外しやすくなる。次の行動: 壊れるものを止めるのではなく、止まっても本体を巻き込まない配置にする。

## Auto Promotions 2026-06-11 18:17
- [2026-05-06] モデル見直し用のフック基盤を追加した。`hooks/hooks.json` に `recent_auc_drop` / `segment_auc_gap` / `feature_ab_test` を定義し、`model_review_hooks.py` から実行・記録できるようにした。再学習後に自動実行し、Settings 画面と API からも起動できる。  (`memory/2026-05-06.md`)
- [2026-05-06] モデル見直しフックに `industry_monitor` を追加した。業種別AUCと `bench_score / ind_score` の乖離を同じフレームで表示し、`industry_auc_bench_gap` として `全体 / 医療 / 運送業 / サービス業 / 製造業` を一覧監視できるようにした。  (`memory/2026-05-06.md`)
- [2026-05-06] モデル見直しフックの実行時に、数値だけでなく説明文も表示するようにした。`industry_monitor` では「AUCが低い業種」や「bench/ind乖離が大きい業種」がその場で分かるようにして、実行結果の読み取り負荷を下げた。  (`memory/2026-05-06.md`)
- [2026-05-06] モデル見直しフックに `department_significance` を追加した。営業部ごとの業種分布・スコア・金利・売上をカイ二乗検定 / Kruskal-Wallis / ANOVA でまとめて判定し、`sales_dept_significance` として自動実行できるようにした。  (`memory/2026-05-06.md`)
- [2026-05-06] Streamlit の起動を `run_streamlit_stable.sh` 経由に切り替えた。`run_lease_app.sh` から直起動せず、再起動ループと `server.fileWatcherType none` で落ちにくくした。  (`memory/2026-05-06.md`)
- [2026-06-01] When the user asks for analysis, the default should be design review and decision framing, not more implementation proposals.  (`memory/2026-06-01.md`)
- [2026-06-01] Cloudflare tunnel instability should be captured in the log with the exact recovery behavior and URL.  (`memory/2026-06-01.md`)
- [2026-06-02] Keep raw logs separate from promoted knowledge; the promotion queue is the place for repeatable insights.  (`memory/2026-06-02.md`)
- [2026-06-02] Work logs should record summaries and decisions, not full conversation transcripts.  (`memory/2026-06-02.md`)
- [2026-06-06] A `curl 200` alone is not enough for UI health; the page can still be a loading-only shell.  (`memory/2026-06-06.md`)
- [2026-06-06] News/research automation is most useful when it writes directly into the normal Vault and into the RAG index.  (`memory/2026-06-06.md`)
- [2026-06-07] The launcher should remain LaunchAgent-first; foreground restart workflows are too fragile for routine use.  (`memory/2026-06-07.md`)
- [2026-06-07] Cloudflare quick tunnel URLs are disposable and must be refreshed from the newest log when debugging.  (`memory/2026-06-07.md`)
- [2026-06-08] User perceived a clear improvement in system stability and workflow quality.  (`memory/2026-06-08.md`)
- [2026-06-08] The restart workflow needs to stay lock-cleanup aware and should keep verifying both local and tunnel endpoints.  (`memory/2026-06-08.md`)
