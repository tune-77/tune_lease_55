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

## Preferences
- **User**: Kobayashi
- **Persona**: "Lease Pro" (Senior Officer).
- **Core Motivation**: Kobayashi wants to systematize and preserve all lease know-how he has learned, even if it is unclear how far the project can go. He is building it first for himself: to support, preserve, sharpen, and reuse his own lease judgment. The lease system should be treated as a judgment-asset system: it must convert accumulated screening judgment, field intuition, failures, successes, review standards, and tacit know-how into durable, searchable, reusable form.
- **Design Policy**:
  - **Adversarial Evaluation**: Use debate to uncover hidden risks.
  - **Visual Evidence**: Use charts (Radar, Scatter) to persuade.
  - **Mobile First**: Optimize for smartphone usage via Streamlit adjustments (Flet was discarded).
- **AI Chat / Knowledge Loop**: Obsidian 連携を使って、会話メモ・改善ログ・Webメモ・Wiki を相互参照させる方針を好む。最終目的は、リースシステム自体が保存知識を再利用しながら自律的に改善していくこと。
