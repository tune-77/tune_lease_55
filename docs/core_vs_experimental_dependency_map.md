# コアスコアリング vs 周辺実験モジュール 依存関係マップ

生成日: 2026-07-31
手法: `scoring_core.py` と `api/main.py` を起点に、Python の `import` 文を静的解析（AST）で辿り、到達可能なモジュールを機械的に洗い出した。**コードの移動・変更は一切行っていない。**

## 注意点・このマップの限界

- 静的な `import`/`from import` のみを追跡している。`api/main.py` 内の `importlib.util.spec_from_file_location`（`data_cases.py` / `base_rate_master.py` / `timesfm_engine.py` の動的ロード）は個別に確認済みで、いずれも既存のコア判定に含まれている。ただし他ファイルの `subprocess` 呼び出しや文字列ベースのモジュール参照（例: シェルスクリプトから叩かれる `scripts/*.py`）は本マップでは検出できない。
- Tier 1 / Tier 2 の境界は「ルーター名から見た用途」による分類で、一部（`api.routers.screening_misc`, `api.domestic_mode` など）は実装を読まないと確信が持てないため「要確認」と明記した。**これは推測であり、Vault上の出典はない。**
- `tests/`, `frontend/`, `_archive/`, `data/`, `mobile_app/` は集計対象外。

---

## Tier 0: 純粋スコアリングコア（30 modules）

`scoring_core.py` 単体からの到達範囲。CLAUDE.md の「要注意領域」に最も近い、審査結果に直結する層。

```
analysis_regression, api.db_connection, app_logger, base_rate_master, case_similarity,
charts, coeff_definitions, config, constants, credit_risk_detector, data_cases,
estat_context, evaluators.base, evaluators.quantum_adapter, evaluators.types,
grade_normalizer, mahalanobis_engine, migrate_to_sqlite, model_review_hooks,
quantum_analysis_module, rule_manager, runtime_paths, scoring.feature_engineering_custom,
scoring.industry_hybrid_model, scoring.model, scoring.predict_one, scoring_core,
train_mahalanobis, umap_anomaly_engine, useful_life_lookup
```

※ `quantum_analysis_module` や `mahalanobis_engine` は名前だけ見ると「実験系」に見えるが、実際は `scoring_core.py` が直接importしておりスコアリング本体の一部。命名だけで判断すると誤る好例。

---

## Tier 1: コア審査API（`api/main.py` 直下・審査業務に直結）

Tier 0 に加え、`api/main.py` が配線しているルーターのうち審査・案件管理・データ層に該当するもの。

```
api.routers.analysis, api.routers.analytics, api.routers.asset_finance, api.routers.cases,
api.routers.demo, api.routers.feedback_loop, api.routers.improvement,
api.routers.judgment_assets, api.routers.judgment_drill, api.routers.knowledge,
api.routers.misc_endpoints, api.routers.ocr, api.routers.pipeline_misc,
api.routers.recipes, api.routers.rule_engine, api.routers.settings,
api.routers.system_misc, api.routers.timesfm, api.scoring_full, api.schemas,
api.database, api.db_query, api.api_key_auth, api.demo_guard, api.cloudrun_writeback,
api.llm_json_guard, api.secret_access, api.add_outcomes_table, api.business_plan_check
```

要確認（審査系かペルソナ系か実装未読）: `api.routers.screening_misc`, `api.domestic_mode`

---

## Tier 2: `api/main.py` に配線済みの実験機能（約90 modules）

コアAPIと**同じファイル (`api/main.py`) から直接import**されているため、ファイルを分けない限りコアと密結合したまま。

| カテゴリ | モジュール例 |
|---|---|
| 紫苑（AIペルソナ・記憶系） | `api.shion_agent`, `api.shion_conscience`, `api.shion_debate_adk`, `api.shion_mana`, `api.shion_memory_recall`, `api.shion_memory_decay`, `api.shion_memory_lanes`, `api.shion_memory_rerank`, `api.shion_memory_taxonomy`, `api.shion_memory_vector`, `api.shion_relationship`, `api.shion_self_analysis`, `api.shion_tone`, `api.shion_hypothesis_collision`, `api.shion_information_weighting`, `api.shion_practical_knowledge`, `api.shion_experience_loop`, `api.routers.shion_eval_health`, `api.routers.shion_meta`, `api.routers.shion_tasks` |
| 軍師AI / ゲーム理論 | `api.gunshi_gemini`, `api.routers.gunshi`, `api.game_theory.dialogue`, `api.game_theory.negotiation`, `api.game_theory.screening`, `api.multi_agent_screening` |
| RAG / Obsidian連携（チャット用） | `api.knowledge.domain_glossary`, `api.knowledge.feedback_watcher`, `api.knowledge.indexer`, `api.knowledge.obsidian_loader`, `api.knowledge.policy_loader`, `api.knowledge.query_expansion`, `api.knowledge.vector_store`, `api.routers.vault_hub`, `api.crystallizer.anomaly_extractor`, `api.crystallizer.obsidian_writer`, `api.crystallizer.pattern_synthesizer` |
| 文脈エンリッチ（チャット用） | `api.context.context_bundle`, `api.context.geo_enricher`, `api.context.season_enricher`, `api.context.sentiment_enricher`, `api.context.time_context` |
| 自律ループ群 | `api.fluid_pipeline`, `api.loop_engineering_common`, `api.usage_loop_engineering`, `api.knowledge_gap_loop`, `api.judgment_divergence_loop`, `api.outcome_drift_loop`, `api.feedback_pattern_loop` |
| その他ペルソナ機能 | `api.aurion_core_guard`, `api.emotion_trigger`, `api.chat_memory`, `api.user_personal_memory`, `api.prompt_generator`, `api.scheduler`, `api.routers.screening_emotions`, `api.routers.chronicle` |

---

## Tier 3: `api/main.py` からもスコアリングからも未到達（569 modules）

リポジトリには存在するが、現在の本番エントリーポイント（API・スコアリング）からは静的に一切参照されていない。Streamlit旧UI (`streamlit_app.py`) 自体もこの層に含まれる＝CLAUDE.mdが言う「Next.js + FastAPI」構成への移行後、旧経路がコードとして残存していることを裏付ける。

| カテゴリ | 件数 | 例 |
|---|---|---|
| 自動化スクリプト群 (`scripts.*`) | 173 | `scripts.analyze_scoring_drift`, `scripts.sync_implemented_to_obsidian`, `scripts.weekly_self_management` 等 |
| 量子/物理系実験 (quantum, clifford, aurion, wave) | 12 | `quantum_finance_engine`, `aurion_wave_engine`, `clifford_poc`, `flet_aurion_wave` |
| データ移行・学習パイプライン | 10 | `migrate_asset_price`, `train_quantum`, `train_umap_anomaly`, `fetch_estat_annual` |
| スクリーニング・審査補助（未配線） | 6 | `screening_domain.lease_provider`, `secondary_review`, `credit_limit` |
| Slack/通知 | 4 | `slack_bot`, `slack_remote_control`, `slack_screening` |
| 金融シミュレーション（未配線） | 4 | `montecarlo_pricing`, `sp500_prediction`, `future_simulation` |
| Obsidian/Vault連携（未配線分） | 3 | `vault_watcher`, `rebuild_obsidian_index`, `add_frontmatter_to_vault` |
| Flet/デスクトップUI試作 | 2 | `flet_clifford_3d`, `flet_clifford_app` |
| テスト・整合性チェック（単体スクリプト） | 9 | `check_integrity`, `test_montecarlo` |
| 上記以外（一回限りツール・小説生成・agent実験等） | 346 | `agent_claude`, `agent_crosstalk`, `novelist_agent`, `humor_generator`, `streamlit_app`, `mobile_app.*` 他多数 |

全件は `docs/`ではなくスクラッチパスに生データを残してある（`core_modules.txt` / `peripheral_modules.txt`）。必要であれば正式にリポジトリへ追加する。

---

## まとめ

- **本当に狭いコア**は Tier 0 の30ファイルのみ。ここは`APPROVAL_LINE`二重定義のような事故が起きた実績があり、最優先で保護すべき層。
- **問題は Tier 2**: 紫苑・軍師・自律ループ系の約90モジュールが `api/main.py` から直接importされており、コアAPIと同じプロセス・同じファイルに同居している。切り分けるならまずここ（ファイル分割 or ルーター登録の明示的グループ化）が効果が大きい。
- Tier 3 の569モジュールはAPIからは届いていないため、通常運用のスコアリング精度には影響しないが、リポジトリの見通しの悪さ・オンボーディングコストには直結している。
