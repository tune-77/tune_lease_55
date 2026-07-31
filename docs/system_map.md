# システム構成マップ

> 目的: システムが複雑化してきたため、「触ってはいけない箇所」と「実験していい箇所」を明確化する。
> 生成日: 2026-08-01 / 対象コミット: master (26c62a0 時点)
>
> **注意**: このドキュメントはコード読解時点のスナップショット。関数名・行番号は変更されうるため、
> 実装前に必ず現物（grepまたはファイル参照）で最新状態を確認すること（CLAUDE.md Freshman Rules 準拠）。

---

## 1. プロンプト注入の全チェーン

紫苑／めぶきちゃんの人格には **2系統の独立したプロンプト構築経路** が存在する。
過去のPR（work_log_20260703_214610, REV-208）で「めぶきベース + 末尾上書き」を混線させて事故った実績があるため、
**この2系統は絶対に混ぜない**こと。

### 1-A. `api/main.py` `/api/chat`（めぶき・紫苑チャットAPI）

`post_chat()`（`api/main.py:8350`）は `req.intent` と `question_category` で3経路に分岐する。

```
POST /api/chat
  │
  ├─ req.intent == "improvement"          → 改善メモ整理経路（api/main.py:8428〜）
  │    Gemini で原文整理 → Obsidian改善ノートへ append_improvement_note
  │    人格プロンプトは使わず、整理専用プロンプト(organize_prompt)のみ
  │
  ├─ question_category == "general"        → 汎用回答経路（api/main.py:8676〜）
  │    RAGをスキップ。neutral_general_system_prompt（人格なし・中立AI）を使用
  │    is_general_response_mode=True の場合は記憶・関係性系ブロックを全てsuppress
  │
  └─ それ以外（通常/RAG経路, api/main.py:9092〜）
       base_prompt_root = _pg_build_ssp(mind, now)  ← api/prompt_generator.build_shion_system_prompt
       これに以下を順に文字列結合（api/main.py:9181 の1行、順序固定）:

       1.  base_prompt_root（人格ベース: mind.json の感情状態から動的生成）
       2.  mode_instruction
       3.  response_mode_context
       4.  basic_lease_question_prompt
       5.  news_focus_context（最新ニュース論点）
       6.  news_brief_context（今日のニュースブリーフ）
       7.  news_actions_context
       8.  obsidian_daily_context
       9.  identity_memory_context（人格記憶）
       10. user_personal_memory_context
       11. experience_loop_context（api/shion_experience_loop）
       12. grey_judgment_context（軍師記録の判断メモリ, main.py:6698 _load_gunshi_judgment_memory）
       13. business_plan_consult_context
       14. continuity_hook_context
       15. delta_awareness_context
       16. memory_to_judgment_context
       17. memory_expression_context
       18. reflection_gate_context
       19. rag_context（DB/Wiki RAG検索結果）
       20. external_research_context
       21. db_context（api/db_query.build_db_context — SQLite統計）
       22. improvement_context（既存改善候補との照合）
       23. judgment_learning_context（judgment_feedback）
       24. memory_recall_context（api/shion_memory_recall）
       25. consciousness_ux_context
       26. shion_specificity_context
       27. shion_light_tone_context
       28. shion_non_domain_context
       29. human_device_resonance_context
       30. judgment_response_shape_context
       31. guidance.prompt_suffix（chat_intent.build_chat_guidance）
       32. （任意）pdca_block（_should_apply_chat_pdca が真の場合のみ）

       → _cap_system_prompt() で文字数上限をトリム（CHAT_SYSTEM_PROMPT_MAX_CHARS, 既定24000字）
       → call_gemini_chat(effective_prompt, history, message)
```

**フォールバック分岐が独立して1本ある**: `question_category == "general"` の経路（8783行目）は
上記と似た結合だが `rag_context / db_context / improvement_context / judgment_learning_context` を含まない軽量版。
2本の結合式は別々にメンテされているため、片方だけ直して同期漏れが起きやすい（実験時要注意）。

### 1-B. `lease_intelligence_dialogue.py` `build_dialogue_context()`（軍師/リース知性体経路）

`api/game_theory/` や軍師API（`api/routers/gunshi.py`, `api/gunshi_gemini.py`）から呼ばれる、
1-Aとは別人格「リース知性体」のプロンプト。`build_dialogue_context()`（716行目）内で f-string 1本に
以下をこの順で埋め込む（`lease_intelligence_dialogue.py:559-673`）:

```
1. recall_section（過去記憶思い出し + 前日内省）
2. 【自己状態】mind_context
3. 【実行環境】environment_block
4. _LEASE_DIALOGUE_EXPERTISE_BLOCK（定数）
5. _LEASE_DIALOGUE_MEMORY_EXPRESSION_BLOCK（定数）
6. 【感情を回答へ反映する規則】emotional_guidance + pad_tone
7. 【関連するObsidian知識】knowledge_block + world_view_section
8. finance_knowledge_block（basic_lease_question_block を先頭にマージ済み）
9. compact_guidance（長文モード時のみ）
10. judgment_asset_block + fear_state_block
11. 【今回の応答モード】mode_guidance
12. tool_block（casual時は簡略版、それ以外はツール一覧フル）
13. 審査ロジック調査の必須規則（固定文）
14. 自分で考え・相談し・学ぶ規則（consult_senior_reasoner 等、固定文）
15. 調査時の3段構造フォーマット規則（固定文）
16. 対話方針（固定文・長い）
17. build_shion_feminine_tone_block()（api/shion_tone）
18. mebuki_block（めぶき稼働状態）
19. caller_guidance（caller=="mebuki" の場合のみ追加）
```

`mode` パラメータ（casual/normal/deep/screening/long）で各ブロックの文字数上限が変わる
（`_clip_prompt_text` で切り詰め）。

### 共通で参照される記憶・感情ステートファイル

両経路とも `data/mind.json`（感情PAD値・world_view）と `data/shion_memory_index.json` 系
（recall/decay/freshness）を読みに行く。**書き込みはスケジューラジョブ経由のみ**（§4参照）。チャット
経路からはread-onlyが原則。

---

## 2. 触ってはいけない箇所

**本番データ・スケジューラ・認証に直結し、変更ミスがそのまま実害（審査結果の誤り・データ破損・
夜間バッチ停止・API無防備化）に繋がる領域。** 変更前に必ず Work Logs (`~/Library/Mobile Documents/
iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Work Logs/`) を確認し、CLAUDE.md の
Plan-First Checkpoint に従うこと。

| 領域 | ファイル | 理由 |
|---|---|---|
| スコアリング確定ロジック | `scoring_core.py`, `constants.py`(`APPROVAL_LINE`/`CONDITIONAL_LINE`/`REVIEW_LINE`) | 審査結果に直結。承認ラインの二重定義事故が過去に実在（CLAUDE.md記載）。参照は必ず `scoring_core.APPROVAL_LINE` import |
| API認証・アクセス制御 | `api/api_key_auth.py`（`ApiKeyAuthMiddleware`）, `api/demo_guard.py`（`DemoReadonlyMiddleware`）, `api/main.py` のミドルウェア登録順（507〜530行目） | `X-API-Key` 検証・デモ環境の削除保護。登録順序を崩すとCORSより内側で認証が効かなくなる |
| スケジューラ本体 | `api/scheduler.py` | 実行時刻・ジョブID変更で夜間バッチが停止/重複実行する。`run_daily_improvement_pipeline.sh` と同様「追記のみ・例外を握りつぶして次のジョブに影響させない」設計を崩さない |
| 記憶減衰・関係性スコアの算出式 | `api/shion_memory_decay.py`（半減期・at_risk閾値）, `api/shion_relationship.py`（無交流ペナルティ・trend判定） | 定数変更が「紫苑らしさ」の長期挙動全体に波及。パラメータ実験は必ずバックアップ・別ブランチで |
| 紫苑システムプロンプトの人格ベース生成 | `api/prompt_generator.py`（`build_shion_system_prompt` / `load_mind`） | §1-Aのベースプロンプト。めぶきベースと軍師系(§1-B)の結合ロジックを混線させると人格が不安定化した実績あり（work_log_20260703） |
| DB本体・案件データ | `data/lease_data.db`, `data/screening_db.sqlite`, `data/users.db`, `data/demo.db` | 個人情報・審査案件を含む。コミット禁止（`.claude/rules/security.md`）。直接UPDATE/DELETEは審査履歴の不可逆破壊につながる |
| 台帳・ログ追記ファイル | `data/ledger.jsonl`, `data/cloudrun_improvement_log.jsonl`, `data/coeff_history.jsonl` 等の `*_log.jsonl` 系 | 追記専用・最後のエントリが有効という規約。過去書き換えや並び替えは履歴破壊。キーは `canonical_key(title)` 形式必須 |
| ゲーム理論の交渉解パラメータの数式部分 | `api/game_theory/negotiation.py`（ナッシュ交渉解の計算式）, `api/game_theory/screening.py`（情報操作検知の判定ロジック本体） | 交渉レンジ・検知スコアが実際の提示条件に影響しうる。パラメータ調整は次章「実験していい箇所」側で吸収する設計を優先 |
| ChromaDB / Obsidian Vault 接続パス | `obsidian_bridge.py` の Vault パス解決部 | iCloudパス優先の実装。パス変更でRAG全壊。uvicornは`.zshrc`を読まないためENVはplistで設定する必要あり |
| Secrets | `.streamlit/secrets.toml`, 環境変数化されたAPIキー類 | Slackトークン・Gemini APIキー等。コミット禁止・ハードコード禁止 |

---

## 3. 実験していい箇所

**失敗してもロールバックが容易で、審査結果や本番データに直接影響しない領域。**
ブランチを切って自由に試してよい。

| 領域 | 場所 | 補足 |
|---|---|---|
| プロンプトブロックの文言・順序（人格の口調・雑談要素） | `api/main.py` の `_build_shion_*_prompt_block` 系関数群（`_build_shion_light_tone_feedback_prompt_block` 等）, `lease_intelligence_dialogue.py` の固定文ブロック（対話方針・tool_block の文言） | ロジックの計算結果ではなく「表現」なので実害が小さい。ただし §1 の結合順序（main.py:9181）を変えるとブロック間の依存が壊れることがあるので順序自体はレビュー対象 |
| フロントエンドUI | `frontend/src/app/` 配下の各ページ、`components/` | strict mode・`apiClient`経由の規約さえ守れば自由度が高い。`eslint --fix`だけ禁止 |
| ゲーム理論の重み・パラメータ（数式ロジックでなく係数） | `api/game_theory/dialogue.py`（繰り返しゲーム分析の表示パラメータ）, `router.py` のデフォルト値（`interest_rate_min/max`, `term_min/max` 等リクエストのデフォルト） | エンドポイント自体の入出力契約を変えなければ実験可能 |
| 紫苑の自己提案・PDCAループの生成ロジック | `api/feedback_pattern_loop.py`, `api/usage_loop_engineering.py` | 生成された提案は `cloudrun_improvement_log.jsonl` に積まれるだけで即実行されない（`dispatch_queue.jsonl` 経由の日次パイプライン設計、CLAUDE.md記憶: 紫苑の対応策フロー設計を参照） |
| game-theory エンドポイントのUI表示・可視化 | フロントの `/game-theory` 系ページ | 計算結果の見せ方の実験はバックエンドに影響しない |
| REV改善パイプラインの候補生成・分類ロジック | `improvement_report_data.py` 等の候補抽出・カテゴライズ | 台帳(`ledger.jsonl`)への書き込み形式さえ守れば調整しやすい |

---

## 4. データファイルの役割一覧（`data/` 配下）

`data/` はコミット禁止（`.claude/rules/security.md`）。以下はコード読解で判明した主要ファイルの役割。
全ファイルの網羅ではなく、頻出・重要なものに絞る。未記載ファイルの用途が知りたい場合は
参照元をgrepしてから扱うこと。

### 審査・スコアリング系

| ファイル | 役割 |
|---|---|
| `lease_data.db` | 案件データ本体（SQLite）。審査履歴・スコア・格付け |
| `screening_db.sqlite` | 審査ゲーム理論（情報操作検知）関連DB |
| `demo.db` | デモ環境用データ（本番と分離） |
| `users.db` | ユーザーアカウント情報 |
| `coeff_auto.json` / `coeff_overrides.json` / `coeff_history.jsonl` | スコアリング係数の自動最適化結果・手動上書き・変更履歴（追記） |
| `industry_benchmarks.json` / `industry_capex_lease.json` / `industry_assets_benchmarks.json` | 業種別ベンチマーク（財務・設備投資・物件評価の参照値） |
| `governance_snapshots.json` | ガバナンス（審査統制）スナップショット |
| `canonical_judgment_rules.json` / `canonical_judgment_rules_preview.json` | 正規化された判断ルール本体とプレビュー |
| `judgment_asset_*` 系 | 「判断資産」（審査判断の再利用可能な知見）の育成候補・成長履歴・フィードバック |

### 紫苑（人格・記憶）系

| ファイル | 役割 |
|---|---|
| `mind.json` | 紫苑の感情状態（PAD値）・world_view（外部環境認識） |
| `shion_memory_index.json` | 紫苑の記憶レコード本体（全件） |
| `shion_memory_usage_log.jsonl` | 記憶が参照された記録（減衰バッチの「最近使用」判定に使用） |
| `shion_memory_freshness.jsonl` | 記憶減衰バッチの出力スナップショット（1行=1回分、追記） |
| `shion_memory_health_state.json` | 記憶健全性の集計状態 |
| `shion_relationship_state.json` | User-紫苑間の関係性スコア・trend |
| `shion_experience_events.jsonl` / `shion_experience_state.json` | 経験ループ（対話ごとの経験イベントと集約状態） |
| `shion_reflection_delta.json` / `shion_timeline_delta.json` | 内省・タイムライン差分（前日比の変化検出） |
| `shion_reasoning_consultations.jsonl` | `consult_senior_reasoner` 呼び出し記録（軍師相談ログ） |
| `shion_self_pdca_log.jsonl` / `shion_self_analysis_cache.json` | 紫苑自身のPDCA・自己分析キャッシュ |
| `shion_pending_tasks.json` | 紫苑が保留した対応候補 |
| `shion_practical_knowledge_map.json` | 実務知識マップ |
| `shion_anniversaries.json` | 記念日・継続日数関連 |
| `shion_monitor_report_state.json` / `shion_monitor_report_log.jsonl` | システム監視レポートの状態・ログ |
| `shion_improvement_triage.jsonl` | 紫苑の改善提案トリアージ結果 |

### 改善パイプライン・フィードバック系

| ファイル | 役割 |
|---|---|
| `cloudrun_improvement_log.jsonl` | 改善提案の集約ログ（自己提案・チャット由来問わず、日次パイプラインが読む） |
| `cloudrun_chat_log.jsonl` / `cloudrun_experience_events.jsonl` / `cloudrun_experience_return.db` | Cloud Run上のチャット・経験イベントの記録 |
| `feedback_pattern_proposals.jsonl` | フィードバック傾向ループ（03:00バッチ）の提案生成元 |
| `usage_loop_events.jsonl` / `usage_loop_proposals.jsonl` | 画面利用ループ（03:30バッチ）のイベント・提案 |
| `improvement_quality_log.jsonl` | 改善提案の品質評価ログ |
| `ledger.jsonl` | REV改善台帳。**キーは `canonical_key(title)` 形式必須**（CLI側のREV-ID形式とは別物、混同すると`needs_review`に戻る実績あり） |
| `response_impact_predictions.jsonl` | 回答が与える影響の事前予測記録 |
| `human_response_feedback.jsonl` | 人間からの応答評価フィードバック |

### RAG・ナレッジ系

| ファイル | 役割 |
|---|---|
| `wiki_embedding_cache.json` | Obsidian Wikiの埋め込みキャッシュ（RAG高速化） |
| `rag_search_log.jsonl` / `rag_hit_log.jsonl` / `rag_feedback_log.jsonl` | RAG検索・ヒット・フィードバックのログ |
| `case_memory_usage_log.jsonl` | 過去案件参照ログ |
| `language_judgment_materials.jsonl` | 言語化された判断材料 |
| `news_judgment_signals.jsonl` / `news_judgment_signals_latest.json` | ニュースから抽出した判断シグナル |
| `lease_news_actions_YYYY-MM-DD.json` / `lease_news_actions_latest.json` | 日次ニュースアクション（日付付き履歴 + 最新版） |

### モデル・数理系

| ファイル | 役割 |
|---|---|
| `ml_rf_v4.pkl` / `lgb_main_model.joblib` / `lgb_qual_model.joblib` / `lgbm_contract_model.pkl` | 各種学習済みモデル（RandomForest/LightGBM） |
| `umap_anomaly_model.joblib` / `umap_embeddings.json` | UMAP異常検知（scoring_core.py参照、モジュールレベルキャッシュ必須の対象） |
| `mahalanobis_model.joblib` | マハラノビス距離モデル |
| `quantum_model.joblib` / `quantum_config.json` / `quantum_feedback.jsonl` / `quantum_backtest_history.jsonl` | 量子干渉スコアリングモジュール関連 |
| `spread_predictor_v2.pkl` / `tunnel_optimizer.joblib` | 金利スプレッド予測・最適化モデル |
| `training_meta.json` | モデル学習時のメタ情報 |

### その他

| ファイル | 役割 |
|---|---|
| `agent_thoughts.jsonl` / `agent_hub_log.jsonl` | エージェント間議論・ハブログ |
| `system_guardrails_audit.jsonl` | ガードレール監査ログ |
| `world_feed.jsonl` / `world_view_notified.json` | 外部環境フィード・world_view既読管理 |
| `slack_notifications.json` | Slack通知設定・状態 |
| `dashboard_stats_cache.json` / `department_stats_cache.json` | ダッシュボード集計キャッシュ |

---

## 5. スケジューラジョブの一覧（`api/scheduler.py`）

タイムゾーンは全ジョブ `Asia/Tokyo`。`misfire_grace_time=300`（5分）。FastAPI起動時に `start_scheduler()` が
全ジョブを登録する。

| 時刻 | ジョブID | 処理関数 | 内容 |
|---|---|---|---|
| 02:00 | `crystallization_daily` | `run_crystallization_batch()` | 外れ値/意見割れ案件を抽出 → Geminiでパターン言語化 → Obsidianへ書き出し（知識結晶化） |
| 03:00 | `shion_feedback_loop_daily` | `run_shion_feedback_loop()` | 人間の応答評価+経験イベントの弱シグナルから改善提案を生成 → `cloudrun_improvement_log.jsonl`へ追記。さらに採用済み提案のbefore/after PDCA評価も実行 |
| 03:30 | `shion_usage_loop_daily` | `run_shion_usage_loop()` | 画面訪問ログからUI/UX改善提案を生成 → 改善ログへ追記 |
| 04:00 | `shion_memory_decay_daily` | `run_shion_memory_decay()` | `shion_memory_index.json` 全レコードの freshness_score を再計算（半減期30日の指数減衰）→ `shion_memory_freshness.jsonl` に追記。0.2未満は at_risk |
| 04:05 | `shion_inactivity_decay_daily` | `run_shion_inactivity_decay()` | 3日以上無交流なら関係性スコアを日割りで減点（`shion_relationship_state.json` 更新） |

いずれも改善提案の**投入のみ**で即時実行はしない設計（[[feedback_shion_countermeasure_flow]]の方針と一致）。
ジョブ内で例外が起きても他ジョブに影響しないよう、各関数は try/except で `{"status": "error", ...}` を返して
ログに残すだけに留めている。**新規ジョブ追加・時刻変更をする場合はこのtry/exceptパターンを崩さないこと。**

---

## 6. 未整理・要フォローアップ

- `api/routers/gunshi.py` / `api/gunshi_gemini.py`（軍師AI本体）はこのドキュメントでは概要のみ。
  プロンプト構築の詳細チェーンは別途棚卸しが必要。
- `run_daily_improvement_pipeline.sh` との連携（スケジューラジョブが積んだ改善候補をこのシェルが
  どう拾うか）は未確認。次回棚卸し対象。
- data/ 配下の `judgment_drills`, `wizard_drafts`, `snapshots`, `backups`, `external`, `recipes`,
  `embedding_compare_cache` はディレクトリのみ確認、中身は未調査。
