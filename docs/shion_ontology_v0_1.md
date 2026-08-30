# 紫苑オントロジー v0.1

作成日: 2026-08-30

状態: `draft / design-only`

適用範囲: tune_lease_55 のリース審査、追加確認、判断資産、結果検証

## 目的

紫苑が扱う案件、物件、判断、確認質問、根拠、判断資産、結果を、保存形式をまたいで同じ意味で参照できるようにする。

この文書は、業務概念と関係の語彙を定める設計上の正本である。SQLite、JSONL、Obsidian、RAG、既存APIの保存先や実行ロジックは置き換えない。

## 境界線

- ライトウェイトな業務オントロジーとして始める。RDF、OWL、グラフDB、外部プラットフォームを必須にしない。
- 概念の定義と実データを分ける。この文書は概念層であり、案件・結果・判断資産の実データは既存の正本に残す。
- オントロジー参照だけで承認・否認やスコアを自動変更しない。
- Raw、Wiki、未承認の研究素材を、判断資産へ自動昇格しない。
- 人間の判断、修正、評価、結果照合を消さず、出典と時系列を保持する。

## コアエンティティ

| ID | エンティティ | 意味 | 主な識別子 | 主な既存表現 |
|---|---|---|---|---|
| `Borrower` | 借手 | リース料の支払義務を負う法人・個人事業者 | `borrower_id`。未整備時は既存顧客コードを優先 | 案件入力、`past_cases`、`screening_records` |
| `LeaseCase` | リース案件 | 申込から審査、条件設定、結果登録までを束ねる単位 | `case_id` | `screening_records.case_id`、各案件JSON |
| `Asset` | リース物件 | リース対象の設備、車両、機械、IT機器など | `asset_id`。未整備時は案件内の安定キー | 案件入力、物件知識ノート |
| `Industry` | 業種 | 借手または案件が属する業種分類 | `industry_code` | 業種コード、業種別知識ノート |
| `Evidence` | 根拠 | 財務資料、見積、営業説明、外部情報、原資料など | `evidence_id` または安定した出典URI | `evidence_paths`、Obsidianノート、添付資料 |
| `RiskSignal` | リスク信号 | 返済原資、物件価値、競合、数値不整合などの観測事項 | `risk_signal_id` | `risk_axis`、Qrisk、異常検知、ニュース信号 |
| `Judgment` | 判断 | ある時点の見立て、判断方向、条件、確信度 | `judgment_id`。無い場合はライフサイクルイベントID | `judgment_lifecycle_events`、審査レビュー |
| `ConfirmationQuestion` | 確認質問 | 不明・曖昧・高影響事項を検証する問い | `question_id` | `shion_followup_sessions.questions_json` |
| `ConfirmationAnswer` | 確認回答 | 確認質問への人間回答と確認状態 | `followup_id + question_id` | `shion_followup_sessions.answers_json` |
| `ApprovalCondition` | 承認条件 | 実行前または実行後に満たすべき条件 | `condition_id`。当面は判断内の局所ID | 追加確認サマリ、稟議コメント |
| `ApplicabilityCondition` | 適用条件 | 判断資産を使う場面・使わない場面を表す条件 | `applicability_condition_id` | `applies_when`、失敗条件、転用条件 |
| `JudgmentAsset` | 判断資産 | 再利用可能で、適用条件・失敗条件・根拠・次アクションを持つ判断の型 | `judgment_asset_id` / `JA-<id>` | `canonical_judgment_rules.json` |
| `Outcome` | 結果 | 成約・失注、延滞、事故、条件履行などの後日事実 | `outcome_id` または `case_id + observed_at` | 案件結果、追加確認の結果登録 |
| `HumanFeedback` | 人間評価 | 判断資産や確認質問が効いたかを示す評価 | `feedback_id` | `helped`、`challenged`、`rejected`、影響ラベル |
| `Agent` | エージェント | 紫苑、審査担当者、補助エージェントなどの実行主体 | `agent_id` | `actor`、実行ログ |
| `Action` | 業務アクション | 確認、条件追加、候補化、昇格、却下、結果登録など | `action_id` またはイベントID | `event_type`、API操作、監査ログ |

## コアリレーション

関係名は機械可読な英語を正規名とし、日本語は画面・文書表示名として使う。

| 関係 | 主語 → 目的語 | 意味 | 多重度の目安 | 必須メタデータ |
|---|---|---|---|---|
| `submittedBy` | `LeaseCase → Borrower` | 案件の借手 | N:1 | `source`, `observed_at` |
| `leases` | `LeaseCase → Asset` | 案件が対象とする物件 | N:N | `source` |
| `supportedBy` | `Judgment → Evidence` | 判断を支える根拠 | N:N | `source`, `confidence` |
| `contradictedBy` | `Judgment/JudgmentAsset → Evidence/Outcome` | 判断を反証または弱める根拠 | N:N | `reason`, `observed_at` |
| `raises` | `LeaseCase/Evidence → RiskSignal` | 案件・根拠からリスク信号が生じる | 1:N | `source`, `confidence` |
| `addresses` | `ConfirmationQuestion → RiskSignal` | 質問が検証するリスク | N:N | `reason` |
| `askedFor` | `ConfirmationQuestion → LeaseCase` | 質問の対象案件 | N:1 | `created_at`, `priority` |
| `answeredBy` | `ConfirmationQuestion → ConfirmationAnswer` | 質問に対する回答 | 1:0..1 | `answered_at`, `status` |
| `changed` | `ConfirmationAnswer → Judgment` | 回答が判断・条件を変えた | N:N | `change_type`, `reason` |
| `addsCondition` | `Judgment/ConfirmationAnswer → ApprovalCondition` | 判断または回答から条件が追加された | 1:N | `reason`, `status` |
| `derivedFrom` | `Judgment/JudgmentAsset/ConfirmationQuestion → JudgmentAsset/Evidence` | どの資産・根拠から派生したか | N:N | `derivation_reason` |
| `appliesTo` | `JudgmentAsset → RiskSignal/Asset/Industry` | 判断資産の適用対象 | N:N | `applies_when` |
| `failsWhen` | `JudgmentAsset → ApplicabilityCondition` | 判断資産を使わない条件 | 1:N | `reason` |
| `validatedBy` | `Judgment/ConfirmationQuestion/JudgmentAsset → Outcome/HumanFeedback` | 後日結果または人間評価で検証された | N:N | `validation_status`, `observed_at` |
| `usedIn` | `JudgmentAsset → LeaseCase` | 判断資産が案件で実際に使われた | N:N | `used_at`, `adaptation_mode` |
| `supersedes` | `JudgmentAsset → JudgmentAsset` | 新しい判断資産が旧資産を改訂する | N:1 | `revision_reason`, `effective_at` |
| `performedBy` | `Action → Agent` | アクションの実行主体 | N:1 | `performed_at` |
| `actsOn` | `Action → LeaseCase/Judgment/JudgmentAsset` | アクションの対象 | N:N | `authorization`, `result` |

## 共通プロパティ

すべての新規エンティティ・関係で可能な限り共通化する。

| プロパティ | 型 | 説明 |
|---|---|---|
| `id` | string | 種別内で一意かつ不変の識別子 |
| `schema_version` | string | このオントロジーとの対応版。初期値は `0.1` |
| `status` | enum | `candidate / active / held / revised / deprecated / rejected` |
| `source` | string | 元テーブル、イベントID、ノートパス、資料URI |
| `created_at` | datetime | 生成日時 |
| `observed_at` | datetime | 事実が観測された日時。生成日時と分ける |
| `actor` | string | 人間、紫苑、処理名などの作成主体 |
| `confidence` | number/null | 0〜1。未評価は0ではなくnull |
| `reason` | string | 関係・状態・変更の理由 |
| `valid_from` | datetime/null | 適用開始日時 |
| `valid_to` | datetime/null | 適用終了日時 |
| `private` | boolean | 機微情報または非公開情報か |

## ID規約 v0.1

既存IDを置き換えず、オントロジー上で参照するときの名前空間だけを付ける。

| 種別 | 表記例 | 方針 |
|---|---|---|
| 案件 | `case:<case_id>` | `screening_records.case_id` を優先 |
| 借手 | `borrower:<customer_code>` | 会社名だけのハッシュを正本にしない |
| 物件 | `asset:<asset_id>` | 物件ID未整備時は `case:<id>/asset:<local-key>` |
| 判断 | `judgment:<event_id>` | `judgment_lifecycle_events.id` を利用可能 |
| 確認質問 | `question:<followup_id>/<question_id>` | セッション内IDの衝突を防ぐ |
| 判断資産 | `ja:<id>` | 画面表示では既存の `JA-<id>` を維持 |
| 根拠 | `evidence:<source-key>` | ノートパス・イベントIDなど安定した出典を使う |
| 結果 | `outcome:<case_id>/<observed_at>` | 同一案件の複数時点結果を保持する |

## 制約・ガードレール

1. `JudgmentAsset` は、適用条件、失敗条件、根拠、次アクションが揃うまで `candidate` とする。
2. `ConfirmationQuestion` は1回の追加確認で最大3件とする。
3. `ConfirmationAnswer` は対応する質問なしに作成しない。
4. `changed` は、判断・承認条件・停止線のどれが変わったかを `change_type` に残す。
5. `validatedBy` は単なる成約・失注だけで成立させず、何が結果に表れたかを記録する。
6. `supersedes` は旧資産を削除しない。旧資産を `revised` または `deprecated` にする。
7. 研究・ニュース・RAGヒットは `Evidence` または `RiskSignal(candidate)` であり、それだけで `JudgmentAsset(active)` にはしない。
8. オントロジーからの書き込みActionは、人間承認、権限、監査ログが揃うまで実装しない。
9. 個人別の人事評価・監視・懲戒を目的とする関係は定義しない。
10. 本スキーマの導入だけを理由に、既存スコアや承認ラインを変更しない。

## 既存実装へのバインディング

| オントロジー | 現在のバインディング候補 | 状態 |
|---|---|---|
| `LeaseCase` | `screening_records.case_id`、案件保存JSON | 既存 |
| `Judgment` | `judgment_lifecycle_events`、`shion_screening_reviews` | 既存・複数経路 |
| `ConfirmationQuestion/Answer` | `shion_followup_sessions` | 既存 |
| `JudgmentAsset` | `data/canonical_judgment_rules.json` | 既存 |
| `HumanFeedback` | `judgment_asset_usage_feedback.jsonl`、追加確認の影響評価 | 既存・複数経路 |
| `Outcome` | `screening_records.outcome`、結果登録イベント | 既存 |
| `Evidence` | `evidence_paths`、Obsidianノート、原資料 | 部分整備 |
| `Borrower` | 顧客コード、会社名、案件入力 | 共通IDが未整備 |
| `Asset` | 案件入力、物件知識ノート | 共通IDが未整備 |
| `RiskSignal` | `risk_axis`、各種リスク出力 | 語彙統一が必要 |

## 代表クエリ

v0.1の有用性は、次の問いに既存データを辿って答えられるかで評価する。

1. この案件で使われた判断資産は何で、どの根拠から派生し、どの確認質問を生んだか。
2. 過去に同じリスク信号へ使われ、実際に判断変更または見落とし防止へ効いた質問は何か。
3. 結果によって反証された判断資産は何で、現在どの資産に改訂されているか。
4. この物件・業種・返済原資条件に適用できる判断資産と、使ってはいけない条件は何か。
5. ある判断資産を変更した場合、影響を受ける質問、案件レビュー、コメント生成はどれか。

## v0.1の完了条件

- コアエンティティと関係について、人間が同じ意味で読める。
- 既存の保存先を置き換えずにバインディング候補が示されている。
- 判断資産の出典、派生、使用、評価、結果検証、改訂を一続きで表現できる。
- RAGの類似検索と、業務関係を辿る検索の違いが明確である。
- 本番機能へ未接続で、スコア・プロンプト・DBスキーマへ影響しない。

## 非目標

- 全社共通オントロジーの完成
- RDF/OWL準拠やグラフDBの導入
- 既存データの一括移行
- NL2Ontology機能の即時実装
- 判断資産の自動昇格
- AIエージェントへの自律的な書き込み権限付与

## 次版へ進む条件

次の3条件が揃った場合だけv0.2を検討する。

1. 代表クエリのうち少なくとも1件で、従来RAGより有用な関係探索が確認できる。
2. `Borrower` と `Asset` の共通ID方針を、既存案件を壊さずに定められる。
3. 実案件で使われた判断資産について、`usedIn → validatedBy` の追跡例が1件以上できる。

## 関連文書・実装

- `reports/shion_ontology_id_audit_20260830.md`
- `docs/raw_wiki_schema_vault_policy.md`
- `docs/shion_memory_architecture.md`
- `api/shion_judgment_os.py`
- `api/screening_followup.py`
- `api/db_connection.py` の `judgment_lifecycle_events`
- `data/canonical_judgment_rules.json`
- `scripts/build_judgment_asset_graph.py`
- `scripts/build_obsidian_retrieval_graph.py`
