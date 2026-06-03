# Q_risk（定性リスク）実装計画書

**作成日**: 2026-06-03
**元ノート**: `@AI_Insight_Evolved_2026-06-03`（lease-wiki-vault）
**対象**: AURION CORE / めぶき の Q_risk 再定義と実装

---

## 0. エグゼクティブサマリ

Obsidianノート `@AI_Insight_Evolved_2026-06-03` の「めぶき深層考察」は、Q_risk を
**「財務矛盾の検知器」から「成約外因子（説明不能残差）の発見装置」へ役割転換せよ**と指示している。

- 現状の Q_risk（`mobile_app/aurion/q_risk.py`）は財務矛盾8パターンを点数化する**サイドカー参考値**で、スコアリング本体には影響しない。
- ノートの新定義: **Q_risk = `score_external_contract_factor`** ＝「信用スコアでは説明できない成約・失注の歪み」。
- まず**数式化せず、発見タグの蓄積を優先**する（順番を逆にしない）。
- データ基盤は既に存在する: `past_cases`（1,925件 / 成約1,152・失注750）と `screening_outcomes`（118件、事後延滞・デフォルト）。

本計画は**既存の財務矛盾Q_riskを壊さず（v1として残し）、新しい成約外因子Q_risk（v2）をサイドカーとして並走させる**段階導入方針を採る。

---

## 1. 現状の Q_risk 実装

### 1.1 定義・計算方法（現行 v1）

中核は `mobile_app/aurion/q_risk.py::detect_q_risk()`。

- **位置づけ**: RF/LGBM スコアリングに影響しない参考値（サイドカー）。例外を外部に伝播させない設計。入力は全て百万円単位。
- **検知パターン**（BR-201〜208 / `FIN-CONTRADICT-001`〜`008`）:
  | コード | severity | 内容 |
  |---|---|---|
  | 001 | high | 粗利率が異常（-50%未満 or 100%超） |
  | 002 | high | 売上ゼロなのに支払リース料が発生 |
  | 003 | high | 営業利益 > 粗利益（数学的に不可能） |
  | 004 | medium | リース残高 / 年商 > 50% |
  | 005 | medium | 総借入 / 年商 > 100% |
  | 006 | medium | 取得額 / 年商 > 30% |
  | 007 | low | 機械設備残高ありなのに減価償却費ゼロ |
  | 008 | medium | 営業利益＞0 だが純利益が大幅マイナス |
- **スコア**: `min(100, high×35 + medium×12 + low×4)`
- **level**: `score≤19 → ok` / `≤49 → caution` / 以上 `high_risk`

### 1.2 連携箇所（呼び出し・参照）

| ファイル | 役割 |
|---|---|
| `mobile_app/aurion/q_risk.py` | v1 本体（財務矛盾検知） |
| `api/main.py:1154-1245` | スコアリング時に `_detect_q_risk()` を呼び、`q_risk_result` を payload へ格納（P2-002） |
| `api/main.py:4399,4408` | チャットプロンプトに「新定義」の説明文（既に文章としては反映済み） |
| `api/schemas.py:71` | `credit_quantum_strong_warning`（信用リスク群×Q_risk 強警戒フラグ） |
| `mobile_app/api.py:145-151,388-434,563,1154` | モバイル向けチャート payload（`Q_risk健全度 = 100 - score×2`）、フォールバック読込 |
| `mobile_app/advisor_strategy.py:42-75,374-380,473-664,754` | 軍師AIの助言生成。`q_risk≥35`/`≥60` を閾値に助言・スタンスを分岐 |
| `mobile_app/chat_assistant.py:321,342,374` | チャットの説明文（新定義を文章で反映済み） |

> **重要な乖離**: コード（`q_risk.py`）は依然 v1 の財務矛盾式のままだが、チャット文章（`api/main.py:4399`、`chat_assistant.py:321`）は既に「成約外因子の探索シグナル」という新定義を語っている。**説明と実装が不一致**。本計画はこのギャップを埋める。

### 1.3 データ基盤（実在を確認済み）

- `past_cases`（1,925件）: `id, timestamp, industry_sub, score, user_eq, final_status, data(JSON), sales_dept, registration_date, estimate_sent_date, customer_response_date, final_result_date`
  - `final_status`: 成約=1,152 / 失注=750 / 検収完了=19 / 稟議中=2 / スコアリングのみ=2
- `screening_outcomes`（118件）: `case_id, contract_date, actual_status(normal/late_30/late_90/default/completed), delinquent, loss_given_default`
- `screening_records`: `outcome`（indexed）

→ **成約/失注ラベル付きの過去案件が十分量あり、新Q_riskの「説明不能残差」分析は実データで実装可能**。

---

## 2. Obsidianノートが示す拡張方向

ノート「めぶき深層考察」の要旨（L107-119）:

1. **役割転換**: Q_risk は「信用スコアを補正する小さな係数」ではなく、**スコアリングモデルの外側で成約・失注を動かす見えない因子を探す探索軸**。
2. **新定義**: `score_external_contract_factor` ＝「信用スコアでは説明できない成約・失注の歪み」。既存スコアで説明できた分を引いた後に残る**成約差分（説明不能残差）**が主戦場。
3. **3つの帳票を先に作る**:
   - `high_score_lost`（高スコア失注群）: 金利競争・条件提示後離脱・競合先・意思決定速度・過剰条件・物件魅力度不足を見る
   - `low_score_won`（低スコア成約群）: 銀行支援・前受金・保証・補助金・物件換金性・既存取引・営業関係性を見る
   - `same_score_split`（同スコア帯で結果が割れた群）: 営業部・業種細分・物件・期間・金利・提案順序を比較
4. **出力は単一数値でなく発見タグ**: `price_competition_gap`, `bank_support_bridge`, `subsidy_timing_bridge`, `asset_resale_anchor`, `sales_route_strength`, `condition_refusal`, `approval_story_missing`, `customer_urgency_high`
5. **順番厳守**: タグが十分蓄積されてから初めて数式化する。**先に数式で縛ると見落とす**。
6. 関連方針（L52-58）: スコア帯別成約率は非単調（60-80帯 < 40-60帯）。PSI/CSI/較正状態をスコア横に出す。根拠ルート可視化が球体化より先。

---

## 3. 実装ステップ（優先度順）

ノートの「数式は後、発見が先」を最優先方針とし、**フェーズ0〜3**で段階導入する。

### フェーズ0: 命名規約とv1温存（前提整備）

- 既存 `detect_q_risk()` を **`q_risk_financial`（v1）** と内部呼称し、後方互換のため一切変更しない。
- 新規 **`q_risk_external`（v2 = 成約外因子）** を別モジュールとして新設。両者を独立サイドカーとして並走。
- **理由**: ノートは「捨てるべきは名前ではなく狭い定義」と明言。v1を消すのではなく定義域を分ける。

### フェーズ1: 3帳票バッチ分析（発見フェーズ・最優先）★

オフラインバッチで過去案件を3群に分類し、発見タグ候補を抽出する。**ここが新Q_riskの本体**。

- **新規**: `mobile_app/aurion/q_risk_external.py`
  - `build_outcome_cohorts(db_path) -> dict`: `past_cases` をスコア帯×`final_status` で集計
    - `high_score_lost`: `score >= 70` かつ `final_status == "失注"`
    - `low_score_won`: `score < 60` かつ `final_status in ("成約","検収完了")`
    - `same_score_split`: 同一スコア帯（10pt刻み）×同一 `industry_sub` で成約と失注が混在する群
  - `extract_factor_tags(case) -> list[str]`: 各案件の `data`(JSON) と列から発見タグを推定（ルールベース。後述の対応表）
- **新規バッチ**: `scripts/qrisk_cohort_report.py`
  - 上記を実行し `reports/qrisk_cohorts_YYYYMMDD.json` と人間可読の `reports/qrisk_cohorts_YYYYMMDD.md` を出力
  - Obsidian保存（memory方針 [[feedback_obsidian_research]]）も検討
- **発見タグ⇄判定材料 対応表（初版・ルールベース）**:
  | タグ | 判定根拠（data/列から） |
  |---|---|
  | `price_competition_gap` | `competitor` 有 かつ `competitor_rate < recommended_rate` |
  | `bank_support_bridge` | 銀行支援依頼書/`main_bank` 関与フラグ有 |
  | `subsidy_timing_bridge` | 補助金関連フィールド有（`subsidies` 紐付け） |
  | `asset_resale_anchor` | 物件換金性スコア高（`lease_asset_score` 上位） |
  | `sales_route_strength` | `sales_dept` 別成約率が全体平均を有意に上回る |
  | `condition_refusal` | `estimate_sent_date` 有だが `customer_response_date` 後に失注 |
  | `approval_story_missing` | 稟議メモ/`data` の説明欄が空・短い |
  | `customer_urgency_high` | 見積→回答→結果の日数が短い（`final_result_date - estimate_sent_date`） |

  > タグ判定の具体的閾値・フィールドマッピングは `past_cases.data` の実JSON構造を1件ダンプして確定させる（フェーズ1着手時の最初の作業）。

### フェーズ2: ランタイム・サイドカー出力（参考タグ表示）

スコアリング1件ごとに、当該案件が将来どの帳票群に入りそうかを**タグ候補として参考表示**（数値化はまだしない）。

- `q_risk_external.py` に `annotate_case(case, score) -> dict` を追加:
  ```json
  {
    "version": "v2-external",
    "cohort_hint": "high_score_lost_watch",
    "factor_tags": ["price_competition_gap", "condition_refusal"],
    "note": "高スコアだが競合金利・条件提示後離脱の兆候。成約外因子に注意。"
  }
  ```
- `api/main.py:1154-1245` の `q_risk_result` 格納部に並べて `q_risk_external` を追加（**v1は維持**、payloadキーを分離）。
  - `schemas.py` に `q_risk_external` のレスポンス型を追加（任意フィールド・後方互換）。
- `mobile_app/advisor_strategy.py`: 既存の `q_risk≥35/60` 数値ロジックは**そのまま温存**。新タグは `additional_guidance` に**説明文として追記**するのみ（数値判定には使わない）。
- `chat_assistant.py` / `api/main.py:4399` の既存「新定義」文章と**実装が一致**する状態になる（3.5節の乖離解消）。

### フェーズ3: 数式化（タグ蓄積後・将来）

タグが十分蓄積された段階で初めて着手。**本計画では設計のみ記載し、実装は保留**。

- 説明不能残差の定式化案: `q_external = actual_outcome − P(成約 | score, segment)`
  - セグメント別成約率較正（ロジスティック or アイソトニック回帰）を基準線とし、その残差を Q_risk(v2) スコアとする。
- PSI/CSI・較正状態の併記（ノートL58「ドリフト監視をスコア横に」）。
- **着手条件**: フェーズ1帳票が複数月分（例: 60日以上）蓄積し、タグ分布が安定すること。

---

## 4. 影響ファイルと変更概要

| ファイル | 区分 | 変更概要 | フェーズ |
|---|---|---|---|
| `mobile_app/aurion/q_risk.py` | 既存 | **変更なし**（v1温存）。内部呼称のみ `q_risk_financial` | 0 |
| `mobile_app/aurion/q_risk_external.py` | 新規 | コホート分類・発見タグ抽出・案件アノテーション | 1,2 |
| `scripts/qrisk_cohort_report.py` | 新規 | 3帳票バッチ生成（json/md出力） | 1 |
| `reports/qrisk_cohorts_*.{json,md}` | 生成物 | 高スコア失注/低スコア成約/同帯分岐の分析結果 | 1 |
| `api/main.py` (≈1154-1245) | 既存 | `q_risk_external` を payload へ**追加**（v1キーは維持） | 2 |
| `api/schemas.py` (≈71) | 既存 | `q_risk_external` レスポンス型を任意フィールドで追加 | 2 |
| `mobile_app/api.py` (≈388-434) | 既存 | モバイルchart payloadに外因子タグ表示を任意追加 | 2 |
| `mobile_app/advisor_strategy.py` | 既存 | 数値ロジック温存。タグを助言文に追記のみ | 2 |
| `mobile_app/chat_assistant.py` | 既存 | 説明文は既に新定義。実装と整合させる確認のみ | 2 |

> フロントエンド（`frontend/`）は当面**表示追加なし**。フェーズ2でAPIが `q_risk_external` を返し始めた後、別タスクで根拠ルートUIに組み込む（ノートL55「根拠ルートが先、球体化は後」に従い、3D化は最後）。

---

## 5. リスクと注意点

### 5.1 設計・スコープ
- **v1を消さない**: 財務矛盾検知（FIN-CONTRADICT-00x）は単独で有用。新定義はv1を**置換でなく拡張**。両者を別キーで並走させる。
- **数式化を急がない**: ノートの最重要メッセージ。フェーズ3は条件成立まで着手しない。先に数式で縛ると未知因子を見落とす。
- **サイドカー原則の堅持**: v2もRF/LGBMスコア本体に影響させない。例外を外部伝播させない（v1と同じ防御設計）。

### 5.2 データ品質
- `past_cases.data`(JSON) のスキーマが案件により不揃いの可能性。タグ判定は**欠損許容（KeyError時はタグ無し）**で実装。
- `final_status` の表記ゆれ: 「検収」「検収完了」は成約扱い（`data_cases.py:171-173` の既存ルールに合わせる）。
- スコア帯別成約率が**非単調**（ノートL56: 60-80帯 < 40-60帯）。これは新Q_riskが捉えるべき現象そのもの。基準線を単調と仮定しない。
- `screening_outcomes` は118件と少量。事後パフォーマンス（延滞/デフォルト）軸の分析はサンプル不足に留意し、フェーズ1では成約/失注軸（past_cases）を主とする。

### 5.3 セキュリティ・運用（プロジェクト規約）
- `data/` 配下・`*.db`・生成 `reports/*.json` はコミット禁止対象。**バッチ生成物はgit管理外**とする（`.claude/rules/security.md`）。
- 発見タグや帳票に**個人情報・機密財務データを生で含めない**（顧客名等はマスキング／集計値のみ）。AIプロンプトへ混入させない。
- DB読取は読み取り専用接続で行い、SQLインジェクション回避（パラメータバインド）。

### 5.4 既存挙動の非破壊
- `advisor_strategy.py` の `q_risk≥35/60` 閾値分岐は既存の助言品質を支えているため**数値ロジックは不変**。新タグは説明テキストとしてのみ付加。
- `credit_quantum_strong_warning`（信用リスク群×Q_risk）の発火条件は現状維持。

### 5.5 実装担当
- メモリ方針 [[feedback_codex_for_edits]] に従い、本計画に基づく**コード実装はCodex（worktree）で実施**する。本ドキュメントは計画策定のみ。
- 実装前チェック [[feedback_screener_pitfalls]]: ①方向性検証（成約/失注比較が想定通りか）②feature_names重複確認 を着手時に必ず実施。

---

## 6. 推奨着手順序（まとめ）

1. **フェーズ1から着手**（発見が先）: `past_cases.data` の実JSON構造を1件確認 → タグ対応表を確定 → `qrisk_cohort_report.py` で3帳票を生成し、タグ分布を観察。
2. 帳票の妥当性をめぶき／担当者でレビュー（高スコア失注・低スコア成約が実際に何で割れているか）。
3. **フェーズ2**: ランタイムにタグ候補を参考出力し、説明と実装の乖離を解消。
4. タグが60日以上安定蓄積後、**フェーズ3**（数式化）を別計画で検討。

> めぶき所見（ノートL119）の通り、Q_risk は「財務矛盾の検知器」から「成約の正体を探す探索灯」へ。**捨てるのは名前ではなく狭い定義**。
