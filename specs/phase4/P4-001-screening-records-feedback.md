---
spec_id: P4-001
phase: 4
title: 支払い実績テーブル連携 — screening_records フィードバックループ
status: implemented
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-15
updated: 2026-05-15
depends_on: [P3-001, P3-002, P3-003]
superseded_by: ""
---

# P4-001 — 支払い実績テーブル連携 — screening_records フィードバックループ

---

## 1. Goal

審査結果（成約 / 失注 / 延滞 / 正常終了）を `screening_records` テーブルに蓄積し、将来のモデル再学習（P4-003）のための教師データを継続的に生成する仕組みを構築する。蓄積データは既存スコアリングロジックには影響せず、純粋なログ目的のサイドカーとして動作させる。

---

## 2. Scope

### In scope
- `data/lease_data.db` への `screening_records` テーブル追加（DDL・マイグレーション）
- 審査実行時に `screening_records` へレコードを INSERT する `record_screening_result()` 関数の実装（`screening_recorder.py` 新規作成）
- Streamlit 審査画面（`tune_lease_55.py` / `components/`）から審査完了後に `record_screening_result()` を呼び出す連携
- Slack Bot (`slack_screening.py`) からも同様に呼び出す連携
- `screening_records` テーブルのデータ参照用 CLI スクリプト（`scripts/export_screening_records.py`）
- `tests/spec_phase4/test_P4-001.py` のテスト作成（Codex 担当）

### Out of scope
- モデル再学習への直接利用（P4-003 で対応）
- EDINET データとの結合（P4-002 以降）
- Streamlit 上での審査履歴一覧画面（本SPEC では記録のみ、表示は別SPEC）
- 既存スコアリングモジュール（`scoring_core.py`, `total_scorer.py`, `asset_scorer.py`）の変更
- `quantum_analysis_module.py` / `aurion/` 以下の変更

---

## 3. Inputs / Outputs

### Inputs（`record_screening_result()` 引数）

| 項目名 | 型 | 必須/任意 | 説明 | 備考 |
|-------|-----|----------|------|------|
| `case_id` | str | 必須 | 案件ID（審査画面で生成済みのUUID等） | |
| `screened_at` | datetime | 必須 | 審査実施日時（UTC） | |
| `total_score` | float | 必須 | 総合スコア（0〜100） | `total_scorer.py` の出力 |
| `asset_score` | float | 必須 | 物件スコア（0〜100） | `asset_scorer.py` の出力 |
| `tenant_score` | float | 任意 | 借手スコア（0〜100） | なければ `None` |
| `q_risk_score` | float | 任意 | 量子リスクスコア（0〜100） | `aurion/q_risk.py` の出力 |
| `competitor_pressure_score` | float | 任意 | 競合圧力スコア（0〜100） | `aurion/stealth_competitor.py` の出力 |
| `outcome` | str | 任意 | 審査結果 `"contracted"` / `"lost"` / `"delinquent"` / `"completed"` / `None` | 後から UPDATE も可 |
| `input_snapshot` | dict | 任意 | 審査入力値のスナップショット（JSON 文字列として保存） | PII除去済みであること |
| `source` | str | 必須 | 入力元 `"streamlit"` / `"slack"` / `"api"` | |

### Outputs（`record_screening_result()` 戻り値）

| 項目名 | 型 | 説明 |
|-------|-----|------|
| `record_id` | int | 挿入されたレコードの ROWID |
| `success` | bool | 書き込み成功フラグ |
| `error` | str \| None | エラーメッセージ（成功時は `None`） |

---

## 4. Data Model

### screening_records テーブル DDL

```sql
CREATE TABLE IF NOT EXISTS screening_records (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id                   TEXT    NOT NULL,
    screened_at               TEXT    NOT NULL,  -- ISO 8601 UTC
    total_score               REAL    NOT NULL,
    asset_score               REAL    NOT NULL,
    tenant_score              REAL,
    q_risk_score              REAL,
    competitor_pressure_score REAL,
    outcome                   TEXT,              -- "contracted" | "lost" | "delinquent" | "completed" | NULL
    input_snapshot            TEXT,              -- JSON文字列
    source                    TEXT    NOT NULL,  -- "streamlit" | "slack" | "api"
    created_at                TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at                TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_screening_records_case_id ON screening_records(case_id);
CREATE INDEX IF NOT EXISTS idx_screening_records_screened_at ON screening_records(screened_at);
CREATE INDEX IF NOT EXISTS idx_screening_records_outcome ON screening_records(outcome);
```

### Python TypedDict

```python
from typing import TypedDict, Optional

class ScreeningRecordInputRequired(TypedDict):
    """必須フィールド（これらが欠けると BR-402 でバリデーションエラー）"""
    case_id: str
    screened_at: str      # ISO 8601 UTC
    total_score: float
    asset_score: float
    source: str           # "streamlit" | "slack" | "api"

class ScreeningRecordInput(ScreeningRecordInputRequired, total=False):
    """任意フィールドを追加した完全な入力型"""
    tenant_score: Optional[float]
    q_risk_score: Optional[float]
    competitor_pressure_score: Optional[float]
    outcome: Optional[str]
    input_snapshot: Optional[dict]

class ScreeningRecordResult(TypedDict):
    record_id: int
    success: bool
    error: Optional[str]
```

---

## 5. API / Interface

### 関数シグネチャ

```python
def record_screening_result(
    case_id: str,
    screened_at: str,          # ISO 8601 UTC 文字列
    total_score: float,
    asset_score: float,
    source: str,
    tenant_score: float | None = None,
    q_risk_score: float | None = None,
    competitor_pressure_score: float | None = None,
    outcome: str | None = None,
    input_snapshot: dict | None = None,
    db_path: str = "data/lease_data.db",
) -> ScreeningRecordResult:
    """
    審査結果を screening_records テーブルに記録する。

    既存スコアリングロジックには影響しない。
    例外を外部に伝播させない設計（失敗時は success=False を返す）。
    """
```

### outcome の後付け UPDATE 関数

```python
def update_screening_outcome(
    case_id: str,
    outcome: str,
    db_path: str = "data/lease_data.db",
) -> ScreeningRecordResult:
    """
    case_id に紐づくレコードのうち id が最大（最後に INSERT された）レコードの outcome を更新する。
    同一 case_id に複数レコードが存在する場合（再審査ユースケース）は最大 id のレコードのみが対象。
    """
```

---

## 6. Business Rules

**BR-401**: テーブル自動作成
- 条件：`record_screening_result()` が初めて呼ばれたとき、または `data/lease_data.db` が存在しない場合
- 処理：`screening_records` テーブルとインデックスを `CREATE TABLE IF NOT EXISTS` で作成してから INSERT する
- 根拠：初回実行時の手動マイグレーション不要化

**BR-402**: 必須フィールドのバリデーション
- 条件：`case_id`、`screened_at`、`total_score`、`asset_score`、`source` のいずれかが欠損（`None` または空文字列 `""` を欠損と見なす）
- 処理：INSERT を行わず `success=False, error="missing required field: {field}"` を返す
- 根拠：欠損データが教師データに混入するのを防ぐ

**BR-403**: スコア範囲チェック
- 条件：`total_score` または `asset_score` が 0.0〜100.0 の範囲外
- 処理：`success=False, error="score out of range: {field}={value}"` を返す
- 根拠：異常値が再学習データを汚染するのを防ぐ

**BR-404**: outcome の許容値チェック
- 条件：`outcome` が `"contracted"`, `"lost"`, `"delinquent"`, `"completed"`, `None` 以外の値
- 処理：`success=False, error="invalid outcome: {value}"` を返す
- 根拠：集計クエリの安全性確保

**BR-405**: input_snapshot の PII 除去チェック
- 条件：`input_snapshot` dict のキー（トップレベルおよびネストされた辞書のキーを再帰的にチェック）に以下の PII キーが含まれる場合
  - 対象キー: `"name"`, `"address"`, `"phone"`, `"email"`, `"company_name"`, `"representative"`, `"hojin_name"`
- 処理：該当キーの値を `"[REDACTED]"` に置換してから JSON 文字列化する。ネストされた辞書（例: `{"applicant": {"name": "山田"}}`）に対しても再帰的に適用する
- 根拠：個人情報保護。審査入力値には氏名・住所・法人名・代表者名が含まれる可能性がある

**BR-406**: 例外非伝播
- 条件：DB 接続失敗・SQLite エラー等の予期しない例外が発生した場合
- 処理：`success=False, error=str(e)` を返す。例外を外部に raise しない
- 根拠：審査フロー本体（スコアリング）を記録失敗で停止させない

---

## 7. UI / UX

本SPECではUIは対象外（記録のみ）。審査履歴の閲覧画面は別SPECで定義する。

---

## 8. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| DB ファイルが存在しない | テーブルを自動作成して INSERT（BR-401） | ディレクトリ `data/` が存在しない場合は `os.makedirs` で作成 |
| 必須フィールド欠損 | `success=False` を返す（BR-402） | 例外なし |
| スコア範囲外 | `success=False` を返す（BR-403） | |
| outcome 不正値 | `success=False` を返す（BR-404） | |
| SQLite UNIQUE 制約違反 | `INSERT OR IGNORE` ではなく通常 INSERT。エラーを `error` フィールドで返す | 同一 case_id の複数レコードは許容（再審査ユースケース） |
| JSON シリアライズ失敗（input_snapshot） | `input_snapshot` を `None` として INSERT 継続 | ログにwarning出力 |

---

## 9. Acceptance Criteria

**AC-1001**: 正常 INSERT が成功する
- Given: 有効な `case_id`, `screened_at`, `total_score=85.0`, `asset_score=70.0`, `source="streamlit"`
- When: `record_screening_result(...)` を呼ぶ
- Then: `success=True` かつ `record_id` が正の整数であること

**AC-1002**: 同一 case_id で複数回 INSERT できる（再審査ユースケース）
- Given: 同じ `case_id` で2回 `record_screening_result(...)` を呼ぶ
- When: 両方の呼び出しが完了する
- Then: `screening_records` に2件のレコードが存在する

**AC-1003**: 必須フィールド（case_id）欠損で失敗する — 空文字列
- Given: `case_id=""` を渡す
- When: `record_screening_result(case_id="", screened_at="2026-05-15T00:00:00Z", total_score=80.0, asset_score=60.0, source="streamlit")`
- Then: `success=False` かつ `error` に `"missing required field"` が含まれる

**AC-1003b**: 必須フィールド（case_id）欠損で失敗する — None
- Given: `case_id=None` を渡す（動的型付けの呼び出し元からの防衛）
- When: `record_screening_result(case_id=None, screened_at="2026-05-15T00:00:00Z", total_score=80.0, asset_score=60.0, source="streamlit")`
- Then: `success=False` かつ `error` に `"missing required field"` が含まれる（例外は発生しない）

**AC-1004**: スコア範囲外（total_score=101.0）で失敗する
- Given: `total_score=101.0`
- When: `record_screening_result(...)` を呼ぶ
- Then: `success=False` かつ DB に INSERT されていない

**AC-1005**: 不正な outcome 値で失敗する
- Given: `outcome="approved"`（許容値外）
- When: `record_screening_result(..., outcome="approved")`
- Then: `success=False` かつ `error` に `"invalid outcome"` が含まれる

**AC-1006**: input_snapshot の PII が REDACTED される
- Given: `input_snapshot={"name": "山田太郎", "total_score": 85.0}`
- When: `record_screening_result(..., input_snapshot={"name": "山田太郎", "total_score": 85.0})`
- Then: DBに保存された JSON 文字列中の `"name"` 値が `"[REDACTED]"` になっている

**AC-1007**: outcome の後付け UPDATE が成功する
- Given: `case_id` で1件 INSERT 済み
- When: `update_screening_outcome(case_id=..., outcome="contracted")` を呼ぶ
- Then: `success=True` かつ DB の outcome が `"contracted"` に更新されている

**AC-1008**: DB が存在しない状態でも正常に INSERT できる（自動作成）
- Given: `db_path` に指定したファイルが存在しない
- When: `record_screening_result(...)` を呼ぶ
- Then: ファイルが作成され `success=True` が返る

**AC-1009**: DB 接続失敗でも例外が外部に伝播しない
- Given: `db_path` に書き込み不可のパスを指定（例: `"/root/no_permission.db"`）
- When: `record_screening_result(...)` を呼ぶ
- Then: 例外が発生せず `success=False` が返る

**AC-1010**: パフォーマンス要件（1件 100ms 以内）
- Given: 任意の有効入力
- When: `record_screening_result()` を50回連続で呼ぶ
- Then: 合計処理時間が 5000ms 以内

---

## 10. Non-Functional Requirements

- **パフォーマンス**: 1件あたり 100ms 以内（DB書き込み込み）
- **後方互換性**: 既存スコアリングの出力値を変更しない
- **例外非伝播**: いかなる内部エラーも外部に raise しない（BR-406）
- **ログ**: INSERT 成功時は `[screening_recorder] case_id={} record_id={}` を出力、失敗時は `[screening_recorder] ERROR: {}` を出力
- **テストカバレッジ**: AC-1001〜AC-1010 全件カバー必須
- **既存モジュール不干渉**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/` 以下を import・変更してはならない

---

## 11. Implementation Notes（Codex向け）

- **ファイル配置**:
  ```
  screening_recorder.py    （プロジェクトルート直下、新規作成）
  scripts/
  └── export_screening_records.py  （CSV出力CLI、新規作成）
  tests/
  └── spec_phase4/
      └── test_P4-001.py   （新規作成）
  ```
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/q_risk.py`, `aurion/stealth_competitor.py`
- **DB 操作**: `sqlite3` 標準ライブラリのみ使用（SQLAlchemy 不要）
- **Connection 管理**: 関数呼び出しごとに `with sqlite3.connect(db_path) as conn:` で接続・切断（コネクションプール不要）
- **screened_at 形式**: `datetime.utcnow().isoformat() + "Z"` で生成、保存時は TEXT 型
- **JSON シリアライズ**: `json.dumps(input_snapshot, ensure_ascii=False)` で UTF-8 保存
- **Streamlit 連携**: `tune_lease_55.py` の審査実行後コールバックで `record_screening_result()` を呼ぶ。`st.session_state` から `case_id` を取得する
- **Slack Bot 連携**: `slack_screening.py` の審査完了ハンドラで呼び出す

---

## 12. Test Plan

### 単体テスト（Codexが作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_1001 | AC-1001 | 正常INSERT → success=True, record_id>0 |
| test_1002 | AC-1002 | 同一case_idで2回INSERT → 2件存在 |
| test_1003 | AC-1003 | case_id="" → success=False, エラーメッセージ確認 |
| test_1004 | AC-1004 | total_score=101.0 → success=False |
| test_1005 | AC-1005 | outcome="approved" → success=False |
| test_1006 | AC-1006 | PII("name")がREDACTEDになる |
| test_1007 | AC-1007 | update_screening_outcome → DBのoutcomeが更新される |
| test_1008 | AC-1008 | DB未存在 → 自動作成してsuccess=True |
| test_1009 | AC-1009 | 書き込み不可パス → 例外なし, success=False |
| test_1010 | AC-1010 | 50回連続 → 5000ms以内 |

### 回帰テスト
- `total_scorer.py` の出力値が本SPEC実装前後で変化しないこと
- Streamlit 審査フロー（`tune_lease_55.py`）が記録失敗時でも正常に動作すること

### 手動確認（実装後）
- [ ] Streamlit 審査画面から審査を実行し、`data/lease_data.db` に `screening_records` レコードが INSERT される
- [ ] Slack Bot から審査を実行し、同様に INSERT される
- [ ] `scripts/export_screening_records.py` を実行し CSV が出力される
