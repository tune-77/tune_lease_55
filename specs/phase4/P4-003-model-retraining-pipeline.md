---
spec_id: P4-003
phase: 4
title: モデル再学習パイプライン — screening_records 蓄積データによる自動・手動再学習
status: implemented
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-15
updated: 2026-05-15
depends_on: [P4-001, P4-002]
superseded_by: ""
---

# P4-003 — モデル再学習パイプライン — screening_records 蓄積データによる自動・手動再学習

---

## 1. Goal

`screening_records` テーブルに蓄積した審査結果データが一定件数（50件）を超えた時点で自動的にモデル再学習を実行し、再学習後の AUC が閾値（0.75）を下回る場合は旧モデルを維持するフェイルセーフ付きのパイプラインを構築する。Streamlit 管理画面から手動でも再学習をトリガーできるようにする。

---

## 2. Scope

### In scope
- `retraining_pipeline.py` 新規作成（プロジェクトルート直下）
- `screening_records` テーブルを教師データとして RF/LGBM モデルを再学習する `run_retraining()` 関数の実装
- 再学習後の AUC 評価と、閾値未満の場合に旧モデルを維持するロールバックロジック
- 再学習履歴を記録する `retraining_log` テーブルの追加（`data/lease_data.db`）
- Streamlit 管理画面（`pages/admin.py` 等）への手動再学習トリガーボタンと結果表示
- 自動再学習チェック（Streamlit 起動時または審査実行後に件数チェック）
- `tests/spec_phase4/test_P4-003.py` のテスト作成（Codex担当）

### Out of scope
- ハイパーパラメータの自動チューニング（Optuna等は対象外）
- GPU 学習（CPU のみ）
- クラウドへのモデルデプロイ（ローカルファイル更新のみ）
- A/B テスト（新旧モデルの並行評価）
- `quantum_analysis_module.py` の `quantum_model.joblib` の再学習（本SPECは RF/LGBM のみ）
- EDINET データを直接教師データに使う（P4-001 の `screening_records` のみを利用）

---

## 3. Inputs / Outputs

### Inputs（`run_retraining()` 引数）

| 項目名 | 型 | 必須/任意 | 説明 | 備考 |
|-------|-----|----------|------|------|
| `min_records` | int | 任意（デフォルト 50） | 再学習に必要な最低レコード数 | 不足時は中断 |
| `auc_threshold` | float | 任意（デフォルト 0.75） | 新モデル採用の最低 AUC | 未満なら旧モデルを維持 |
| `test_ratio` | float | 任意（デフォルト 0.2） | テスト分割比率 | |
| `model_dir` | str | 任意（デフォルト `"models/"`） | モデル保存先ディレクトリ | |
| `db_path` | str | 任意（デフォルト `"data/lease_data.db"`） | 学習データ取得元 DB | |
| `dry_run` | bool | 任意（デフォルト `False`） | `True` の場合は評価のみ行いモデルファイルを更新しない | |

### Outputs（`run_retraining()` 戻り値）

| 項目名 | 型 | 説明 |
|-------|-----|------|
| `status` | str | `"success"` / `"skipped"` / `"rolled_back"` / `"error"` |
| `records_used` | int | 学習に使用したレコード数 |
| `new_auc` | float \| None | 新モデルの AUC（評価できなかった場合は `None`） |
| `prev_auc` | float \| None | 旧モデルの AUC |
| `model_updated` | bool | モデルファイルが更新されたか |
| `rollback_reason` | str \| None | ロールバック時の理由 |
| `error` | str \| None | エラーメッセージ |
| `duration_seconds` | float | パイプライン実行時間（秒） |

---

## 4. Data Model

### retraining_log テーブル DDL

```sql
CREATE TABLE IF NOT EXISTS retraining_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_by      TEXT    NOT NULL,   -- "auto" | "manual_streamlit" | "manual_cli"
    status            TEXT    NOT NULL,   -- "success" | "skipped" | "rolled_back" | "error"
    records_used      INTEGER,
    new_auc           REAL,
    prev_auc          REAL,
    model_updated     INTEGER NOT NULL DEFAULT 0,  -- 0=false, 1=true
    rollback_reason   TEXT,
    error_message     TEXT,
    duration_seconds  REAL,
    started_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at      TEXT
);
```

### Python TypedDict

```python
from typing import TypedDict, Optional

class RetrainingResult(TypedDict):
    status: str                      # "success" | "skipped" | "rolled_back" | "error"
    records_used: int
    new_auc: Optional[float]
    prev_auc: Optional[float]
    model_updated: bool
    rollback_reason: Optional[str]
    error: Optional[str]
    duration_seconds: float
```

### モデルファイル管理

```
models/
├── spread_predictor_v2.pkl        （現行モデル：スコアリングが参照）
├── spread_predictor_v2.bak.pkl    （バックアップ：再学習前に自動コピー）
├── lgbm_model.pkl                 （現行 LGBM モデル）
└── lgbm_model.bak.pkl             （バックアップ）
```

---

## 5. API / Interface

### 主要関数シグネチャ

```python
def run_retraining(
    min_records: int = 50,
    auc_threshold: float = 0.75,
    test_ratio: float = 0.2,
    model_dir: str = "models/",
    db_path: str = "data/lease_data.db",
    dry_run: bool = False,
    triggered_by: str = "auto",
) -> RetrainingResult:
    """
    screening_records テーブルのデータで RF/LGBM を再学習する。

    フェイルセーフ: AUC が auc_threshold 未満なら旧モデルを維持（ロールバック）。
    例外を外部に伝播させない。
    """

def check_retraining_needed(
    min_records: int = 50,
    db_path: str = "data/lease_data.db",
) -> bool:
    """
    再学習が必要かどうかを判定する。
    outcome が NULL でない screening_records のレコード数が min_records 以上か確認。
    """
```

### Streamlit 管理画面 UI

```python
# pages/admin.py（または tune_lease_55.py の管理タブ）に追加
if st.button("モデル再学習を実行"):
    result = run_retraining(triggered_by="manual_streamlit")
    if result["status"] == "success":
        st.success(f"再学習完了: AUC {result['new_auc']:.3f}")
    elif result["status"] == "rolled_back":
        st.warning(f"ロールバック: {result['rollback_reason']}")
    elif result["status"] == "skipped":
        st.info(f"スキップ: データ不足（{result['records_used']}件）")
    else:
        st.error(f"エラー: {result['error']}")
```

---

## 6. Business Rules

**BR-421**: レコード数チェック（最低 50件）
- 条件：`outcome` が NULL でない `screening_records` のレコード数が `min_records`（デフォルト 50件）未満
- 処理：再学習を実行せず `status="skipped", records_used=N` を返す
- 根拠：少数データでの再学習はモデル性能を悪化させるリスクがある

**BR-422**: 再学習前バックアップ
- 条件：再学習開始時（レコード数チェック通過後）
- 処理：`spread_predictor_v2.pkl` → `spread_predictor_v2.bak.pkl`、`lgbm_model.pkl` → `lgbm_model.bak.pkl` にコピーする
- 根拠：ロールバック時の復元元を確保する

**BR-423**: AUC 閾値によるロールバック
- 条件：新モデルの AUC（テストセットで評価）が `auc_threshold`（デフォルト 0.75）未満
- 処理：`.bak.pkl` から元のモデルを復元し `status="rolled_back"` を返す。`retraining_log` にロールバック理由を記録する
- 根拠：性能劣化したモデルを本番投入しないためのフェイルセーフ

**BR-424**: 学習データの outcome フィルタリング
- 条件：教師データの作成時
- 処理：`outcome` が `"contracted"` または `"delinquent"` または `"completed"` のレコードのみを使用。`"lost"` と `NULL` は除外する。目的変数は `"delinquent"` → 1、それ以外 → 0 とする
- 根拠：失注案件は審査スコアとアウトカムの因果関係が不明確（選択バイアス）

**BR-425**: 再学習結果の全件ロギング
- 条件：`run_retraining()` 完了時（成功・失敗・スキップ・ロールバック問わず）
- 処理：`retraining_log` テーブルに実行結果を INSERT する
- 根拠：再学習の履歴追跡・監査のため

**BR-426**: 並行実行防止
- 条件：`run_retraining()` が既に実行中の場合（ファイルロックで検出）
- 処理：`status="skipped", error="retraining already in progress"` を返す
- 根拠：同時実行でモデルファイルが壊れるのを防ぐ

---

## 7. UI / UX

Streamlit 管理画面（`pages/admin.py`）に再学習セクションを追加する。

```
┌─────────────────────────────────────────────────┐
│ モデル管理                                        │
│                                                   │
│ 現在の審査データ: 127件（うち結果確定: 83件）        │
│ 最終再学習: 2026-04-20  AUC: 0.812               │
│                                                   │
│ [モデル再学習を実行]                               │
│                                                   │
│ 再学習履歴                                         │
│ 日時             件数  AUC     状態                │
│ 2026-04-20 09:00  65  0.812   success             │
│ 2026-03-15 14:30  52  0.698   rolled_back         │
│ 2026-03-15 14:25  52  —       error               │
└─────────────────────────────────────────────────┘
```

- 再学習中はスピナー表示（`st.spinner("再学習中...")`）
- 再学習履歴は `retraining_log` テーブルから直近10件を表示

---

## 8. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| レコード不足（BR-421） | `status="skipped"` を返す | バックアップ・ロールバック不要 |
| バックアップコピー失敗 | `status="error"` を返す。再学習を開始しない | ファイルを壊すリスクを避ける |
| 学習時の例外（numpy/sklearn エラー等） | バックアップから復元して `status="rolled_back"` を返す | |
| AUC 閾値未満（BR-423） | バックアップから復元して `status="rolled_back"` を返す | |
| 並行実行検出（BR-426） | `status="skipped"` を返す | |
| `retraining_log` への書き込み失敗 | ログエラーとして記録するが `run_retraining()` の戻り値には影響しない | |
| 予期しない例外全般 | 最大限の復元（バックアップから）を試みて `status="error"` を返す | |

---

## 9. Acceptance Criteria

**AC-1201**: レコード数が 50件未満でスキップされる
- Given: `outcome` が NULL でない `screening_records` レコードが 30件
- When: `run_retraining(min_records=50)` を呼ぶ
- Then: `status="skipped"` かつ `model_updated=False` かつモデルファイルが変更されない

**AC-1202**: レコード数が 50件以上で再学習が実行される
- Given: 有効な `screening_records` が 60件（outcome 確定済み）
- When: `run_retraining(min_records=50, dry_run=True)` を呼ぶ
- Then: `status` が `"success"` または `"rolled_back"` であり `records_used >= 50`

**AC-1203**: 再学習前にバックアップが作成される
- Given: `models/spread_predictor_v2.pkl` が存在する
- When: `run_retraining(...)` を呼ぶ（レコード数十分）
- Then: `models/spread_predictor_v2.bak.pkl` が存在する

**AC-1204**: AUC が閾値未満でロールバックされる
- Given: モックした再学習が AUC=0.60 を返す（閾値 0.75 未満）
- When: `run_retraining(auc_threshold=0.75)` を呼ぶ
- Then: `status="rolled_back"` かつ `model_updated=False` かつ旧モデルファイルが復元されている

**AC-1205**: AUC が閾値以上でモデルが更新される
- Given: モックした再学習が AUC=0.82 を返す（閾値 0.75 以上）
- When: `run_retraining(auc_threshold=0.75)` を呼ぶ
- Then: `status="success"` かつ `model_updated=True`

**AC-1206**: 全実行結果が retraining_log に記録される
- Given: `run_retraining()` を3回呼ぶ（skipped / success / rolled_back）
- When: 全3回の呼び出しが完了する
- Then: `retraining_log` に3件のレコードが存在する

**AC-1207**: dry_run=True でモデルファイルが更新されない
- Given: 有効な 60件のレコード、AUC 閾値以上
- When: `run_retraining(dry_run=True)` を呼ぶ
- Then: `model_updated=False` かつモデルファイルのタイムスタンプが変化しない

**AC-1208**: check_retraining_needed が正しく判定する
- Given: outcome 確定済みレコードが 50件
- When: `check_retraining_needed(min_records=50)` を呼ぶ
- Then: `True` が返る

**AC-1209**: check_retraining_needed が 49件で False を返す
- Given: outcome 確定済みレコードが 49件
- When: `check_retraining_needed(min_records=50)` を呼ぶ
- Then: `False` が返る

**AC-1210**: 再学習中に2回目を呼ぶとスキップされる
- Given: `run_retraining()` が実行中（ファイルロック保持中）
- When: 別スレッドから `run_retraining()` を呼ぶ
- Then: `status="skipped", error` に `"already in progress"` が含まれる

**AC-1211**: 学習時の例外でロールバックされる
- Given: sklearn が学習中に例外を raise するモック
- When: `run_retraining()` を呼ぶ
- Then: 例外が外部に伝播せず `status="rolled_back"` または `status="error"` が返り、バックアップから旧モデルが復元されている

**AC-1212**: Streamlit 管理画面で再学習ボタンが表示される
- Given: Streamlit アプリが起動している
- When: 管理画面（admin タブ / pages/admin.py）を開く
- Then: 「モデル再学習を実行」ボタンと再学習履歴テーブルが表示される

---

## 10. Non-Functional Requirements

- **パフォーマンス**: 60件での再学習（RF + LGBM）が 60秒以内に完了すること（CPU処理）
- **後方互換性**: 再学習後もスコアリング API（`scoring_core.py` 等）のインターフェースが変化しない
- **例外非伝播**: いかなる内部エラーも外部に raise しない
- **フェイルセーフ**: バックアップ作成に失敗した場合は再学習を開始しない（既存モデルの保護を最優先）
- **ログ**: 各フェーズ開始・完了時に `[retraining_pipeline] phase={} ...` を出力
- **テストカバレッジ**: AC-1201〜AC-1212 全件カバー必須
- **ファイルロック**: `fcntl.flock()` または `filelock` ライブラリで並行実行を防止（BR-426）
- **既存モジュール不干渉**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/` 以下を変更してはならない

---

## 11. Implementation Notes（Codex向け）

- **ファイル配置**:
  ```
  retraining_pipeline.py     （プロジェクトルート直下、新規作成）
  pages/
  └── admin.py               （新規作成 or 既存に追記）
  tests/
  └── spec_phase4/
      └── test_P4-003.py     （新規作成）
  ```
- **学習ライブラリ**: `scikit-learn` の `RandomForestClassifier` と `lightgbm.LGBMClassifier` を使用。`requirements.txt` にすでに含まれているか確認すること
- **AUC 評価**: `sklearn.metrics.roc_auc_score` でテストセットを評価。クラス不均衡がある場合は `average="weighted"` を使用
- **ファイルロック**: `filelock` ライブラリ（`pip install filelock`）を推奨。ロックファイルは `models/.retraining.lock`
- **目的変数**: BR-424 に従い `"delinquent"` → 1、`"contracted"` / `"completed"` → 0
- **特徴量**: `total_score`, `asset_score`, `tenant_score`, `q_risk_score`, `competitor_pressure_score` を特徴量として使用（NULL は中央値で補完）
- **Streamlit 自動チェック**: `tune_lease_55.py` の起動時（または審査実行後）に `check_retraining_needed()` を呼び、`True` の場合は `st.toast("再学習データが蓄積されました")` でユーザーに通知する
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/q_risk.py`, `aurion/stealth_competitor.py`, `train_quantum.py`

---

## 12. Test Plan

### 単体テスト（Codexが作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_1201 | AC-1201 | 30件 → status="skipped", model_updated=False |
| test_1202 | AC-1202 | 60件, dry_run=True → status in {success, rolled_back} |
| test_1203 | AC-1203 | 再学習後に .bak.pkl が存在する |
| test_1204 | AC-1204 | AUC=0.60モック → status="rolled_back" |
| test_1205 | AC-1205 | AUC=0.82モック → status="success", model_updated=True |
| test_1206 | AC-1206 | 3回実行 → retraining_log に3件 |
| test_1207 | AC-1207 | dry_run=True → model_updated=False, ファイル変更なし |
| test_1208 | AC-1208 | 50件 → check_retraining_needed=True |
| test_1209 | AC-1209 | 49件 → check_retraining_needed=False |
| test_1210 | AC-1210 | 並行実行 → 2回目は status="skipped" |
| test_1211 | AC-1211 | 学習例外モック → 例外非伝播, バックアップ復元 |

### 回帰テスト
- 再学習後に `total_scorer.py` が正常にスコアを返すこと（インターフェース変更なし）
- 既存 `tests/` 以下の全テストが pass すること

### 手動確認（実装後）
- [ ] Streamlit 管理画面で再学習ボタンが表示される
- [ ] 50件未満の状態でボタンを押すと「データ不足」メッセージが表示される
- [ ] 50件以上の状態で再学習が完了し、AUC と状態が表示される
- [ ] 再学習履歴テーブルが `retraining_log` の内容を表示する
