---
spec_id: P2-001
phase: 2
title: aurion/q_risk.py — 財務矛盾検知モジュール
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: []
superseded_by: ""
---

# P2-001 — aurion/q_risk.py — 財務矛盾検知モジュール

---

## 1. Goal

`mobile_app/aurion/q_risk.py` を新規作成し、財務データ内の論理的矛盾・リスクパターンを検知して **Q_risk スコア（0〜100）** と **矛盾パターンリスト** を返す独立モジュールを提供する。既存の RF/LGBM スコアリングロジックには一切触れず、参考値（サイドカー）として動作させる。

---

## 2. Scope

### In scope
- `mobile_app/aurion/__init__.py`（空ファイル）の新規作成
- `mobile_app/aurion/q_risk.py` の新規作成
- 8種の財務矛盾パターン定義（BR-201〜BR-208）
- `detect_q_risk()` エントリポイント関数の実装
- `tests/spec_phase2/test_P2-001.py` のテスト作成（Codex担当）

### Out of scope
- 既存スコアリングモジュール（`scoring_core.py`, `total_scorer.py`, `asset_scorer.py`）の変更
- 既存 `quantum_analysis_module.py` の変更・削除
- APIへの統合（P2-002 で対応）
- UIへの表示（P2-003 で対応）
- DBへの書き込み・機械学習モデルの学習

---

## 3. Inputs / Outputs

### Inputs

| 項目名 | 型 | 必須/任意 | 説明 | 単位 |
|-------|-----|----------|------|------|
| `gross_profit` | float | 任意（省略時 0） | 粗利益 | 百万円 |
| `op_profit` | float | 任意（省略時 0） | 営業利益 | 百万円 |
| `net_income` | float | 任意（省略時 0） | 純利益 | 百万円 |
| `nenshu` | float | 任意（省略時 0） | 年商（売上高） | 百万円 |
| `dep_expense` | float | 任意（省略時 0） | 支払リース料 | 百万円 |
| `depreciation` | float | 任意（省略時 0） | 減価償却費 | 百万円 |
| `machines` | float | 任意（省略時 0） | 機械設備残高 | 百万円 |
| `bank_credit` | float | 任意（省略時 0） | 銀行借入残高 | 百万円 |
| `lease_credit` | float | 任意（省略時 0） | リース残高 | 百万円 |
| `acquisition_cost` | float | 任意（省略時 0） | リース取得額（今回案件） | 百万円 |

### Outputs

| 項目名 | 型 | 説明 |
|-------|-----|------|
| `score` | int | Q_risk スコア（0〜100）。高いほど財務矛盾リスクが高い |
| `level` | str | `"ok"` \| `"caution"` \| `"high_risk"` |
| `patterns` | list[str] | 検知された矛盾パターンコードのリスト |
| `pattern_details` | list[PatternDetail] | 各矛盾パターンの詳細 |

---

## 4. Data Model

```python
from typing import TypedDict

class PatternDetail(TypedDict):
    code: str        # 矛盾パターンコード。例: "FIN-CONTRADICT-001"
    severity: str    # "high" | "medium" | "low"
    message: str     # 日本語メッセージ（審査官向け）
    values: dict     # 検知に使用した実際の値（デバッグ・説明用）

class QRiskResult(TypedDict):
    score: int                       # 0〜100
    level: str                       # "ok" | "caution" | "high_risk"
    patterns: list[str]              # コードリスト
    pattern_details: list[PatternDetail]
```

### 矛盾パターン一覧

| code | severity | 発生条件 |
|------|----------|---------|
| `FIN-CONTRADICT-001` | high | `gross_profit / nenshu` が -0.5 未満 または 1.0 超（粗利率異常） |
| `FIN-CONTRADICT-002` | high | `nenshu == 0` かつ `dep_expense > 0`（売上ゼロ・費用正） |
| `FIN-CONTRADICT-003` | high | `op_profit > gross_profit + 1`（営業利益が粗利を超過、数学的不整合） |
| `FIN-CONTRADICT-004` | medium | `lease_credit / nenshu > 0.5`（リース残高/年商 超過） |
| `FIN-CONTRADICT-005` | medium | `(bank_credit + lease_credit) / nenshu > 1.0`（総債務/年商 超過） |
| `FIN-CONTRADICT-006` | medium | `acquisition_cost / nenshu > 0.3`（今回取得額/年商 超過） |
| `FIN-CONTRADICT-007` | low | `machines > 1.0` かつ `depreciation == 0`（機械設備あり・減価償却ゼロ） |
| `FIN-CONTRADICT-008` | medium | `op_profit > 0` かつ `net_income < op_profit * -2.0`（純利益が営業利益の-2倍未満） |

### level 決定ロジック（スコア閾値）

| score 範囲 | level |
|-----------|-------|
| 0〜19 | `"ok"` |
| 20〜49 | `"caution"` |
| 50〜100 | `"high_risk"` |

### スコア計算式

```
score = min(100, high_count × 35 + medium_count × 12 + low_count × 4)
```

- `high_count`: severity=="high" のパターン数
- `medium_count`: severity=="medium" のパターン数
- `low_count`: severity=="low" のパターン数

---

## 5. API / Interface

### 関数シグネチャ

```python
def detect_q_risk(
    gross_profit: float = 0.0,
    op_profit: float = 0.0,
    net_income: float = 0.0,
    nenshu: float = 0.0,
    dep_expense: float = 0.0,
    depreciation: float = 0.0,
    machines: float = 0.0,
    bank_credit: float = 0.0,
    lease_credit: float = 0.0,
    acquisition_cost: float = 0.0,
) -> QRiskResult:
    """
    財務データから矛盾パターンを検知し、Q_risk スコアを返す。

    スコア計算（RF/LGBM）には影響しない参考値。
    例外を外部に伝播させない設計。入力はすべて百万円単位。

    Returns:
        QRiskResult: score, level, patterns, pattern_details を持つ dict
    """
```

---

## 6. Business Rules

**BR-201**: 粗利率異常検知
- 条件：`nenshu > 0` かつ（`gross_profit / nenshu < -0.5` または `gross_profit / nenshu > 1.0`）
- 処理：`FIN-CONTRADICT-001`（severity="high"）を pattern_details に追加
- 根拠：粗利率が -50% 未満または 100% 超は財務諸表の誤入力・粉飾を示す可能性が高い

**BR-202**: 売上ゼロ・費用正の矛盾検知
- 条件：`nenshu == 0` かつ `dep_expense > 0`
- 処理：`FIN-CONTRADICT-002`（severity="high"）を追加
- 根拠：売上ゼロで支払リース料が発生しているのは、データ入力ミスまたは休眠状態の可能性

**BR-203**: 営業利益・粗利矛盾検知
- 条件：`op_profit > gross_profit + 1`（1百万円の許容差を設ける）
- 処理：`FIN-CONTRADICT-003`（severity="high"）を追加
- 根拠：営業利益 = 粗利 - 販管費 であるため、営業利益が粗利を超えることは数学的に不可能

**BR-204**: リース残高/年商 超過検知
- 条件：`nenshu > 0` かつ `lease_credit / nenshu > 0.5`
- 処理：`FIN-CONTRADICT-004`（severity="medium"）を追加
- 根拠：リース残高が年商の50%超は過剰なオフバランス活用であり、実質的な過剰債務を示す

**BR-205**: 総債務/年商 超過検知
- 条件：`nenshu > 0` かつ `(bank_credit + lease_credit) / nenshu > 1.0`
- 処理：`FIN-CONTRADICT-005`（severity="medium"）を追加
- 根拠：総借入が年商を超える場合、返済能力に疑義が生じる（社内基準）

**BR-206**: 取得額/年商 超過検知
- 条件：`nenshu > 0` かつ `acquisition_cost / nenshu > 0.3`
- 処理：`FIN-CONTRADICT-006`（severity="medium"）を追加
- 根拠：今回リース取得額が年商の30%超は、規模に対して過大な設備投資を示す

**BR-207**: 減価償却費欠落検知
- 条件：`machines > 1.0` かつ `depreciation == 0`
- 根拠：機械設備残高がある場合、通常は減価償却費が発生する。ゼロは入力漏れの可能性
- 処理：`FIN-CONTRADICT-007`（severity="low"）を追加

**BR-208**: 純利益・営業利益の大幅乖離検知
- 条件：`op_profit > 0` かつ `net_income < op_profit * -2.0`
- 処理：`FIN-CONTRADICT-008`（severity="medium"）を追加
- 根拠：営業利益が正でも純利益が著しくマイナスの場合、特別損失や隠れた債務の可能性

---

## 7. UI / UX

本SPECでは UI は対象外。P2-003 で index.html の表示を定義する。

---

## 8. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| `nenshu == 0`（ゼロ除算リスク） | 比率計算をスキップ（BR-204/205/206 は非適用） | BR-202 は別途チェック |
| 全入力がゼロ | `score=0, level="ok", patterns=[]` を返す | エラーにしない |
| 予期しない例外 | `score=0, level="ok", patterns=[], pattern_details=[]` を返す | ログ出力あり |
| 負の入力値 | そのまま計算に使用（異常検知の対象になり得る） | バリデーションなし |

---

## 9. Acceptance Criteria

**AC-401**: 粗利率異常（100%超）で high_risk が返る
- Given: `nenshu=100.0, gross_profit=120.0`（粗利率 120%）
- When: `detect_q_risk(gross_profit=120.0, nenshu=100.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-001"` が含まれ、`level == "high_risk"`

**AC-402**: 粗利率異常（-50%未満）で high が返る
- Given: `nenshu=100.0, gross_profit=-60.0`（粗利率 -60%）
- When: `detect_q_risk(gross_profit=-60.0, nenshu=100.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-001"` が含まれる

**AC-403**: 正常な粗利率で FIN-CONTRADICT-001 が出ない
- Given: `nenshu=100.0, gross_profit=30.0`（粗利率 30%）
- When: `detect_q_risk(gross_profit=30.0, nenshu=100.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-001"` が含まれない

**AC-404**: 売上ゼロ・費用正の矛盾を検知する
- Given: `nenshu=0, dep_expense=5.0`
- When: `detect_q_risk(nenshu=0.0, dep_expense=5.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-002"` が含まれる

**AC-405**: 営業利益が粗利を超える矛盾を検知する
- Given: `gross_profit=50.0, op_profit=60.0`（op > gross + 1）
- When: `detect_q_risk(gross_profit=50.0, op_profit=60.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-003"` が含まれ、severity=="high"

**AC-406**: 許容差1百万円以内は FIN-CONTRADICT-003 が出ない
- Given: `gross_profit=50.0, op_profit=51.0`（差 = 1.0、許容範囲内）
- When: `detect_q_risk(gross_profit=50.0, op_profit=51.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-003"` が含まれない

**AC-407**: リース残高/年商超過を検知する
- Given: `nenshu=100.0, lease_credit=60.0`（60%超）
- When: `detect_q_risk(nenshu=100.0, lease_credit=60.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-004"` が含まれる

**AC-408**: 総債務/年商超過を検知する
- Given: `nenshu=100.0, bank_credit=80.0, lease_credit=40.0`（計120%）
- When: `detect_q_risk(nenshu=100.0, bank_credit=80.0, lease_credit=40.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-005"` が含まれる

**AC-409**: 取得額/年商超過を検知する
- Given: `nenshu=100.0, acquisition_cost=35.0`（35%超）
- When: `detect_q_risk(nenshu=100.0, acquisition_cost=35.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-006"` が含まれる

**AC-410**: 機械設備あり・減価償却ゼロを検知する
- Given: `machines=10.0, depreciation=0.0`
- When: `detect_q_risk(machines=10.0, depreciation=0.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-007"` が含まれ、severity=="low"

**AC-411**: 純利益・営業利益の大幅乖離を検知する
- Given: `op_profit=10.0, net_income=-25.0`（net_income < op_profit × -2 = -20）
- When: `detect_q_risk(op_profit=10.0, net_income=-25.0)` を呼ぶ
- Then: `patterns` に `"FIN-CONTRADICT-008"` が含まれる

**AC-412**: 全入力ゼロで ok が返る
- Given: 全パラメータ = 0.0
- When: `detect_q_risk()` を呼ぶ（全デフォルト）
- Then: `score == 0` かつ `level == "ok"` かつ `patterns == []`

**AC-413**: スコア計算の検証（high × 1 件）
- Given: `FIN-CONTRADICT-001` のみ発生（high × 1）
- When: `detect_q_risk(gross_profit=120.0, nenshu=100.0)` を呼ぶ
- Then: `score == 35` かつ `level == "caution"`

**AC-414**: スコア計算の検証（high × 2 件で high_risk）
- Given: `FIN-CONTRADICT-001` と `FIN-CONTRADICT-003` が同時発生（high × 2）
- When: `detect_q_risk(gross_profit=120.0, nenshu=100.0, op_profit=130.0)` を呼ぶ
- Then: `score == 70` かつ `level == "high_risk"`

**AC-415**: パフォーマンス要件（1件50ms以内）
- Given: 任意の有効入力
- When: `detect_q_risk()` を100回連続で呼ぶ
- Then: 合計処理時間が 5000ms 以内

---

## 10. Non-Functional Requirements

- **パフォーマンス**: 1件あたり 50ms 以内（CPU処理のみ、DB参照・ML推論なし）
- **後方互換性**: 既存モジュールの出力値は一切変更しない
- **例外非伝播**: 関数内部でいかなる例外が発生しても外部に伝播させない。`score=0, level="ok"` を返す
- **ログ**: `patterns` が1件以上ある場合は `[aurion.q_risk]` プレフィックスで `patterns`, `score` をprint出力する
- **テストカバレッジ**: AC-401〜AC-415 全件カバー必須
- **既存モジュール不干渉**: `quantum_analysis_module.py` を import・変更・削除してはならない

---

## 11. Implementation Notes（Codex向け）

- **ファイル配置**:
  ```
  mobile_app/
  ├── aurion/
  │   ├── __init__.py   （空ファイル）
  │   └── q_risk.py
  ```
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `mobile_app/api.py`（本SPECでは）
- **外部依存なし**: 標準ライブラリのみ使用。numpy/pandas は不要
- **入力単位**: 全て百万円。変換は呼び出し側（api.py）が責任を持つ
- **ゼロ除算ガード**: `nenshu == 0` の場合は比率計算（BR-204/205/206）をスキップ
- **テストファイル**: `tests/spec_phase2/test_P2-001.py` に作成
- **インポート方法（api.py から）**:
  ```python
  from aurion.q_risk import detect_q_risk
  ```

---

## 12. Test Plan

### 単体テスト（Codex が作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_401 | AC-401 | 粗利率120% → FIN-CONTRADICT-001, high_risk |
| test_402 | AC-402 | 粗利率-60% → FIN-CONTRADICT-001 |
| test_403 | AC-403 | 粗利率30% → FIN-CONTRADICT-001 なし |
| test_404 | AC-404 | 売上ゼロ・費用正 → FIN-CONTRADICT-002 |
| test_405 | AC-405 | op_profit > gross_profit + 1 → FIN-CONTRADICT-003 |
| test_406 | AC-406 | op_profit = gross_profit + 1 → FIN-CONTRADICT-003 なし |
| test_407 | AC-407 | lease_credit/nenshu=60% → FIN-CONTRADICT-004 |
| test_408 | AC-408 | 総債務/nenshu=120% → FIN-CONTRADICT-005 |
| test_409 | AC-409 | acquisition_cost/nenshu=35% → FIN-CONTRADICT-006 |
| test_410 | AC-410 | machines=10, depreciation=0 → FIN-CONTRADICT-007 |
| test_411 | AC-411 | net_income < op_profit×-2 → FIN-CONTRADICT-008 |
| test_412 | AC-412 | 全入力ゼロ → score=0, level="ok" |
| test_413 | AC-413 | high×1 → score=35, caution |
| test_414 | AC-414 | high×2 → score=70, high_risk |
| test_415 | AC-415 | 100回連続 → 5000ms以内 |

### 回帰テスト
- `mobile_app/api.py` の既存テストが全件 pass すること（本SPECでは api.py を変更しないため）
