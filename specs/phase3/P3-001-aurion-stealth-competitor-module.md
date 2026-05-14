---
spec_id: P3-001
phase: 3
title: aurion/stealth_competitor.py — ステルス競合推定モジュール
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: [P2-001, P2-002]
superseded_by: ""
---

# P3-001 — aurion/stealth_competitor.py — ステルス競合推定モジュール

---

## 1. Goal

`mobile_app/aurion/stealth_competitor.py` を新規作成し、`spread_pred`（RF 回帰モデルが出力した推奨スプレッド）と申告競合情報の乖離を検知することで **ステルス競合圧力スコア（0〜100）** と **圧力パターンリスト** を返す独立モジュールを提供する。既存の RF/LGBM スコアリングロジックには一切触れず、参考値（サイドカー）として動作させる。

---

## 2. Scope

### In scope
- `mobile_app/aurion/stealth_competitor.py` の新規作成
- 4種の競合圧力パターン定義（BR-301〜BR-304）
- `detect_stealth_competitor()` エントリポイント関数の実装
- `tests/spec_phase3/test_P3-001.py` のテスト作成（Codex 担当）

### Out of scope
- 既存スコアリングモジュール（`scoring_core.py`, `total_scorer.py`, `asset_scorer.py`）の変更
- 既存 `quantum_analysis_module.py` / `aurion/q_risk.py` の変更
- APIへの統合（P3-002 で対応）
- UI への表示（P3-003 で対応）
- DB への書き込み・機械学習モデルの学習
- `spread_predictor_v2.pkl` モデル自体の変更・再学習

---

## 3. Inputs / Outputs

### Inputs

| 項目名 | 型 | 必須/任意 | 説明 | 備考 |
|-------|-----|----------|------|------|
| `spread_pred` | float | 必須 | RF モデルが予測した推奨スプレッド（%） | `api.py` 内の `spread_pred` をそのまま渡す |
| `base_rate` | float | 必須 | 基準金利（%） | `_get_period_rate()` の返値 |
| `competitor` | int | 必須 | 競合申告フラグ（0=なし, 1=あり） | |
| `competitor_rate` | float | 任意（省略時 0.0） | 申告競合金利（%） | competitor=0 のときは 0.0 を渡す |
| `grade` | int | 必須 | 顧客格付け（1〜10） | 数値が小さいほど優良 |
| `acquisition_cost` | float | 任意（省略時 0.0） | リース取得額（百万円） | |
| `nenshu` | float | 任意（省略時 0.0） | 年商（百万円） | |

### Outputs

| 項目名 | 型 | 説明 |
|-------|-----|------|
| `score` | int | 競合圧力スコア（0〜100）。高いほどステルス競合の可能性が高い |
| `level` | str | `"ok"` \| `"caution"` \| `"high_risk"` |
| `patterns` | list[str] | 検知された圧力パターンコードのリスト |
| `pattern_details` | list[PatternDetail] | 各パターンの詳細 |

---

## 4. Data Model

```python
from typing import TypedDict

class PatternDetail(TypedDict):
    code: str        # 圧力パターンコード。例: "COMP-STEALTH-001"
    severity: str    # "high" | "medium" | "low"
    message: str     # 日本語メッセージ（審査官向け）
    values: dict     # 検知に使用した実際の値（デバッグ・説明用）

class StealthCompetitorResult(TypedDict):
    score: int                             # 0〜100
    level: str                             # "ok" | "caution" | "high_risk"
    patterns: list[str]                    # コードリスト
    pattern_details: list[PatternDetail]
```

### 圧力パターン一覧

| code | severity | 発生条件 |
|------|----------|---------|
| `COMP-STEALTH-001` | high | `competitor == 0` かつ `spread_pred < 1.5` （競合未申告なのにスプレッドが低すぎる） |
| `COMP-STEALTH-002` | high | `competitor == 1` かつ `competitor_rate > 0` かつ `competitor_rate < base_rate + 0.3` （申告競合金利が相場外れに低い） |
| `COMP-STEALTH-003` | medium | grade ≤ 3 かつ spread_pred < 1.0、または grade 4〜6 かつ spread_pred < 0.8、または grade ≥ 7 かつ spread_pred < 0.5 （格付け対比でスプレッドが過度に圧縮） |
| `COMP-STEALTH-004` | medium | `competitor == 1` かつ `competitor_rate > 0` かつ `abs(spread_pred - (competitor_rate - base_rate)) > 1.5` （申告競合金利と予測スプレッドの乖離が大きい） |

### level 決定ロジック（スコア閾値）

| score 範囲 | level |
|-----------|-------|
| 0〜19 | `"ok"` |
| 20〜49 | `"caution"` |
| 50〜100 | `"high_risk"` |

### スコア計算式

```
score = min(100, high_count × 40 + medium_count × 15 + low_count × 5)
```

- `high_count`: severity=="high" のパターン数
- `medium_count`: severity=="medium" のパターン数
- `low_count`: severity=="low" のパターン数

---

## 5. API / Interface

### 関数シグネチャ

```python
def detect_stealth_competitor(
    spread_pred: float,
    base_rate: float,
    competitor: int,
    competitor_rate: float = 0.0,
    grade: int = 5,
    acquisition_cost: float = 0.0,
    nenshu: float = 0.0,
) -> StealthCompetitorResult:
    """
    spread_pred と申告競合情報の乖離からステルス競合圧力を検知する。

    スコア計算（RF/LGBM）には影響しない参考値（サイドカー）。
    例外を外部に伝播させない設計。

    Returns:
        StealthCompetitorResult: score, level, patterns, pattern_details を持つ dict
    """
```

---

## 6. Business Rules

**BR-301**: 競合未申告・低スプレッド検知
- 条件：`competitor == 0` かつ `spread_pred < 1.5`
- 処理：`COMP-STEALTH-001`（severity="high"）を pattern_details に追加
- 根拠：競合がいないにもかかわらずスプレッドが 1.5% 未満は市場感と乖離しており、営業担当が競合の存在を未申告のまま競争入札相当の条件を提示している可能性が高い

**BR-302**: 相場外れの申告競合金利検知
- 条件：`competitor == 1` かつ `competitor_rate > 0` かつ `competitor_rate < base_rate + 0.3`
- 処理：`COMP-STEALTH-002`（severity="high"）を追加
- 根拠：基準金利 + 0.3% 未満の競合金利は市場慣行から逸脱しており、実在しない・あるいは条件を誇張した競合申告の可能性がある

**BR-303**: 格付け対比スプレッド圧縮検知
- 条件：以下のいずれか
  - `grade <= 3` かつ `spread_pred < 1.0`
  - `4 <= grade <= 6` かつ `spread_pred < 0.8`
  - `grade >= 7` かつ `spread_pred < 0.5`
- 処理：`COMP-STEALTH-003`（severity="medium"）を追加
- 根拠：格付け別の市場スプレッド水準と著しく乖離する場合、未申告の競合圧力が価格決定に影響していると推定される

**BR-304**: 申告競合金利・予測スプレッド乖離検知
- 条件：`competitor == 1` かつ `competitor_rate > 0` かつ `abs(spread_pred - (competitor_rate - base_rate)) > 1.5`
- 処理：`COMP-STEALTH-004`（severity="medium"）を追加
- 根拠：spread_predictor が推定した市場スプレッドと申告競合スプレッドの乖離が 1.5% 超の場合、別の未申告競合が存在するか申告内容が実態と異なる可能性がある

---

## 7. UI / UX

本SPECでは UI は対象外。P3-003 で index.html の表示を定義する。

---

## 8. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| `grade` が 1〜10 の範囲外 | `grade = 5`（中央値）にクリップして計算継続 | エラーにしない |
| `spread_pred` が範囲外（-0.5〜6.0 の外） | そのまま計算に使用（異常値検知の対象になり得る） | バリデーションなし |
| `competitor_rate` が負値 | BR-302 / BR-304 をスキップ（`competitor_rate > 0` 条件で自然に除外） | |
| `base_rate` が 0.0 | そのまま計算に使用 | ゼロ除算なし |
| 予期しない例外 | `score=0, level="ok", patterns=[], pattern_details=[]` を返す | ログ出力あり |

---

## 9. Acceptance Criteria

**AC-701**: 競合未申告・低スプレッドで COMP-STEALTH-001 が検知される
- Given: `competitor=0, spread_pred=1.2, base_rate=1.0, grade=5`
- When: `detect_stealth_competitor(spread_pred=1.2, base_rate=1.0, competitor=0, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-001"` が含まれ、`level == "high_risk"`

**AC-702**: 競合未申告だが spread が 1.5% 以上なら COMP-STEALTH-001 が出ない
- Given: `competitor=0, spread_pred=1.5, base_rate=1.0, grade=5`
- When: `detect_stealth_competitor(spread_pred=1.5, base_rate=1.0, competitor=0, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-001"` が含まれない

**AC-703**: 相場外れの申告競合金利で COMP-STEALTH-002 が検知される
- Given: `competitor=1, competitor_rate=1.1, base_rate=1.0`（comp_rate - base = 0.1 < 0.3）
- When: `detect_stealth_competitor(spread_pred=1.8, base_rate=1.0, competitor=1, competitor_rate=1.1, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-002"` が含まれる

**AC-704**: 申告競合金利が base_rate + 0.3 以上なら COMP-STEALTH-002 が出ない
- Given: `competitor=1, competitor_rate=1.3, base_rate=1.0`（comp_rate - base = 0.3、境界値）
- When: `detect_stealth_competitor(spread_pred=1.8, base_rate=1.0, competitor=1, competitor_rate=1.3, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-002"` が含まれない

**AC-705**: grade ≤ 3 の優良先でスプレッドが 1.0% 未満なら COMP-STEALTH-003 が検知される
- Given: `grade=2, spread_pred=0.8, base_rate=1.0`
- When: `detect_stealth_competitor(spread_pred=0.8, base_rate=1.0, competitor=0, grade=2)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-003"` が含まれる

**AC-706**: grade 4〜6 でスプレッドが 0.8% 未満なら COMP-STEALTH-003 が検知される
- Given: `grade=5, spread_pred=0.6, base_rate=1.0`
- When: `detect_stealth_competitor(spread_pred=0.6, base_rate=1.0, competitor=0, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-003"` が含まれる

**AC-707**: grade ≥ 7 でスプレッドが 0.5% 未満なら COMP-STEALTH-003 が検知される
- Given: `grade=8, spread_pred=0.3, base_rate=1.0`
- When: `detect_stealth_competitor(spread_pred=0.3, base_rate=1.0, competitor=0, grade=8)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-003"` が含まれる

**AC-708**: grade 4〜6 でスプレッドが 0.8% 以上なら COMP-STEALTH-003 が出ない
- Given: `grade=5, spread_pred=0.8, base_rate=1.0`（境界値）
- When: `detect_stealth_competitor(spread_pred=0.8, base_rate=1.0, competitor=0, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-003"` が含まれない

**AC-709**: 申告競合金利と予測スプレッドの乖離が 1.5% 超で COMP-STEALTH-004 が検知される
- Given: `competitor=1, competitor_rate=3.0, base_rate=1.0, spread_pred=1.2`（|(1.2) - (3.0-1.0)| = 0.8）→ 検知なし
- 正しいテスト: `competitor=1, competitor_rate=4.0, base_rate=1.0, spread_pred=1.2`（|(1.2) - (4.0-1.0)| = 1.8 > 1.5）
- When: `detect_stealth_competitor(spread_pred=1.2, base_rate=1.0, competitor=1, competitor_rate=4.0, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-004"` が含まれる

**AC-710**: 申告競合金利と予測スプレッドの乖離が 1.5% 以下では COMP-STEALTH-004 が出ない
- Given: `competitor=1, competitor_rate=2.5, base_rate=1.0, spread_pred=2.0`（|(2.0) - (2.5-1.0)| = 0.5 ≤ 1.5）
- When: `detect_stealth_competitor(spread_pred=2.0, base_rate=1.0, competitor=1, competitor_rate=2.5, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-004"` が含まれない

**AC-711**: competitor=0 で competitor_rate が指定されても BR-302/BR-304 は発動しない
- Given: `competitor=0, competitor_rate=1.1, spread_pred=2.0, base_rate=1.0, grade=5`
- When: `detect_stealth_competitor(spread_pred=2.0, base_rate=1.0, competitor=0, competitor_rate=1.1, grade=5)` を呼ぶ
- Then: `patterns` に `"COMP-STEALTH-002"` も `"COMP-STEALTH-004"` も含まれない

**AC-712**: 全パターン未検知で ok が返る
- Given: `competitor=0, spread_pred=2.5, base_rate=1.0, grade=5`（全ルール非該当）
- When: `detect_stealth_competitor(spread_pred=2.5, base_rate=1.0, competitor=0, grade=5)` を呼ぶ
- Then: `score == 0` かつ `level == "ok"` かつ `patterns == []`

**AC-713**: スコア計算の検証（high × 1 件）
- Given: COMP-STEALTH-001 のみ発生（high × 1）
- When: `detect_stealth_competitor(spread_pred=1.2, base_rate=1.0, competitor=0, grade=5)` を呼ぶ（COMP-STEALTH-001 のみ、grade=5 で COMP-STEALTH-003 非該当）
- Then: `score == 40` かつ `level == "caution"`

**AC-714**: スコア計算の検証（high × 2 件で high_risk）
- Given: COMP-STEALTH-001 と COMP-STEALTH-002 が同時発生（high × 2）
- When: `detect_stealth_competitor(spread_pred=1.2, base_rate=1.0, competitor=1, competitor_rate=1.1, grade=5)` を呼ぶ
- Then: `score == 80` かつ `level == "high_risk"`

**AC-715**: grade 範囲外は 5 にクリップして計算を継続する
- Given: `grade=15, spread_pred=2.5, base_rate=1.0, competitor=0`
- When: `detect_stealth_competitor(spread_pred=2.5, base_rate=1.0, competitor=0, grade=15)` を呼ぶ
- Then: 例外が発生しない

**AC-716**: competitor_rate が負値でも例外が発生しない
- Given: `competitor=1, competitor_rate=-0.5, spread_pred=2.0, base_rate=1.0, grade=5`
- When: `detect_stealth_competitor(spread_pred=2.0, base_rate=1.0, competitor=1, competitor_rate=-0.5, grade=5)` を呼ぶ
- Then: 例外が発生しない（BR-302/BR-304 はスキップ）

**AC-717**: パフォーマンス要件（1件 50ms 以内）
- Given: 任意の有効入力
- When: `detect_stealth_competitor()` を100回連続で呼ぶ
- Then: 合計処理時間が 5000ms 以内

---

## 10. Non-Functional Requirements

- **パフォーマンス**: 1件あたり 50ms 以内（CPU 処理のみ、DB 参照・ML 推論なし）
- **後方互換性**: 既存モジュールの出力値は一切変更しない
- **例外非伝播**: 関数内部でいかなる例外が発生しても外部に伝播させない。`score=0, level="ok"` を返す
- **ログ**: `patterns` が1件以上ある場合は `[aurion.stealth_competitor]` プレフィックスで `patterns`, `score` をprint出力する
- **テストカバレッジ**: AC-701〜AC-717 全件カバー必須
- **既存モジュール不干渉**: `quantum_analysis_module.py`, `aurion/q_risk.py`, `scoring_core.py`, `total_scorer.py`, `asset_scorer.py` を import・変更・削除してはならない

---

## 11. Implementation Notes（Codex向け）

- **ファイル配置**:
  ```
  mobile_app/
  └── aurion/
      ├── __init__.py       （P2-001 で作成済み、変更不要）
      ├── q_risk.py         （P2-001 で作成済み、変更不要）
      └── stealth_competitor.py  （本SPEC で新規作成）
  ```
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/q_risk.py`, `mobile_app/api.py`（本SPEC では）
- **外部依存なし**: 標準ライブラリのみ使用。numpy/pandas は不要
- **grade クリップ**: `grade = max(1, min(10, grade))` で範囲外入力をガードする
- **ゼロ除算なし**: スコア計算に除算はないため追加ガード不要
- **テストファイル**: `tests/spec_phase3/test_P3-001.py` に作成

---

## 12. Test Plan

### 単体テスト（Codex が作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_701 | AC-701 | competitor=0, spread=1.2 → COMP-STEALTH-001, high_risk |
| test_702 | AC-702 | competitor=0, spread=1.5 → COMP-STEALTH-001 なし |
| test_703 | AC-703 | comp_rate=1.1, base=1.0 → COMP-STEALTH-002 |
| test_704 | AC-704 | comp_rate=1.3, base=1.0 → COMP-STEALTH-002 なし（境界値） |
| test_705 | AC-705 | grade=2, spread=0.8 → COMP-STEALTH-003 |
| test_706 | AC-706 | grade=5, spread=0.6 → COMP-STEALTH-003 |
| test_707 | AC-707 | grade=8, spread=0.3 → COMP-STEALTH-003 |
| test_708 | AC-708 | grade=5, spread=0.8 → COMP-STEALTH-003 なし（境界値） |
| test_709 | AC-709 | 乖離1.8% → COMP-STEALTH-004 |
| test_710 | AC-710 | 乖離0.5% → COMP-STEALTH-004 なし |
| test_711 | AC-711 | competitor=0 では BR-302/304 非発動 |
| test_712 | AC-712 | 全パターン未検知 → score=0, ok |
| test_713 | AC-713 | high×1 → score=40, caution |
| test_714 | AC-714 | high×2 → score=80, high_risk |
| test_715 | AC-715 | grade=15 → 例外なし |
| test_716 | AC-716 | competitor_rate < 0 → 例外なし |
| test_717 | AC-717 | 100回連続 → 5000ms以内 |

### 回帰テスト
- `mobile_app/api.py` の既存テストが全件 pass すること（本SPEC では api.py を変更しないため）
