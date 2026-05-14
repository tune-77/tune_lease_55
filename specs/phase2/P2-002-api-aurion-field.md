---
spec_id: P2-002
phase: 2
title: POST /predict — aurion.q_risk フィールド追加
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: [P2-001]
superseded_by: ""
---

# P2-002 — POST /predict — aurion.q_risk フィールド追加

---

## 1. Goal

`mobile_app/api.py` の `POST /predict` レスポンスに `aurion.q_risk` フィールドを追加し、P2-001 で実装した財務矛盾検知結果を **参考値** として返す。スコア（`score`, `judgment`）の計算ロジックは変更しない。

---

## 2. Scope

### In scope
- `mobile_app/api.py` への `detect_q_risk()` 呼び出しの追加
- `POST /predict` レスポンスへの `"aurion"` フィールド追加
- `GET /health` レスポンスへの `aurion_module_loaded` フラグ追加
- `tests/spec_phase2/test_P2-002.py` のテスト作成（Codex担当）

### Out of scope
- `detect_q_risk()` の実装（P2-001 で対応済み）
- UIへの表示（P2-003 で対応）
- 既存 `score`, `judgment`, `sys_score_b`, `recommended_rate` フィールドの変更
- `POST /predict` のリクエストスキーマ変更（P2-001 の inputs は全て既存フィールドから取得可能）

---

## 3. Inputs / Outputs

### Inputs（変更なし）

`POST /predict` のリクエスト JSON は P1-002 時点から変更なし。`detect_q_risk()` に必要な財務データはすでにリクエストに含まれている（`gross_profit`, `op_profit`, `net_income`, `nenshu`, `dep_expense`, `depreciation`, `machines`, `bank_credit`, `lease_credit`, `acquisition_cost`）。

### Outputs（追加フィールドのみ）

#### レスポンス差分

```json
{
  "score": 72,
  "judgment": "承認",
  "aurion": {
    "q_risk": {
      "score": 35,
      "level": "caution",
      "patterns": ["FIN-CONTRADICT-004"],
      "pattern_details": [
        {
          "code": "FIN-CONTRADICT-004",
          "severity": "medium",
          "message": "リース残高が年商の50%を超えています。過剰なオフバランス活用の可能性があります。",
          "values": {
            "lease_credit": 60.0,
            "nenshu": 100.0,
            "ratio": 0.6
          }
        }
      ]
    }
  }
}
```

#### GET /health レスポンス差分

```json
{
  "aurion_module_loaded": true
}
```

---

## 4. Data Model

`QRiskResult` は P2-001 で定義済み。本SPECではラップのみ。

```python
# レスポンスの aurion フィールド構造
{
    "aurion": {
        "q_risk": QRiskResult   # P2-001 の detect_q_risk() 戻り値そのまま
    }
}
```

---

## 5. API / Interface

### POST /predict 変更箇所（api.py）

```python
# P2-002: aurion q_risk 検知（参考値、スコアに影響しない）
try:
    from aurion.q_risk import detect_q_risk
    q_risk_result = detect_q_risk(
        gross_profit=gp,
        op_profit=op,
        net_income=ni,
        nenshu=ns,
        dep_expense=dep,
        depreciation=depr,
        machines=mach,
        bank_credit=bk,
        lease_credit=lc,
        acquisition_cost=acq,
    )
    aurion_loaded = True
except Exception:
    q_risk_result = {"score": 0, "level": "ok", "patterns": [], "pattern_details": []}
    aurion_loaded = False

# return jsonify() の中に追加
"aurion": {
    "q_risk": q_risk_result
}
```

### GET /health 変更箇所

```python
"aurion_module_loaded": aurion_loaded,
```

---

## 6. Business Rules

**BR-211**: aurion.q_risk は参考値
- 条件：常に適用
- 処理：`aurion.q_risk` フィールドは `score`, `judgment` の計算後に独立して呼び出す。その結果が既存フィールドに影響してはならない
- 根拠：Phase 2 方針「既存 RF/LR には触れない」

**BR-212**: aurion モジュール未ロード時のフォールバック
- 条件：`from aurion.q_risk import detect_q_risk` が ImportError 等で失敗した場合
- 処理：`aurion.q_risk = {"score": 0, "level": "ok", "patterns": [], "pattern_details": []}` を返す（APIは200で成功）
- 根拠：aurion モジュールの失敗で審査 API 全体を止めてはならない

**BR-213**: q_risk 呼び出しタイミング
- 条件：常に適用
- 処理：`detect_q_risk()` は既存スコア予測（RF/LGBM）の完了後に呼び出す
- 根拠：既存処理フローへの割り込みを防ぐ

**BR-214**: ログ出力
- 条件：`q_risk_result["level"] in ("caution", "high_risk")` の場合
- 処理：`[api.aurion]` プレフィックスで `level`, `score`, `patterns` をprint出力する
- 根拠：要注意案件の可視性確保

**BR-215**: health エンドポイントへのフラグ追加
- 条件：常に適用
- 処理：`GET /health` レスポンスに `aurion_module_loaded: bool` を追加する
- 根拠：本番監視・デバッグ時に aurion モジュールのロード状態を確認できるようにする

---

## 7. UI / UX

本SPECでは UI は対象外。P2-003 で index.html の表示を定義する。

---

## 8. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| `detect_q_risk()` が例外をスローした場合 | `aurion.q_risk = {score:0, level:"ok", ...}` を返す | API は200成功 |
| `aurion/q_risk.py` が存在しない（ImportError）| フォールバック値を返す | API は200成功 |
| `detect_q_risk()` が50ms超過した場合 | そのまま待つ（タイムアウトなし） | P2-001 のパフォーマンス要件で担保 |

---

## 9. Acceptance Criteria

**AC-501**: aurion.q_risk フィールドが POST /predict レスポンスに含まれる
- Given: aurion/q_risk.py が正常にロードされている
- When: `POST /predict` を任意の有効な JSON で呼ぶ
- Then: レスポンス JSON に `"aurion"` キーが存在し、`"aurion.q_risk"` が `score`, `level`, `patterns`, `pattern_details` を持つ

**AC-502**: 財務矛盾がある入力で q_risk.score > 0 が返る
- Given: `gross_profit=120, nenshu=100`（粗利率120%の矛盾あり）
- When: `POST /predict` で上記を含む JSON を送る
- Then: `aurion.q_risk.score > 0` かつ `aurion.q_risk.patterns` に `"FIN-CONTRADICT-001"` が含まれる

**AC-503**: 財務矛盾がない入力で q_risk.level == "ok" が返る
- Given: 正常な財務データ（粗利率30%、リース残高/年商30%等）
- When: `POST /predict` で上記を含む JSON を送る
- Then: `aurion.q_risk.level == "ok"` かつ `aurion.q_risk.patterns == []`

**AC-504**: 既存フィールド（score, judgment）が aurion 追加後も変化しない
- Given: aurion 追加前後で同一入力
- When: `POST /predict` を呼ぶ
- Then: `score`, `judgment`, `sys_score_b`, `recommended_rate` の値が P1-002 時点と同一（aurion は独立）

**AC-505**: aurion モジュール未ロード時もAPIが200で成功する
- Given: `aurion/` ディレクトリを削除した環境（モック）
- When: `POST /predict` を呼ぶ
- Then: HTTPステータス200 かつ `aurion.q_risk.score == 0` かつ `aurion.q_risk.level == "ok"`

**AC-506**: GET /health に aurion_module_loaded フィールドが追加される
- Given: aurion/q_risk.py が正常にロードされている
- When: `GET /health` を呼ぶ
- Then: レスポンス JSON に `"aurion_module_loaded": true` が含まれる

**AC-507**: caution/high_risk 時にログが出力される
- Given: `q_risk.level == "caution"` となる入力
- When: `POST /predict` を呼ぶ
- Then: `[api.aurion]` プレフィックスのログが stdout に出力される

**AC-508**: pattern_details に values（実際の値）が含まれる
- Given: `FIN-CONTRADICT-004`（リース残高/年商超過）が発生する入力
- When: `POST /predict` を呼ぶ
- Then: `pattern_details[].values` に `lease_credit`, `nenshu`, `ratio` が含まれる

---

## 10. Non-Functional Requirements

- **後方互換性**: 既存フィールド（`score`, `judgment`, `sys_score_b`, `spread_pred`, `recommended_rate`, `warnings`, `rule_check_status`）は変更しない
- **パフォーマンス**: `detect_q_risk()` の呼び出しにより `POST /predict` のレイテンシ増加が 50ms 以内であること
- **ログ**: `[api.aurion]` プレフィックスで出力。既存の `[api]` ログと区別する
- **テストカバレッジ**: AC-501〜AC-508 全件カバー必須

---

## 11. Implementation Notes（Codex向け）

- **変更対象ファイル**: `mobile_app/api.py` のみ
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `mobile_app/lease_rule_checks.py`
- **import 位置**: `try/except` ブロック内（モジュール未ロード時のフォールバックのため）。ファイルトップレベルの import にしてはならない
- **呼び出し位置**: `return jsonify({...})` の直前。スコア計算ブロックの後
- **値の渡し方**: `gp`, `op`, `ni`, `ns`, `dep`, `depr`, `mach`, `bk`, `lc`, `acq` はすでに `predict()` 関数内で `float` に変換済みなので、そのまま渡す
- **テストファイル**: `tests/spec_phase2/test_P2-002.py` に作成。Flask テストクライアント（`app.test_client()`）を使用
- **aurion module の import パス**: `mobile_app/` 内から `from aurion.q_risk import detect_q_risk`（`aurion/` が `mobile_app/` 直下にあるため）

---

## 12. Test Plan

### 単体テスト（Codex が作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_501 | AC-501 | /predict レスポンスに aurion.q_risk キーが存在する |
| test_502 | AC-502 | 粗利率120%入力 → q_risk.score > 0, FIN-CONTRADICT-001 |
| test_503 | AC-503 | 正常財務データ → q_risk.level == "ok" |
| test_504 | AC-504 | aurion 追加後も score/judgment が変化しない |
| test_505 | AC-505 | aurion モジュール未ロード時 → 200成功, score=0 |
| test_506 | AC-506 | /health に aurion_module_loaded フィールドが含まれる |
| test_507 | AC-507 | caution 時に [api.aurion] ログが出る |
| test_508 | AC-508 | pattern_details に values が含まれる |

### 回帰テスト
- `POST /predict` の既存フィールド（`score`, `judgment` 等）が P1-002 時点と同一値を返すこと
- `GET /health` の既存フィールドが変化しないこと
