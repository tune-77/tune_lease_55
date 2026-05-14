---
spec_id: P3-002
phase: 3
title: POST /predict — aurion.competitor_pressure フィールド追加
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: [P3-001]
superseded_by: ""
---

# P3-002 — POST /predict — aurion.competitor_pressure フィールド追加

---

## 1. Goal

`mobile_app/api.py` の `POST /predict` レスポンスの `aurion` オブジェクトに `competitor_pressure` フィールドを追加する。既存の `aurion.q_risk` フィールドは変更しない。スコア計算（RF/LGBM）には影響しない参考値として追加する。

---

## 2. Scope

### In scope
- `mobile_app/api.py` への `detect_stealth_competitor()` 呼び出し追加
- `POST /predict` レスポンスの `aurion.competitor_pressure` フィールド追加
- `GET /health` レスポンスへの `stealth_competitor_module_loaded` フラグ追加
- `tests/spec_phase3/test_P3-002.py` のテスト作成（Codex 担当）

### Out of scope
- スコア計算ロジックの変更
- `aurion.q_risk` フィールドの変更
- 既存フィールドの削除・型変更
- `stealth_competitor.py` モジュール本体の実装（P3-001 で対応）
- UI 表示（P3-003 で対応）

---

## 3. Inputs / Outputs

### APIへの追加入力（リクエスト変更なし）

`POST /predict` のリクエスト JSON への変更はない。`detect_stealth_competitor()` に渡す値は全てリクエスト既存フィールドから取得する。

| `detect_stealth_competitor()` 引数 | 取得元リクエストフィールド | 備考 |
|----------------------------------|--------------------------|------|
| `spread_pred` | `api.py` 内部変数 `spread_pred`（計算済み） | リクエストフィールドではなく内部計算値 |
| `base_rate` | `api.py` 内部変数 `base_rate_val` | |
| `competitor` | リクエスト `competitor`（"競合あり"→1, それ以外→0） | 既存のエンコード済み変数 `competitor` を使用 |
| `competitor_rate` | リクエスト `competitor_rate`（float） | デフォルト 0.0 |
| `grade` | リクエスト `grade`（int） | |
| `acquisition_cost` | リクエスト `acquisition_cost`（百万円） | |
| `nenshu` | リクエスト `nenshu`（百万円） | |

### APIレスポンス拡張

**変更前（P2-002 時点）**:
```json
{
  "aurion": {
    "q_risk": {
      "score": 35,
      "level": "caution",
      "patterns": ["FIN-CONTRADICT-001"],
      "pattern_details": [...]
    }
  }
}
```

**変更後（本SPEC適用後）**:
```json
{
  "aurion": {
    "q_risk": {
      "score": 35,
      "level": "caution",
      "patterns": ["FIN-CONTRADICT-001"],
      "pattern_details": [...]
    },
    "competitor_pressure": {
      "score": 40,
      "level": "caution",
      "patterns": ["COMP-STEALTH-001"],
      "pattern_details": [
        {
          "code": "COMP-STEALTH-001",
          "severity": "high",
          "message": "競合未申告ですが推奨スプレッドが1.5%未満です。ステルス競合の存在を確認してください。",
          "values": {
            "spread_pred": 1.2,
            "threshold": 1.5,
            "competitor": 0
          }
        }
      ]
    }
  }
}
```

---

## 4. Data Model

`competitor_pressure` フィールドの型は `StealthCompetitorResult`（P3-001 で定義）に準拠する。

---

## 5. API / Interface

### api.py への変更箇所

P2-002 の `detect_q_risk()` 呼び出しパターンに倣い、以下のパターンで追加する。

```python
# P3-002: aurion stealth_competitor 検知（参考値、スコアに影響しない）
_STEALTH_FALLBACK = {"score": 0, "level": "ok", "patterns": [], "pattern_details": []}
try:
    from aurion.stealth_competitor import detect_stealth_competitor as _detect_stealth
    stealth_result = _detect_stealth(
        spread_pred=spread_pred,
        base_rate=base_rate_val,
        competitor=competitor,
        competitor_rate=comp_rate,
        grade=grade,
        acquisition_cost=acq,
        nenshu=ns,
    )
except Exception:
    stealth_result = _STEALTH_FALLBACK

if stealth_result["level"] in ("caution", "high_risk"):
    print(f"[api.aurion.stealth] level={stealth_result['level']} score={stealth_result['score']} patterns={stealth_result['patterns']}")
```

### レスポンス JSON への追加

```python
return jsonify({
    # ... 既存フィールドは変更なし ...
    "aurion": {
        "q_risk": q_risk_result,          # P2-002 からの既存フィールド（変更なし）
        "competitor_pressure": stealth_result,  # 本SPEC で追加
    },
})
```

### GET /health への追加フラグ

```python
"stealth_competitor_module_loaded": _stealth_loaded,  # bool
```

`_stealth_loaded` は `try: from aurion.stealth_competitor import detect_stealth_competitor; _stealth_loaded = True` で起動時にセットする。

---

## 6. Business Rules

**BR-311**: フォールバック動作
- 条件：`aurion.stealth_competitor` モジュールが未ロード（ImportError）または実行中に例外が発生
- 処理：`{"score": 0, "level": "ok", "patterns": [], "pattern_details": []}` をフォールバック値として使用
- 根拠：サイドカー設計原則 — モジュール障害が主スコア計算に影響してはならない

**BR-312**: 計算順序
- 条件：`spread_pred` の計算完了後（既存コードで `spread_pred = max(-0.5, min(6.0, spread_pred))` のクリップ後）
- 処理：クリップ済みの `spread_pred` を `detect_stealth_competitor()` に渡す
- 根拠：モデル出力値の正規化後の値を使うことで一貫した評価が可能

**BR-313**: レスポンス後方互換
- 条件：常に
- 処理：`aurion` オブジェクトの既存フィールド（`q_risk`）は変更しない。`competitor_pressure` を追加するのみ
- 根拠：既存クライアント（index.html）の破壊的変更を防ぐ

**BR-314**: ログ出力基準
- 条件：`stealth_result["level"]` が `"caution"` または `"high_risk"` の場合
- 処理：`[api.aurion.stealth]` プレフィックスで `level`, `score`, `patterns` をprint出力
- 根拠：P2-002 の `q_risk` ログ出力と一貫したパターン

**BR-315**: health エンドポイントへの追加
- 条件：常に
- 処理：`GET /health` レスポンスに `stealth_competitor_module_loaded: bool` を追加
- 根拠：P2-002 の `aurion_module_loaded` と一貫した可観測性

---

## 7. UI / UX

本SPECでは UI は対象外。P3-003 で index.html の表示を定義する。

---

## 8. Error Handling

| エラー条件 | 処理 |
|-----------|------|
| `aurion.stealth_competitor` の ImportError | フォールバック値を使用。ログに `[api.aurion.stealth] import failed` を出力 |
| `detect_stealth_competitor()` が例外を投げた場合 | `except Exception:` でキャッチしフォールバック値を使用 |
| `spread_pred` が計算できなかった場合（モデル未ロード） | 既存の `spread_pred` デフォルト値（中央値 2.5）をそのまま渡す |

---

## 9. Acceptance Criteria

**AC-801**: POST /predict レスポンスに `aurion.competitor_pressure` が含まれる
- Given: api.py が正常起動し、stealth_competitor モジュールがロード済み
- When: 有効な入力で `POST /predict` を呼ぶ
- Then: レスポンス JSON の `aurion.competitor_pressure` に `score`, `level`, `patterns`, `pattern_details` が存在する

**AC-802**: `aurion.q_risk` フィールドが変更されていない
- Given: P2-002 時点の `aurion.q_risk` レスポンス構造
- When: `POST /predict` を呼ぶ
- Then: `aurion.q_risk.score`, `aurion.q_risk.level`, `aurion.q_risk.patterns`, `aurion.q_risk.pattern_details` が全て存在し、P2-002 時点から型・構造が変わっていない

**AC-803**: stealth_competitor モジュール未ロード時にフォールバックが返る
- Given: `aurion/stealth_competitor.py` が存在しない状態（モジュール削除等）
- When: `POST /predict` を呼ぶ
- Then: `aurion.competitor_pressure` は `{"score": 0, "level": "ok", "patterns": [], "pattern_details": []}` が返り、HTTP ステータスは 200

**AC-804**: spread_pred がクリップ後の値で stealth_competitor に渡される
- Given: 入力から計算された生の spread が -1.0 など範囲外
- When: `POST /predict` を呼ぶ
- Then: `detect_stealth_competitor()` に渡される `spread_pred` は `max(-0.5, min(6.0, raw_spread))` でクリップされた値

**AC-805**: GET /health に `stealth_competitor_module_loaded` が含まれる
- Given: api.py が正常起動
- When: `GET /health` を呼ぶ
- Then: レスポンスに `stealth_competitor_module_loaded: true` が含まれる

**AC-806**: stealth_competitor で caution/high_risk が検知された場合ログが出力される
- Given: `detect_stealth_competitor()` が level="caution" を返す入力
- When: `POST /predict` を呼ぶ
- Then: サーバー標準出力に `[api.aurion.stealth]` を含むログが出力される

**AC-807**: 既存スコア（RF/LGBM）の出力値が変化しない
- Given: Phase 2 完了時点と同一の入力
- When: `POST /predict` を呼ぶ
- Then: `score`, `probability`, `judgment`, `spread_pred` が Phase 2 完了時点から変化しない

**AC-808**: `competitor_pressure.score` が 0〜100 の範囲内
- Given: 任意の有効入力
- When: `POST /predict` を呼ぶ
- Then: `aurion.competitor_pressure.score` は 0 以上 100 以下の整数

---

## 10. Non-Functional Requirements

- **後方互換性**: 既存フィールドの削除・型変更は禁止
- **例外非伝播**: stealth_competitor の例外は `except Exception:` でキャッチし、主スコア計算を阻害しない
- **テストカバレッジ**: AC-801〜AC-808 全件カバー必須

---

## 11. Implementation Notes（Codex向け）

- **変更対象ファイル**: `mobile_app/api.py` のみ（stealth_competitor.py は P3-001 で作成済み前提）
- **追加位置**: P2-002 の q_risk 呼び出しブロック（コメント `# P2-002:` の直後）の後に追記する
- **インポート方式**: 遅延インポート（try/except の中で import）を使い、モジュール未存在時にサーバーが起動できるようにする
- **テストファイル**: `tests/spec_phase3/test_P3-002.py` に作成
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/q_risk.py`, `mobile_app/index.html`（本SPEC では）

---

## 12. Test Plan

### 単体テスト（Codex が作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_801 | AC-801 | POST /predict レスポンスに competitor_pressure が存在する |
| test_802 | AC-802 | aurion.q_risk の構造が変化していない |
| test_803 | AC-803 | モジュール未ロード時はフォールバック値（score=0, ok） |
| test_804 | AC-804 | spread_pred がクリップ後の値で渡される |
| test_805 | AC-805 | GET /health に stealth_competitor_module_loaded |
| test_806 | AC-806 | caution/high_risk 時のログ出力 |
| test_807 | AC-807 | 既存スコア不変（回帰テスト） |
| test_808 | AC-808 | competitor_pressure.score は 0〜100 |

### 回帰テスト
- Phase 1 / Phase 2 の AC（AC-001〜AC-614）が引き続き pass すること
