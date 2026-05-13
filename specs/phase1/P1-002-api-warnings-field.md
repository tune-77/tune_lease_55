---
spec_id: P1-002
phase: 1
title: POST /predict APIレスポンスへの warnings[] フィールド追加
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: [P1-001]
superseded_by: ""
---

# P1-002 — POST /predict APIレスポンスへの warnings[] フィールド追加

---

## 1. Goal

`mobile_app/api.py` の `POST /predict` エンドポイントに P1-001 で作成した `check_lease_rules()` を組み込み、APIレスポンスに `warnings[]` および `rule_check_status` フィールドを追加する。スコア計算（score, probability, judgment）には一切影響しない。

---

## 2. Scope

### In scope
- `mobile_app/api.py` の `POST /predict` ハンドラに `check_lease_rules()` 呼び出しを追加
- リクエストボディに新規オプションフィールド 4件を追加
- レスポンスに `warnings[]` および `rule_check_status` フィールドを追加

### Out of scope
- スコア計算ロジック（score, probability, judgment）の変更
- 既存レスポンスフィールドの削除・型変更
- `GET /health` エンドポイントの変更
- UIの変更（P1-003 で対応）
- `lease_rule_checks.py` の実装（P1-001 で対応）

---

## 3. Inputs / Outputs

### 新規リクエストフィールド（POST /predict ボディへの追加分）

| 項目名 | 型 | 必須/任意 | デフォルト | 説明 |
|-------|-----|----------|----------|------|
| `asset_type` | str | 任意 | `""` | 物件種別名（例: "電子計算機"）。空文字の場合、耐用年数チェックはスキップ |
| `is_re_lease` | bool | 任意 | `false` | 再リース予定フラグ |
| `insurance_applicable` | str | 任意 | `"不明"` | 動産保険付保状況。"付保済" \| "未付保" \| "不明" |
| `re_lease_insurance` | str | 任意 | `"不明"` | 再リース保険付保状況。"付保済" \| "未付保" \| "不明" |

### 新規レスポンスフィールド（既存フィールドへの追加分）

| 項目名 | 型 | 説明 |
|-------|-----|------|
| `warnings` | list[dict] | WarningItem のリスト（P1-001 の WarningItem 定義参照）。警告なしの場合は `[]` |
| `rule_check_status` | str | "ok" \| "warning" \| "high_risk" \| "unknown" \| "skipped" |

### 既存レスポンスフィールド（変更なし）

```json
{
  "score":             <int>,
  "probability":       <float>,
  "judgment":          <str>,
  "sys_score_b":       <float>,
  "model_pipeline":    <str>,
  "spread_pred":       <float>,
  "base_rate":         <float>,
  "recommended_rate":  <float>,
  "rate_range": {
    "low":  <float>,
    "high": <float>
  }
}
```

---

## 4. Data Model

### 完全なレスポンス例（warnings あり）

```json
{
  "score": 72,
  "probability": 0.7234,
  "judgment": "承認",
  "sys_score_b": 68.2,
  "model_pipeline": "LGBM→RF",
  "spread_pred": 1.85,
  "base_rate": 2.10,
  "recommended_rate": 3.95,
  "rate_range": {
    "low": 3.65,
    "high": 4.25
  },
  "warnings": [
    {
      "code": "TERM_EXCEEDS_LEGAL_LIFE",
      "severity": "high",
      "message": "リース期間が法定耐用年数（4年）を超えています（リース期間: 5年）。減価償却上の問題が生じる可能性があります。",
      "source": "法人税法施行令別表第一"
    },
    {
      "code": "INSURANCE_NOT_COVERED",
      "severity": "low",
      "message": "動産保険が未付保です。物件滅失リスクのため付保を推奨します。",
      "source": "リース会社内規"
    }
  ],
  "rule_check_status": "high_risk"
}
```

### 完全なレスポンス例（warnings なし）

```json
{
  "score": 55,
  "probability": 0.5512,
  "judgment": "条件付",
  ...（既存フィールド）...,
  "warnings": [],
  "rule_check_status": "ok"
}
```

---

## 5. API / Interface

### POST /predict ハンドラ変更箇所

```python
# 追加するフィールド取得（既存フィールド取得ブロックの末尾に追加）
asset_type_str      = str(data.get("asset_type", ""))
is_re_lease         = bool(data.get("is_re_lease", False))
insurance_app       = str(data.get("insurance_applicable", "不明"))
re_lease_ins        = str(data.get("re_lease_insurance", "不明"))

# check_lease_rules 呼び出し（スコア計算の直後、jsonify の直前）
try:
    from lease_rule_checks import check_lease_rules
    rule_result = check_lease_rules(
        lease_term_months=int(lt),
        asset_type=asset_type_str,
        is_re_lease=is_re_lease,
        insurance_applicable=insurance_app,
        re_lease_insurance=re_lease_ins,
    )
    warnings        = rule_result["warnings"]
    rule_check_status = rule_result["status"]
except Exception:
    warnings          = []
    rule_check_status = "skipped"

# jsonify に追加
return jsonify({
    ...（既存フィールド）...,
    "warnings":          warnings,
    "rule_check_status": rule_check_status,
})
```

---

## 6. Business Rules

**BR-111**: warnings フィールドはスコアに影響しない
- 条件：常時
- 処理：`check_lease_rules()` の呼び出し結果は `warnings` と `rule_check_status` にのみ格納する。`score`, `probability`, `judgment` は変更しない
- 根拠：Phase 1 の設計方針「スコアには影響しない」（plan.md）

**BR-112**: check_lease_rules 失敗時のフォールバック
- 条件：`check_lease_rules()` のインポートまたは呼び出しで例外が発生した場合
- 処理：`warnings=[]`, `rule_check_status="skipped"` を設定し、スコア計算結果はそのまま返す
- 根拠：このモジュールの障害でスコア計算APIを停止させてはならない

**BR-113**: 既存フィールドの不変性
- 条件：常時
- 処理：`score`, `probability`, `judgment`, `sys_score_b`, `model_pipeline`, `spread_pred`, `base_rate`, `recommended_rate`, `rate_range` の型・値を変更しない
- 根拠：既存のモバイルアプリUIとの後方互換性

**BR-114**: lease_term の型変換
- 条件：`lt` が float（`data.get("lease_term", 60)` はfloat）の場合
- 処理：`int(lt)` に変換して `check_lease_rules()` に渡す
- 根拠：`check_lease_rules()` の `lease_term_months` パラメータは int 型

**BR-115**: rule_check_status="skipped" はエラーではない
- 条件：`check_lease_rules()` がインポートできない場合（環境依存）
- 処理：HTTPステータスは 200 を返す。"skipped" はルールチェック機能が無効であることを示すだけ
- 根拠：APIの可用性を最優先する

---

## 7. UI / UX

本SPECでは UI は対象外。P1-003 で index.html の表示を定義する。

---

## 8. Error Handling

| エラー条件 | 処理 | HTTPステータス |
|-----------|------|-------------|
| 既存のリクエスト数値変換エラー | 既存動作通り `400` を返す（変更なし） | 400 |
| `check_lease_rules` インポート失敗 | `warnings=[]`, `rule_check_status="skipped"`, 通常レスポンス | 200 |
| `check_lease_rules` 呼び出し例外 | `warnings=[]`, `rule_check_status="skipped"`, 通常レスポンス | 200 |
| 新規フィールド（asset_type等）が不正な型 | デフォルト値で処理継続（str変換） | 200 |

---

## 9. Acceptance Criteria

**AC-201**: warnings フィールドが常にレスポンスに含まれる
- Given: `lease_rule_checks.py` が存在しない（または存在する）状態
- When: 最小限のリクエスト（既存フィールドのみ）で `POST /predict` を呼ぶ
- Then: レスポンス JSON に `warnings` キーが存在し、list 型である（値は `[]` または警告リスト）

**AC-202**: rule_check_status フィールドが常にレスポンスに含まれる
- Given: 任意のリクエスト
- When: `POST /predict` を呼ぶ
- Then: レスポンス JSON に `rule_check_status` キーが存在し、文字列型である

**AC-203**: 既存フィールドが変化しない
- Given: 新規フィールドなしの既存リクエストボディ
- When: 変更前後で同一リクエストを送信
- Then: `score`, `probability`, `judgment`, `spread_pred`, `base_rate`, `recommended_rate`, `rate_range` の値が変化しない

**AC-204**: 耐用年数超過の場合に正しい warning が返る
- Given: `asset_type="電子計算機"`, `lease_term=72`（= 6年 > 法定4年）
- When: `POST /predict` を `asset_type` フィールド付きで呼ぶ
- Then: レスポンスの `warnings` に `code=="TERM_EXCEEDS_LEGAL_LIFE"` が含まれ、`rule_check_status == "high_risk"`

**AC-205**: 全チェック通過の場合に warnings が空になる
- Given: `asset_type="工作機械"`, `lease_term=60`（10年法定耐用年数の50%）, 保険付保済み
- When: `POST /predict` を呼ぶ
- Then: `warnings == []` かつ `rule_check_status == "ok"`

**AC-206**: lease_rule_checks 失敗時でも 200 が返り score は正常
- Given: `check_lease_rules()` が例外を throw する状態（モックで再現）
- When: `POST /predict` を呼ぶ
- Then: HTTPステータス 200、`score` は正常値、`rule_check_status == "skipped"`、`warnings == []`

**AC-207**: asset_type が未送信でも 200 が返る（後方互換性）
- Given: 既存クライアントが `asset_type` を送信しない
- When: `asset_type` フィールドなしで `POST /predict` を呼ぶ
- Then: HTTPステータス 200、`warnings == []`、既存フィールドが正常値

**AC-208**: is_re_lease=true かつ re_lease_insurance="未付保" で警告が返る
- Given: `is_re_lease=true`, `re_lease_insurance="未付保"`
- When: `POST /predict` を呼ぶ
- Then: `warnings` に `code=="RE_LEASE_INSURANCE_NOT_COVERED"` が含まれる

---

## 10. Non-Functional Requirements

- **後方互換性**: 新規フィールドは全てオプション。既存クライアントが新規フィールドを送信しなくても正常動作する
- **パフォーマンス**: `check_lease_rules()` の追加による全体レイテンシ増加は 100ms 以内（AC-115 より 50ms 以内のはずだが余裕を持たせる）
- **ログ**: `rule_check_status != "ok"` かつ `rule_check_status != "unknown"` の場合は `[api]` プレフィックスで `rule_check_status` と `warnings` コードリストを print 出力する

---

## 11. Implementation Notes（Codex向け）

- **変更ファイル**: `mobile_app/api.py` のみ
- **インポート位置**: `from lease_rule_checks import check_lease_rules` は `predict()` 関数内のtry-exceptブロック内でインポートする（サーバ起動時のインポートエラーでAPIが停止しないようにするため）
- **lt の型注意**: `lt = float(data.get("lease_term", 60))` で取得されているため、`check_lease_rules` 呼び出し時は `int(lt)` に変換する
- **テストファイル**: `tests/spec_phase1/test_P1-002.py` に作成。Flask test_client を使ったインテグレーションテストを書く
- **触れてはいけないロジック**: `_predict_spread()`, `_build_feat_vector()`, `_compute_sys_score_b_existing()`, スコア計算ブロック全体

---

## 12. Test Plan

### 単体テスト（Codex が作成）
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_201 | AC-201 | 最小リクエスト → warnings キーが存在 |
| test_202 | AC-202 | 任意リクエスト → rule_check_status キーが存在 |
| test_203 | AC-203 | 既存フィールドが変化しないこと（同一リクエスト比較） |
| test_204 | AC-204 | 電子計算機72ヶ月 → TERM_EXCEEDS_LEGAL_LIFE, high_risk |
| test_205 | AC-205 | 工作機械60ヶ月+付保 → warnings=[], ok |
| test_206 | AC-206 | check_lease_rules mock例外 → 200, skipped |
| test_207 | AC-207 | asset_type なし → 200, warnings=[] |
| test_208 | AC-208 | is_re_lease=true+未付保 → RE_LEASE_INSURANCE_NOT_COVERED |

### 回帰テスト
- 既存の `/predict` テスト（test_api.py 等）が全件 pass すること
- score/probability/judgment の値が変化しないこと（同一シード・同一入力で確認）
