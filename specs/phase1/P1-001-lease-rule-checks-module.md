---
spec_id: P1-001
phase: 1
title: lease_rule_checks.py — リースルールチェックモジュール
status: approved
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: []
superseded_by: ""
---

# P1-001 — lease_rule_checks.py — リースルールチェックモジュール

---

## 1. Goal

`mobile_app/lease_rule_checks.py` を新規作成し、法定耐用年数・期待使用期間・動産保険・再リース保険の4種チェックを独立モジュールとして提供する。スコア計算ロジックには一切触れず、警告（warnings）を返すだけのサイドカー設計とする。

---

## 2. Scope

### In scope
- `mobile_app/lease_rule_checks.py` の新規作成
- 法定耐用年数マスタ（dict）をモジュール内定数として保持
- 下記4種チェック関数の実装
  1. 法定耐用年数チェック（リース期間 > 法定耐用年数）
  2. 期待使用期間チェック（リース期間 > 期待使用期間）
  3. 動産保険チェック（保険未付保の場合に警告）
  4. 再リース保険チェック（再リース予定かつ保険未付保の場合に警告）
- `check_lease_rules()` エントリポイント関数
- `tests/spec_phase1/test_P1-001.py` のテスト作成（Codex担当）

### Out of scope
- 既存スコアリングモジュール（`scoring_core.py`, `total_scorer.py`, `asset_scorer.py`）の変更
- APIへの統合（P1-002 で対応）
- UIへの表示（P1-003 で対応）
- DBへの書き込み
- 法定耐用年数マスタのDB管理・動的更新

---

## 3. Inputs / Outputs

### Inputs
| 項目名 | 型 | 必須/任意 | 説明 |
|-------|-----|----------|------|
| `lease_term_months` | int | 必須 | リース期間（月数）。1以上の整数 |
| `asset_type` | str | 任意 | 物件種別名（例: "電子計算機", "建設機械"）。空文字の場合は耐用年数チェックをスキップ |
| `is_re_lease` | bool | 任意（省略時 False） | 再リース予定フラグ |
| `insurance_applicable` | str | 任意（省略時 "不明"） | 動産保険付保状況。"付保済" \| "未付保" \| "不明" |
| `re_lease_insurance` | str | 任意（省略時 "不明"） | 再リース保険付保状況。"付保済" \| "未付保" \| "不明" |

### Outputs
| 項目名 | 型 | 説明 |
|-------|-----|------|
| `status` | str | "ok" \| "warning" \| "high_risk" \| "unknown" \| "error" |
| `warnings` | list[WarningItem] | 警告オブジェクトのリスト。警告なしの場合は空リスト |

---

## 4. Data Model

```python
from typing import TypedDict, Literal

class WarningItem(TypedDict):
    code: str       # 警告コード。例: "TERM_EXCEEDS_LEGAL_LIFE"
    severity: str   # "high" | "medium" | "low"
    message: str    # 日本語メッセージ
    source: str     # 根拠（法令名・マスタ名等）

class RuleCheckResult(TypedDict):
    status: str             # "ok" | "warning" | "high_risk" | "unknown" | "error"
    warnings: list[WarningItem]
```

### 警告コード一覧

| code | severity | 発生条件 |
|------|----------|---------|
| `TERM_EXCEEDS_LEGAL_LIFE` | high | リース期間（月数）> 法定耐用年数（月数） |
| `TERM_NEAR_LEGAL_LIFE` | medium | リース期間（月数）> 法定耐用年数（月数）× 0.9（かつ EXCEEDS未発生） |
| `TERM_EXCEEDS_EXPECTED_USAGE` | medium | リース期間（月数）> 期待使用期間（月数） |
| `INSURANCE_NOT_COVERED` | low | `insurance_applicable == "未付保"` |
| `RE_LEASE_INSURANCE_NOT_COVERED` | medium | `is_re_lease == True` かつ `re_lease_insurance == "未付保"` |

### status 決定ロジック

| 条件 | status |
|------|--------|
| warnings に severity=="high" が1件以上 | "high_risk" |
| warnings に severity=="medium" 以上が1件以上（high_riskでない場合） | "warning" |
| warnings に severity=="low" のみ | "warning" |
| warnings が空 かつ エラーなし | "ok" |
| asset_type が空文字（耐用年数チェック不可）かつ他に警告なし | "unknown" |
| `lease_term_months <= 0` | "error" |

### 法定耐用年数マスタ（モジュール内定数）

法人税法施行令別表第一に基づく代表値。単位：年。

```python
LEGAL_USEFUL_LIFE_YEARS: dict[str, int] = {
    # 電子機器・OA機器
    "電子計算機": 4,      # サーバ・PC・タブレット等
    "複写機": 5,
    "ファクシミリ": 5,
    "複合機": 5,
    "通信機器": 10,
    "カメラ": 5,
    # 工業機械
    "工作機械": 10,
    "印刷機械": 10,
    "農業機械": 7,
    "建設機械": 6,
    "フォークリフト": 3,
    # 輸送機器
    "自動車（普通）": 6,
    "自動車（小型）": 4,
    "トラック": 5,
    "バス": 5,
    # 医療機器
    "医療機器": 6,
    "歯科用機器": 7,
    # 設備機器
    "エアコン": 6,
    "冷凍・冷蔵機器": 6,
    "厨房機器": 8,
    "運搬機具": 5,
    "自動販売機": 5,
}
```

---

## 5. API / Interface

### 関数シグネチャ

```python
def check_lease_rules(
    lease_term_months: int,
    asset_type: str = "",
    is_re_lease: bool = False,
    insurance_applicable: str = "不明",
    re_lease_insurance: str = "不明",
) -> RuleCheckResult:
    """
    リースルールチェックを実行し、警告リストを返す。

    スコア計算には影響しない。例外を外部に伝播させない設計。

    Returns:
        RuleCheckResult: status と warnings を持つ dict
    """
```

---

## 6. Business Rules

**BR-101**: 法定耐用年数超過チェック
- 条件：`asset_type` がマスタに存在し、`lease_term_months > LEGAL_USEFUL_LIFE_YEARS[asset_type] * 12`
- 処理：warnings に `TERM_EXCEEDS_LEGAL_LIFE`（severity="high"）を追加
- 根拠：法人税法施行令別表第一（器具及び備品）。法定耐用年数を超えるリース期間は減価償却上の問題が生じる

**BR-102**: 法定耐用年数近接チェック
- 条件：BR-101 が非該当、かつ `lease_term_months > LEGAL_USEFUL_LIFE_YEARS[asset_type] * 12 * 0.9`
- 処理：warnings に `TERM_NEAR_LEGAL_LIFE`（severity="medium"）を追加
- 根拠：残存耐用年数が10%未満は再リース余地が限定的なため要注意

**BR-103**: 期待使用期間超過チェック
- 条件：`expected_usage_period.find_item_by_name(asset_type)` で期待使用期間が取得でき、`lease_term_months > expected_period_months`
- 処理：warnings に `TERM_EXCEEDS_EXPECTED_USAGE`（severity="medium"）を追加
- 根拠：プロジェクトの `expected_usage_period.py` および `期待使用期間.json` マスタに基づく社内基準

**BR-104**: 動産保険未付保チェック
- 条件：`insurance_applicable == "未付保"`
- 処理：warnings に `INSURANCE_NOT_COVERED`（severity="low"）を追加
- 根拠：動産総合保険への付保はリース会社の一般的推奨事項。未付保は物件滅失リスクを高める

**BR-105**: 再リース保険未付保チェック
- 条件：`is_re_lease == True` かつ `re_lease_insurance == "未付保"`
- 処理：warnings に `RE_LEASE_INSURANCE_NOT_COVERED`（severity="medium"）を追加
- 根拠：再リース期間中は新品時より物件劣化が進むため、保険継続の確認が必要

**BR-106**: マスタ不在フォールバック
- 条件：`asset_type` が `LEGAL_USEFUL_LIFE_YEARS` に存在しない場合
- 処理：法定耐用年数チェック（BR-101/BR-102）をスキップ
- 根拠：不明な物件種別に誤った警告を出さないため

**BR-107**: 期待使用期間マスタ不在フォールバック
- 条件：`find_item_by_name(asset_type)` が None を返す場合
- 処理：期待使用期間チェック（BR-103）をスキップ（エラーとしない）
- 根拠：マスタ整備中の種別に誤った警告を出さないため

**BR-108**: 不正入力ガード
- 条件：`lease_term_months <= 0`
- 処理：warnings=[]、status="error" を返す。例外は発生させない
- 根拠：ゼロ・負のリース期間は入力不正であり、警告より先にエラー扱いが適切

**BR-109**: 例外サイレント処理
- 条件：関数内部で予期しない例外が発生した場合
- 処理：`warnings=[]`、`status="unknown"` を返す。例外を外部に伝播させない
- 根拠：このモジュールの失敗でスコア計算全体を止めてはならない

---

## 7. UI / UX

本SPECでは UI は対象外。P1-003 で index.html の表示を定義する。

---

## 8. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| `lease_term_months <= 0` | `status="error"`, `warnings=[]` を返す | 例外なし |
| `asset_type` がマスタにない | 耐用年数チェックをスキップ | エラーではなく正常フロー |
| `find_item_by_name()` が None を返す | 期待使用期間チェックをスキップ | エラーではなく正常フロー |
| `asset_type` が空文字 | 全耐用年数チェックをスキップ | `status="unknown"`（他の警告がない場合） |
| 予期しない例外 | `status="unknown"`, `warnings=[]` を返す | ログ出力あり |

---

## 9. Acceptance Criteria

**AC-101**: 法定耐用年数超過で high_risk が返る
- Given: `LEGAL_USEFUL_LIFE_YEARS["電子計算機"] = 4` （= 48ヶ月）
- When: `check_lease_rules(lease_term_months=60, asset_type="電子計算機")` を呼ぶ
- Then: `status == "high_risk"` かつ `warnings` に `code=="TERM_EXCEEDS_LEGAL_LIFE"`, `severity=="high"` が1件含まれる

**AC-102**: 法定耐用年数内は ok が返る
- Given: `LEGAL_USEFUL_LIFE_YEARS["電子計算機"] = 4`
- When: `check_lease_rules(lease_term_months=36, asset_type="電子計算機")` を呼ぶ
- Then: `status == "ok"` かつ `warnings == []`

**AC-103**: 法定耐用年数の90%超（近接）で medium 警告が返る
- Given: `LEGAL_USEFUL_LIFE_YEARS["電子計算機"] = 4` （閾値: 48 × 0.9 = 43.2ヶ月）
- When: `check_lease_rules(lease_term_months=45, asset_type="電子計算機")` を呼ぶ
- Then: `warnings` に `code=="TERM_NEAR_LEGAL_LIFE"`, `severity=="medium"` が1件含まれ、`TERM_EXCEEDS_LEGAL_LIFE` は含まれない

**AC-104**: 期待使用期間超過で medium 警告が返る
- Given: `asset_type="電子計算機"`（`期待使用期間.json` の `max_years=4` = 48ヶ月）
- When: `check_lease_rules(lease_term_months=60, asset_type="電子計算機")` を呼ぶ
- Then: `warnings` に `code=="TERM_EXCEEDS_EXPECTED_USAGE"`, `severity=="medium"` が含まれる

**AC-105**: 動産保険未付保で low 警告が返る
- Given: 有効なリース期間と物件種別
- When: `check_lease_rules(lease_term_months=36, asset_type="建設機械", insurance_applicable="未付保")` を呼ぶ
- Then: `warnings` に `code=="INSURANCE_NOT_COVERED"`, `severity=="low"` が含まれる

**AC-106**: 保険付保済みで保険警告が出ない
- Given: 有効なリース期間と物件種別
- When: `check_lease_rules(lease_term_months=36, asset_type="建設機械", insurance_applicable="付保済")` を呼ぶ
- Then: `warnings` に `INSURANCE_NOT_COVERED` が含まれない

**AC-107**: 再リース予定かつ再リース保険未付保で medium 警告が返る
- Given: 有効なリース期間と物件種別
- When: `check_lease_rules(lease_term_months=36, asset_type="工作機械", is_re_lease=True, re_lease_insurance="未付保")` を呼ぶ
- Then: `warnings` に `code=="RE_LEASE_INSURANCE_NOT_COVERED"`, `severity=="medium"` が含まれる

**AC-108**: 再リース予定なし の場合、再リース保険チェックがスキップされる
- Given: `is_re_lease=False`
- When: `check_lease_rules(lease_term_months=36, asset_type="工作機械", is_re_lease=False, re_lease_insurance="未付保")` を呼ぶ
- Then: `warnings` に `RE_LEASE_INSURANCE_NOT_COVERED` が含まれない

**AC-109**: 不正リース期間でエラーが返る（境界値）
- Given: `lease_term_months=0`
- When: `check_lease_rules(lease_term_months=0, asset_type="電子計算機")` を呼ぶ
- Then: `status == "error"` かつ `warnings == []` かつ例外が発生しない

**AC-110**: 負のリース期間でエラーが返る
- Given: `lease_term_months=-12`
- When: `check_lease_rules(lease_term_months=-12, asset_type="電子計算機")` を呼ぶ
- Then: `status == "error"` かつ `warnings == []`

**AC-111**: マスタ不在の物件種別でエラーにならない
- Given: `asset_type="存在しない種別ABC"`（`LEGAL_USEFUL_LIFE_YEARS` にも `期待使用期間.json` にも存在しない）
- When: `check_lease_rules(lease_term_months=60, asset_type="存在しない種別ABC")` を呼ぶ
- Then: 例外が発生せず、`TERM_EXCEEDS_LEGAL_LIFE` / `TERM_EXCEEDS_EXPECTED_USAGE` は `warnings` に含まれない

**AC-112**: asset_type 空文字で unknown が返る
- Given: 保険もすべて "不明"
- When: `check_lease_rules(lease_term_months=60, asset_type="")` を呼ぶ
- Then: `status == "unknown"` かつ `warnings == []`

**AC-113**: 複数警告が同時に返る
- Given: 耐用年数超過 かつ 動産保険未付保
- When: `check_lease_rules(lease_term_months=72, asset_type="電子計算機", insurance_applicable="未付保")` を呼ぶ
- Then: `warnings` に `TERM_EXCEEDS_LEGAL_LIFE` と `INSURANCE_NOT_COVERED` の2件が含まれ、`status == "high_risk"`

**AC-114**: 全チェック通過で ok が返る
- Given: 短いリース期間・付保済み・再リース不要
- When: `check_lease_rules(lease_term_months=24, asset_type="工作機械", is_re_lease=False, insurance_applicable="付保済")` を呼ぶ
- Then: `status == "ok"` かつ `warnings == []`

**AC-115**: パフォーマンス要件（1件50ms以内）
- Given: 任意の有効入力
- When: `check_lease_rules()` を100回連続で呼ぶ
- Then: 合計処理時間が5000ms（= 100 × 50ms）以内

---

## 10. Non-Functional Requirements

- **パフォーマンス**: 1件あたり 50ms 以内（CPU処理のみ、DB参照なし）
- **後方互換性**: 既存APIレスポンスの既存フィールドは本モジュールでは変更しない（P1-002 で対応）
- **例外非伝播**: 関数内部でいかなる例外が発生しても、外部に伝播させない。`status="unknown"` を返す
- **ログ**: `warnings` が1件以上ある場合は `[lease_rule_checks]` プレフィックスで `asset_type`, `lease_term_months`, `warnings` コードをprint出力する
- **テストカバレッジ**: AC-101〜AC-115 全件カバー必須

---

## 11. Implementation Notes（Codex向け）

- **ファイル配置**: `mobile_app/lease_rule_checks.py` に作成（`mobile_app/api.py` と同ディレクトリ）
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `mobile_app/api.py`（本SPECでは）
- **期待使用期間の参照方法**:
  ```python
  import sys, os
  sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
  from expected_usage_period import find_item_by_name
  ```
  `find_item_by_name()` は `Optional[dict]` を返す。期待使用期間の上限は `item["max_years"] * 12`（月換算）。実際のキー名は `min_years`（最短期待年数）/ `max_years`（最長期待年数）であり、`expected_period_months` / `min_months` / `max_months` キーは存在しない。BR-103 の `expected_period_months` は `int(item["max_years"]) * 12` で算出する
- **法定耐用年数マスタ**: `mobile_app/lease_rule_checks.py` 冒頭の定数 `LEGAL_USEFUL_LIFE_YEARS` として定義。DBには格納しない
- **テストファイル**: `tests/spec_phase1/test_P1-001.py` に作成
- **インポートパス**: `mobile_app/` 内からのインポートは `from lease_rule_checks import check_lease_rules`

---

## 12. Test Plan

### 単体テスト（Codex が作成）
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_101 | AC-101 | 電子計算機60ヶ月 → high_risk, TERM_EXCEEDS_LEGAL_LIFE |
| test_102 | AC-102 | 電子計算機36ヶ月 → ok |
| test_103 | AC-103 | 電子計算機45ヶ月 → TERM_NEAR_LEGAL_LIFE |
| test_104 | AC-104 | 期待使用期間超過 → TERM_EXCEEDS_EXPECTED_USAGE |
| test_105 | AC-105 | 動産保険未付保 → INSURANCE_NOT_COVERED |
| test_106 | AC-106 | 保険付保済み → 保険警告なし |
| test_107 | AC-107 | 再リース予定+保険未付保 → RE_LEASE_INSURANCE_NOT_COVERED |
| test_108 | AC-108 | 再リース不要+保険未付保 → 再リース警告なし |
| test_109 | AC-109 | lease_term_months=0 → error, 例外なし |
| test_110 | AC-110 | lease_term_months=-12 → error |
| test_111 | AC-111 | 未知の物件種別 → 例外なし, 耐用年数警告なし |
| test_112 | AC-112 | asset_type="" → unknown |
| test_113 | AC-113 | 超過+保険未付保 → 2件の warnings, high_risk |
| test_114 | AC-114 | 全条件クリア → ok, warnings=[] |
| test_115 | AC-115 | 100回連続呼び出し → 5000ms以内 |

### 回帰テスト
- `mobile_app/api.py` の既存テストが全件 pass すること（本SPECでは api.py を変更しないため）
