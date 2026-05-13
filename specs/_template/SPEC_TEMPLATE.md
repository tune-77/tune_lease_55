---
spec_id: PX-NNN
phase: X
title: （機能名・モジュール名）
status: draft
author: Claude Opus
reviewer: ""
version: "1.0"
created: YYYY-MM-DD
updated: YYYY-MM-DD
depends_on: []
superseded_by: ""
---

# PX-NNN — （タイトル）

---

## 1. Goal

> （このSPECが達成する目的を1〜2文で。「何を」「なぜ」を明確に。）

---

## 2. Scope

### In scope
- （実装する機能・変更する範囲を箇条書き）

### Out of scope
- （このSPECでは実装しない事項。誤解を防ぐために明示する）

---

## 3. Inputs / Outputs

### Inputs
| 項目名 | 型 | 必須/任意 | 説明 |
|-------|-----|----------|------|
| （例: lease_term_months） | int | 必須 | リース期間（月数） |

### Outputs
| 項目名 | 型 | 説明 |
|-------|-----|------|
| （例: warnings） | list[dict] | 警告オブジェクトのリスト |

---

## 4. Data Model

```python
# 出力データ構造の例（TypedDict or dataclassで定義）
class WarningItem(TypedDict):
    code: str          # 例: "TERM_EXCEEDS_LEGAL_LIFE"
    severity: str      # "high" | "medium" | "low"
    message: str
    source: str        # 根拠（法令・マスタ名等）
```

---

## 5. API / Interface

### 関数シグネチャ（新規モジュールの場合）
```python
def check_lease_rules(
    lease_term_months: int,
    asset_name: str,
    is_re_lease: bool = False,
    insurance_applicable: str = "unknown",
) -> dict:
    """
    Returns:
        {
            "status": "ok" | "warning" | "high_risk",
            "warnings": [WarningItem, ...]
        }
    """
```

### APIレスポンス拡張（既存APIに追加する場合）
```json
{
  "（既存フィールド）": "...",
  "（追加フィールド）": {
    "説明": "ここに追加内容を記述"
  }
}
```

---

## 6. Business Rules

<!-- BR-xxx 番号は全SPEC通しでユニークにすること -->

**BR-001**: （ルール名）
- 条件：（いつ適用されるか）
- 処理：（何をするか）
- 根拠：（法令・社内規定・マスタ等）

**BR-002**: （ルール名）
- 条件：
- 処理：
- 根拠：

---

## 7. UI / UX（フロントエンド変更がある場合のみ）

- （変更する画面・コンポーネント）
- （表示条件・レイアウト・カラー等）

```
【表示例】
┌─────────────────────────────────────────────────┐
│ ⚠️ リース期間警告                                 │
│ リース期間が法定耐用年数に近く、再リース余地は      │
│ 限定的です。再リース前提の場合は保険対象を確認。    │
└─────────────────────────────────────────────────┘
```

---

## 8. Error Handling

| エラー条件 | 処理 | ユーザー向けメッセージ |
|-----------|------|---------------------|
| lease_term_months が 0 以下 | `status: "error"` を返す | 「リース期間が不正です」 |
| asset_name が空文字 | 耐用年数チェックをスキップ | （警告なし、ログに記録） |
| expected_usage_period.py がKeyError | `status: "unknown"` を返す | （表示なし） |

---

## 9. Acceptance Criteria

<!-- AC-xxx 番号は全SPEC通しでユニークにすること -->
<!-- Given-When-Then 形式で全て記述。Codexはこれを基にテストを作成する -->

**AC-001**: （テスト名）
- Given: （前提条件）
- When: （操作・入力）
- Then: （期待される結果）

**AC-002**: （テスト名）
- Given:
- When:
- Then:

**AC-003**: 境界値テスト
- Given: lease_term_months = 0
- When: `check_lease_rules(lease_term_months=0, asset_name="電子計算機")` を呼ぶ
- Then: `status: "error"` が返り、例外は発生しない

**AC-004**: マスタ不在テスト
- Given: expected_usage_period マスタに存在しない asset_name
- When: チェック関数を呼ぶ
- Then: エラーではなく `status: "unknown"` が返る

---

## 10. Non-Functional Requirements

- **パフォーマンス**: 1件あたり 50ms 以内（CPU処理のみ、DB参照なし）
- **後方互換性**: 既存APIレスポンスの既存フィールドは変更しない
- **ログ**: 警告が発生した場合は case_id・warnings・タイムスタンプをログに残す
- **テストカバレッジ**: AC-xxx 全件カバー必須

---

## 11. Implementation Notes（Codex向け）

> このセクションはCodexへの実装指示。設計判断の「なぜ」を書く。

- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`（スコア計算ロジック）
- **使うべき既存モジュール**: `expected_usage_period.py` の `get_expected_period()` 関数を再利用
- **新規ファイル配置**: `lease_rule_checks.py` はプロジェクトルート直下に配置
- **インポート方針**: `from expected_usage_period import get_expected_period` で参照
- **テストファイル**: `tests/test_lease_rule_checks.py` に作成

---

## 12. Test Plan

### 単体テスト（Codexが作成）
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_001 | AC-001 | （内容） |
| test_002 | AC-002 | （内容） |
| test_003 | AC-003 | 境界値: lease_term=0 |
| test_004 | AC-004 | マスタ不在フォールバック |

### 回帰テスト
- 既存スコアリングの出力値が変化しないこと（`total_scorer.py` の出力比較）

### 手動確認（実装後）
- [ ] Streamlit審査画面で警告バナーが表示される
- [ ] 警告なし案件でバナーが出ない
- [ ] APIレスポンスの既存フィールドが壊れていない
