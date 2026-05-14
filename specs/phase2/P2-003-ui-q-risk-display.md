---
spec_id: P2-003
phase: 2
title: index.html — Q_risk 財務矛盾スコア参考表示
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.1"
created: 2026-05-14
updated: 2026-05-14
depends_on: [P2-001, P2-002]
superseded_by: ""
---

# P2-003 — index.html — Q_risk 財務矛盾スコア参考表示

---

## 1. Goal

`mobile_app/index.html` に **Q_risk 財務矛盾スコア参考パネル** を追加し、`POST /predict` レスポンスの `aurion.q_risk` を審査官に視覚的に提示する。「参考値であり審査スコアには影響しない」旨を明示し、誤解を防ぐ。

また、P2-001 の8種財務矛盾パターンのうち4種（FIN-CONTRADICT-003/005/007/008）が参照する **財務詳細フィールド4件**（`op_profit`・`bank_credit`・`machines`・`depreciation`）を財務情報カードに追加し、フロントエンドから API へ送信できるようにする。

---

## 2. Scope

### In scope
- `mobile_app/index.html` への Q_risk 表示パネルの追加
- `aurion.q_risk.level` に応じた配色（ok: 緑、caution: 黄色、high_risk: 赤）
- `aurion.q_risk.pattern_details` の矛盾リスト表示
- JavaScript: `renderQRisk()` 関数の追加（`renderWarnings()` と独立）
- 参考値免責表示（「スコアには影響しません」）
- 財務詳細フィールド4件の HTML 入力要素追加:
  - `op_profit`（営業利益・百万円）
  - `bank_credit`（銀行借入残高・百万円）
  - `machines`（機械設備残高・百万円）
  - `depreciation`（減価償却費・百万円）
- 4フィールドの `POST /predict` リクエストへの組み込み（`collectFormData()` 相当箇所）
- 4フィールドの `clearAll()` でのリセット

### Out of scope
- `aurion/q_risk.py` の変更（P2-001 で対応済み）
- `mobile_app/api.py` の変更（P2-002 で対応済み）
- Streamlit 側 UI の変更（`tune_lease_55.py` / `components/`）
- Q_risk スコアによる自動判定ロジック・承認フローへの組み込み
- スコアの計算式・閾値の変更

---

## 3. Inputs / Outputs

### Inputs（JavaScript）

`POST /predict` レスポンスの `aurion.q_risk` オブジェクト（P2-002 で追加）：

```json
{
  "score": 35,
  "level": "caution",
  "patterns": ["FIN-CONTRADICT-004"],
  "pattern_details": [
    {
      "code": "FIN-CONTRADICT-004",
      "severity": "medium",
      "message": "リース残高が年商の50%を超えています。過剰なオフバランス活用の可能性があります。",
      "values": { "lease_credit": 60.0, "nenshu": 100.0, "ratio": 0.6 }
    }
  ]
}
```

### Outputs（HTML表示）

Q_risk パネルをスコア表示エリアの直下に挿入する。

```
┌─────────────────────────────────────────────────────────────────┐
│  AURION Q_risk（財務矛盾スコア）               ※ 参考値         │
├─────────────────────────────────────────────────────────────────┤
│  スコア: 35 / 100       レベル: ⚠️ 要注意（caution）            │
├─────────────────────────────────────────────────────────────────┤
│  検知された矛盾パターン:                                         │
│  • [MEDIUM] FIN-CONTRADICT-004                                  │
│    リース残高が年商の50%を超えています。                          │
│    過剰なオフバランス活用の可能性があります。                      │
│    (lease_credit=60.0, nenshu=100.0, ratio=0.60)               │
├─────────────────────────────────────────────────────────────────┤
│  ℹ️ このスコアは参考値です。審査スコア・判定には影響しません。     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Model

本SPECに固有のデータモデルなし。P2-002 の `QRiskResult` を JavaScript で消費する。

---

## 5. API / Interface

### 追加 JavaScript 関数

```javascript
function renderQRisk(qRisk) {
    // qRisk: { score, level, patterns, pattern_details }
    // id="q-risk-panel" の要素を更新する
}
```

### 追加 HTML 要素

```html
<!-- スコア表示エリア直下に配置 -->
<div id="q-risk-panel" style="display:none; margin-top:16px;">
  <!-- renderQRisk() が動的に生成 -->
</div>
```

---

## 6. Business Rules

**BR-221**: Q_risk パネルは予測結果受信後のみ表示
- 条件：`POST /predict` のレスポンス受信後
- 処理：`id="q-risk-panel"` を `display:block` にして `renderQRisk()` を呼ぶ
- 根拠：初期状態では非表示。フォーム送信前に不要な要素を表示しない

**BR-222**: level に応じた配色
- 条件：`aurion.q_risk.level` の値
- 処理：
  - `"ok"` → 緑系ボーダー（`#28a745` に準ずる色）
  - `"caution"` → 黄色系ボーダー（P1-003 の警告バナーと同系色 `#ffc107`）
  - `"high_risk"` → 赤系ボーダー（`#dc3545` に準ずる色）
- 根拠：P1-003 の警告バナー配色と統一し、審査官の視認性を確保する

**BR-223**: 矛盾パターンが0件の場合の表示
- 条件：`aurion.q_risk.patterns.length === 0`
- 処理：「財務矛盾は検知されませんでした」と表示（パネルは表示する）
- 根拠：何も表示しないと「APIが動いていない」と誤解される

**BR-224**: 参考値免責表示の必須化
- 条件：常に表示
- 処理：パネル下部に「このスコアは参考値です。審査スコア・判定には影響しません。」を固定表示
- 根拠：審査官が Q_risk スコアを正式スコアと混同することを防ぐ

**BR-225**: pattern_details.values の小数表示
- 条件：`values` オブジェクト内の数値
- 処理：小数点以下2桁で表示（例: `0.60`, `100.00`）
- 根拠：比率（ratio）を読みやすく表示する

**BR-226**: aurion フィールド未存在時の非表示
- 条件：レスポンス JSON に `aurion` キーが存在しない（フォールバック時）
- 処理：`id="q-risk-panel"` を非表示のまま保持する。エラー表示しない
- 根拠：P2-001/P2-002 未デプロイ環境でも既存UIが壊れないようにする

**BR-227**: `op_profit`・`bank_credit`・`machines`・`depreciation` は数値のみ受け付ける
- 条件：4フィールドへの入力
- 処理：HTML `type="number"` かつ `inputmode="decimal"` を指定する
- 根拠：API は float を期待するため、非数値文字列を入力段階で排除する

**BR-228**: 4フィールド空白時は 0 として API へ送信する
- 条件：フォーム送信時にフィールドが空白（未入力）
- 処理：`parseFloat(...) || 0` により 0 に変換して `POST /predict` ペイロードに含める
- 根拠：P2-001 `detect_q_risk()` は各フィールドのデフォルト値を 0.0 と定義しており、省略時と同一の挙動を保証する

**BR-229**: 4フィールドに負の値が入力された場合はエラー表示する
- 条件：送信時に `op_profit < 0` または `bank_credit < 0` または `machines < 0` または `depreciation < 0`
- 処理：該当フィールド名と「マイナス不可」メッセージをエラー表示し、フォーム送信を中断する
- 根拠：財務矛盾パターンの計算上、これらフィールドは非負を前提としている（赤字は `net_income` / `ord_profit` で表現）

**BR-230**: 4フィールドは `clearAll()` でリセットされる
- 条件：「クリア」操作実行時
- 処理：既存フィールドと同様に 4フィールドを空文字列にリセットする
- 根拠：全フィールドを一括クリアする既存 UX を維持する

---

## 7. UI / UX

### レイアウト位置

```
[審査結果エリア]
  └─ スコア・判定表示
  └─ 推奨金利表示
  └─ [警告バナー] ← P1-003
  └─ [Q_risk パネル] ← P2-003 新規追加
```

### 配色・スタイル仕様

| level | ボーダー色 | 背景色 | ラベル文字 |
|-------|-----------|-------|-----------|
| ok | `#28a745` | `#f0fff4` | ✅ 異常なし（ok） |
| caution | `#ffc107` | `#fffdf0` | ⚠️ 要注意（caution） |
| high_risk | `#dc3545` | `#fff5f5` | 🔴 高リスク（high_risk） |

### severity バッジ

矛盾パターン行の先頭に severity バッジを表示：
- `high` → 赤背景白文字「HIGH」
- `medium` → 橙背景白文字「MEDIUM」
- `low` → 灰背景白文字「LOW」

### 表示例（caution）

```
┌── AURION Q_risk（財務矛盾スコア）─────────── ⚠️ 要注意 ─ ※ 参考値 ──┐
│  スコア: 35 / 100                                                    │
├────────────────────────────────────────────────────────────────────┤
│  検知された矛盾パターン（1件）:                                        │
│                                                                      │
│  [MEDIUM] FIN-CONTRADICT-004                                        │
│  リース残高が年商の50%を超えています。過剰なオフバランス活用の可能性。 │
│  lease_credit=60.00 / nenshu=100.00 / ratio=0.60                   │
├────────────────────────────────────────────────────────────────────┤
│  ℹ️ このスコアは参考値です。審査スコア・判定には影響しません。          │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. Error Handling

| エラー条件 | 処理 | 備考 |
|-----------|------|------|
| `response.aurion` が undefined | パネルを非表示のまま | エラーメッセージ不要 |
| `pattern_details` が null/undefined | 「矛盾は検知されませんでした」を表示 | |
| `score` が数値でない | `0` として扱う | |

---

## 9. Acceptance Criteria

**AC-601**: Q_risk パネルが POST /predict 後に表示される
- Given: ページ読み込み直後（初期状態）
- When: フォームを送信して `POST /predict` のレスポンスを受信する
- Then: `id="q-risk-panel"` が表示状態（display:block）になる

**AC-602**: level=="ok" 時に緑系スタイルが適用される
- Given: `aurion.q_risk.level == "ok"`
- When: `renderQRisk()` が呼ばれる
- Then: パネルのボーダー色が `#28a745` 系になり、ラベル「✅ 異常なし」が表示される

**AC-603**: level=="caution" 時に黄色系スタイルが適用される
- Given: `aurion.q_risk.level == "caution"`
- When: `renderQRisk()` が呼ばれる
- Then: パネルのボーダー色が `#ffc107` 系になり、ラベル「⚠️ 要注意」が表示される

**AC-604**: level=="high_risk" 時に赤系スタイルが適用される
- Given: `aurion.q_risk.level == "high_risk"`
- When: `renderQRisk()` が呼ばれる
- Then: パネルのボーダー色が `#dc3545` 系になり、ラベル「🔴 高リスク」が表示される

**AC-605**: 矛盾パターン0件時に「異常なし」メッセージが表示される
- Given: `aurion.q_risk.patterns == []`
- When: `renderQRisk()` が呼ばれる
- Then: 「財務矛盾は検知されませんでした」が表示される

**AC-606**: 矛盾パターン1件以上の場合にリストが表示される
- Given: `aurion.q_risk.pattern_details` に1件以上含まれる
- When: `renderQRisk()` が呼ばれる
- Then: 各 `pattern_details` の `code`, `message` が HTML に表示される

**AC-607**: severity バッジが正しく表示される
- Given: `FIN-CONTRADICT-004`（severity="medium"）を含むレスポンス
- When: `renderQRisk()` が呼ばれる
- Then: パターン行に「MEDIUM」バッジが表示される

**AC-608**: 参考値免責文が常に表示される
- Given: level に関わらず
- When: `renderQRisk()` が呼ばれる
- Then: 「このスコアは参考値です。審査スコア・判定には影響しません。」が表示される

**AC-609**: aurion フィールドなし時に既存UIが壊れない
- Given: レスポンス JSON に `aurion` キーが存在しない
- When: JavaScript が `renderQRisk()` を呼ばずにスキップする
- Then: 既存の score/judgment 表示・警告バナーに影響がない

**AC-610**: P1-003 の警告バナーと Q_risk パネルが共存して表示される
- Given: `warnings[]` に1件以上 かつ `aurion.q_risk.level == "caution"`
- When: フォームを送信する
- Then: 警告バナー（P1-003）と Q_risk パネル（P2-003）が両方表示される

**AC-611**: 4フィールドに有効な数値を入力すると API リクエストに含まれる
- Given: `op_profit=30`, `bank_credit=50`, `machines=20`, `depreciation=5` を入力
- When: フォームを送信する
- Then: `POST /predict` のリクエストボディに `op_profit: 30`, `bank_credit: 50`, `machines: 20`, `depreciation: 5` が含まれる

**AC-612**: 4フィールドが空白の場合は 0 として送信される
- Given: `op_profit`・`bank_credit`・`machines`・`depreciation` を未入力のまま
- When: フォームを送信する
- Then: `POST /predict` のリクエストボディに `op_profit: 0`, `bank_credit: 0`, `machines: 0`, `depreciation: 0` が含まれる

**AC-613**: 4フィールドにマイナス値を入力するとエラー表示されフォームが送信されない
- Given: `bank_credit` に `-10` を入力
- When: フォームを送信しようとする
- Then: 「マイナス不可」エラーが表示され、`POST /predict` は送信されない

**AC-614**: `clearAll()` 実行後に4フィールドが空になる
- Given: `op_profit=30`, `bank_credit=50`, `machines=20`, `depreciation=5` を入力済み
- When: クリアボタンを押す
- Then: 4フィールドの値がすべて空文字列になる

---

## 10. Non-Functional Requirements

- **後方互換性**: `aurion` フィールドが存在しない（P2-002 未デプロイ）環境で既存 UI が壊れないこと
- **JavaScript エラー非伝播**: `renderQRisk()` 内で例外が発生しても `try/catch` で隠蔽し、他の表示処理を止めない
- **CSS 外部依存なし**: 既存の inline style / style タグのみ使用。外部 CSS ライブラリを追加しない
- **テストカバレッジ**: AC-601〜AC-610 全件カバー必須

---

## 11. Implementation Notes（Codex向け）

- **変更対象ファイル**: `mobile_app/index.html` のみ
- **触れてはいけないファイル**: `mobile_app/api.py`, `mobile_app/lease_rule_checks.py`, `mobile_app/aurion/q_risk.py`（本SPECでは）
- **既存コードとの統合**:
  - `renderWarnings()` と同じパターンで `renderQRisk()` を実装する
  - `POST /predict` 成功時の `.then()` 内で `renderWarnings(data.warnings)` の後に `renderQRisk(data.aurion?.q_risk)` を呼ぶ
  - オプショナルチェーン `?.` を使用して `aurion` が undefined でも安全に参照する
- **HTML 挿入位置**: P1-003 の警告バナー要素（`id="warnings-section"`）の直下
- **スタイル**: 既存 `<style>` タグ内に追記。新規 `<style>` タグを追加しない
- **テストファイル**: `tests/spec_phase2/test_P2-003.py`（JavaScript DOM テストが困難な場合は手動確認項目として記録）
- **4フィールドの配置**: 財務情報カード内、既存の `net_income`（当期純利益）と `dep_expense`（支払リース料）フィールドの間またはその後ろに配置する。既存フィールドとの重複なし（`dep_expense` は支払リース料であり `depreciation`（減価償却費）とは別フィールド）
- **4フィールドの API 送信**: `parseFloat(document.getElementById("op_profit").value) || 0` パターンで既存フィールドと同様に `collectFormData()` 相当箇所に追加する
- **4フィールドの clearAll() リセット**: 既存の `["nenshu", "gross_profit", ...]` 配列に `"op_profit"`, `"bank_credit"`, `"machines"`, `"depreciation"` を追加する

---

## 12. Test Plan

### 単体テスト（Codex が作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_601 | AC-601 | /predict レスポンス後にパネルが display:block になる |
| test_602 | AC-602 | level=ok → 緑スタイル・「✅ 異常なし」ラベル |
| test_603 | AC-603 | level=caution → 黄色スタイル・「⚠️ 要注意」ラベル |
| test_604 | AC-604 | level=high_risk → 赤スタイル・「🔴 高リスク」ラベル |
| test_605 | AC-605 | patterns=[] → 「財務矛盾は検知されませんでした」表示 |
| test_606 | AC-606 | pattern_details 1件 → code と message が HTML に含まれる |
| test_607 | AC-607 | severity=medium → 「MEDIUM」バッジが含まれる |
| test_608 | AC-608 | 参考値免責文が常に表示される |
| test_609 | AC-609 | aurion なしレスポンス → 既存 UI に影響なし |
| test_610 | AC-610 | 警告バナーと Q_risk パネルが共存する |
| test_611 | AC-611 | 4フィールドに数値入力 → APIリクエストに含まれる |
| test_612 | AC-612 | 4フィールド空白 → 0 として送信される |
| test_613 | AC-613 | マイナス値入力 → エラー表示・送信中断 |
| test_614 | AC-614 | clearAll() → 4フィールドが空になる |

### 手動確認（実装後）

- [ ] `level == "ok"` 時に緑パネルが表示される
- [ ] `level == "caution"` 時に黄色パネルと矛盾リストが表示される
- [ ] `level == "high_risk"` 時に赤パネルが表示される
- [ ] パターンが0件時に「異常なし」メッセージが表示される
- [ ] 参考値免責文が全レベルで表示される
- [ ] P1-003 の警告バナーと同時表示しても崩れない
- [ ] `aurion` フィールドなしのレスポンス（旧API）でページが壊れない
- [ ] `op_profit`・`bank_credit`・`machines`・`depreciation` の入力欄が財務情報カード内に表示される
- [ ] 4フィールドを入力して送信すると APIリクエストに値が含まれる
- [ ] 4フィールドを空白で送信すると API に 0 が送られる
- [ ] 4フィールドにマイナス値を入力するとエラーが表示され送信が止まる
- [ ] クリアボタンで4フィールドがリセットされる
