---
spec_id: P3-003
phase: 3
title: index.html — ステルス競合圧力スコア参考表示
status: approved
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: [P3-001, P3-002]
superseded_by: ""
---

# P3-003 — index.html — ステルス競合圧力スコア参考表示

---

## 1. Goal

`mobile_app/index.html` に **ステルス競合圧力スコア参考パネル** を追加し、`POST /predict` レスポンスの `aurion.competitor_pressure` を審査官・営業担当に視覚的に提示する。「参考値であり審査スコアには影響しない」旨を明示し、誤解を防ぐ。P2-003 で追加した Q_risk パネルと同じ設計パターンを踏襲する。

---

## 2. Scope

### In scope
- `mobile_app/index.html` へのステルス競合圧力表示パネルの追加
- `aurion.competitor_pressure.level` に応じた配色（ok: 緑、caution: 黄色、high_risk: 赤）
- `aurion.competitor_pressure.pattern_details` の圧力パターンリスト表示
- JavaScript: `renderCompetitorPressure()` 関数の追加（`renderQRisk()` と独立）
- 参考値免責表示（「スコアには影響しません」）

### Out of scope
- `aurion/stealth_competitor.py` の変更（P3-001 で対応済み）
- `mobile_app/api.py` の変更（P3-002 で対応済み）
- Streamlit 側 UI の変更（`tune_lease_55.py` / `components/`）
- ステルス競合スコアによる自動判定ロジック・承認フローへの組み込み
- スコアの計算式・閾値の変更
- 新規フォームフィールドの追加（P3-003 でリクエスト変更なし）

---

## 3. Inputs / Outputs

### Inputs（JavaScript）

`POST /predict` レスポンスの `aurion.competitor_pressure` オブジェクト（P3-002 で追加）：

```json
{
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
```

### Outputs（HTML 表示）

ステルス競合圧力パネルを Q_risk パネルの直下に挿入する。

```
┌─────────────────────────────────────────────────────────────────┐
│  AURION ステルス競合推定                         ※ 参考値         │
│  スプレッド乖離シグナル                                            │
│                                                                 │
│  競合圧力スコア: 40  [caution - 要確認]                           │
│  ───────────────────────────────────────────────                │
│  [!] COMP-STEALTH-001 (high)                                    │
│      競合未申告ですが推奨スプレッドが1.5%未満です。                  │
│      ステルス競合の存在を確認してください。                          │
│      spread_pred: 1.2% / threshold: 1.5%                       │
│                                                                 │
│  ※ このスコアは審査スコアには影響しません                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Model

HTML/JavaScript の観点でのデータ構造は P3-002 で定義した `StealthCompetitorResult` 型に準拠。

---

## 5. API / Interface

### JavaScript 関数シグネチャ

```javascript
/**
 * aurion.competitor_pressure オブジェクトをパネルに描画する。
 * @param {Object} cp - aurion.competitor_pressure（score, level, patterns, pattern_details）
 */
function renderCompetitorPressure(cp) {
    // cp が null / undefined / cp.patterns.length == 0 の場合は非表示
    // P2-003 の renderQRisk() と同パターン
}
```

### 呼び出し箇所

`POST /predict` の成功ハンドラー内で、`renderQRisk()` 呼び出しの直後に追加する：

```javascript
// P2-003 からの既存呼び出し（変更なし）
renderQRisk(data.aurion?.q_risk);
// P3-003 で追加
renderCompetitorPressure(data.aurion?.competitor_pressure);
```

### パネルの HTML 要素 ID

| 要素 | ID | 説明 |
|-----|-----|------|
| パネルコンテナ | `competitor-pressure-panel` | `display: none` → 検知時に表示 |
| スコア表示 | `competitor-pressure-score` | `40` 等の整数値 |
| レベルバッジ | `competitor-pressure-level` | `"ok"` / `"caution"` / `"high_risk"` |
| パターンリスト | `competitor-pressure-patterns` | `<ul>` 要素 |

---

## 6. Business Rules

**BR-321**: パネル表示条件
- 条件：`aurion.competitor_pressure` が存在し、`patterns.length > 0`
- 処理：`competitor-pressure-panel` の `display` を `block` にする
- 逆条件（`patterns.length == 0`）：パネルは非表示のまま

**BR-322**: 配色ルール

| level | 背景色 | テキスト色 | 左ボーダー色 |
|-------|-------|----------|------------|
| `ok` | `#f0fdf4` | `#166534` | `#22c55e` |
| `caution` | `#fffbeb` | `#92400e` | `#f59e0b` |
| `high_risk` | `#fef2f2` | `#991b1b` | `#ef4444` |

**BR-323**: severity アイコンマッピング

| severity | アイコン |
|----------|---------|
| `high` | `⚠️` |
| `medium` | `🔶` |
| `low` | `ℹ️` |

**BR-324**: `values` の表示
- 条件：`pattern_details[i].values` が空でない場合
- 処理：`values` の各キー・バリューを小さいフォントで表示する
- 例：`spread_pred: 1.2% / threshold: 1.5%`

**BR-325**: フォールバック表示
- 条件：`aurion.competitor_pressure` が `undefined` または `null`（モジュール未ロード時のフォールバック）
- 処理：パネルを非表示のまま維持する。エラー表示はしない

**BR-326**: Q_risk パネルとの位置関係
- 条件：常に
- 処理：ステルス競合圧力パネルは Q_risk パネルの **直下** に配置する

---

## 7. UI / UX

### パネル表示例（caution 時）

```html
<div id="competitor-pressure-panel" style="display:none; margin-top:12px; padding:14px;
     border-left:4px solid #f59e0b; background:#fffbeb; border-radius:6px;">
  <div style="font-weight:bold; color:#92400e; margin-bottom:6px;">
    🔍 AURION ステルス競合推定
    <span style="font-size:0.75em; font-weight:normal; margin-left:8px;">※ 参考値</span>
  </div>
  <div>競合圧力スコア: <strong id="competitor-pressure-score">0</strong>
    &nbsp;<span id="competitor-pressure-level" style="font-size:0.85em;"></span>
  </div>
  <ul id="competitor-pressure-patterns" style="margin-top:8px; padding-left:18px;"></ul>
  <div style="font-size:0.75em; color:#6b7280; margin-top:8px;">
    ※ このスコアは審査スコアには影響しません
  </div>
</div>
```

### パネル非表示（ok かつ patterns 空の場合）

パネルコンテナは DOM に存在するが `display:none` を維持し、画面スペースを取らない。

---

## 8. Error Handling

| エラー条件 | 処理 |
|-----------|------|
| `data.aurion` が `undefined` | `renderCompetitorPressure(undefined)` が呼ばれてもエラーにならない（早期リターン） |
| `pattern_details` が空配列 | パターンリストに何も表示しない（パネル自体も非表示） |
| `values` オブジェクトが空 | values 行を表示しない（エラーにしない） |

---

## 9. Acceptance Criteria

**AC-901**: patterns が空のとき競合圧力パネルが非表示
- Given: api から `aurion.competitor_pressure = {score: 0, level: "ok", patterns: [], pattern_details: []}` が返る
- When: `renderCompetitorPressure()` を呼ぶ
- Then: `#competitor-pressure-panel` の `display` が `none` のまま

**AC-902**: caution 時にパネルが表示され黄色配色になる
- Given: `aurion.competitor_pressure.level = "caution"`、`patterns = ["COMP-STEALTH-001"]`
- When: `renderCompetitorPressure()` を呼ぶ
- Then: `#competitor-pressure-panel` が表示され、背景色が `#fffbeb`、ボーダー色が `#f59e0b`

**AC-903**: high_risk 時にパネルが表示され赤色配色になる
- Given: `aurion.competitor_pressure.level = "high_risk"`
- When: `renderCompetitorPressure()` を呼ぶ
- Then: `#competitor-pressure-panel` が表示され、背景色が `#fef2f2`、ボーダー色が `#ef4444`

**AC-904**: スコア値が `#competitor-pressure-score` に表示される
- Given: `aurion.competitor_pressure.score = 40`
- When: `renderCompetitorPressure()` を呼ぶ
- Then: `#competitor-pressure-score` のテキストが `"40"`

**AC-905**: パターンの message が `#competitor-pressure-patterns` にリスト表示される
- Given: `pattern_details[0].message = "競合未申告ですが推奨スプレッドが1.5%未満です。..."`, `severity = "high"`
- When: `renderCompetitorPressure()` を呼ぶ
- Then: `#competitor-pressure-patterns` の `<ul>` に `⚠️` アイコンとメッセージが含まれる、かつ `message` および `code` は `escapeHtml()` でサニタイズされた上で innerHTML に挿入される

**AC-906**: medium severity のパターンには `🔶` アイコンが表示される
- Given: `pattern_details[0].severity = "medium"`
- When: `renderCompetitorPressure()` を呼ぶ
- Then: 対応する `<li>` に `🔶` が含まれる

**AC-907**: values が存在する場合は値が表示される
- Given: `pattern_details[0].values = {"spread_pred": 1.2, "threshold": 1.5}`
- When: `renderCompetitorPressure()` を呼ぶ
- Then: `spread_pred: 1.2` と `threshold: 1.5` が対応する `<li>` 内に表示される

**AC-908**: `aurion.competitor_pressure` が undefined でも例外が発生しない
- Given: `data.aurion.competitor_pressure` が `undefined`（モジュール未ロード時）
- When: `renderCompetitorPressure(undefined)` を呼ぶ
- Then: JavaScript 例外が発生しない。パネルは非表示のまま

**AC-909**: `renderQRisk()` の既存動作が変化しない（回帰確認）
- Given: `aurion.q_risk` の正常レスポンス
- When: P3-003 変更後に `POST /predict` を呼ぶ
- Then: Q_risk パネル（`#q-risk-panel`）の表示・配色・パターン表示が P2-003 時点から変化しない

**AC-910**: ステルス競合圧力パネルは Q_risk パネルの直下に配置される
- Given: `index.html` に P3-003 変更が適用済み
- When: ページを表示する
- Then: DOM 上で `#competitor-pressure-panel` が `#q-risk-panel` の直後の兄弟要素として配置されている

---

## 10. Non-Functional Requirements

- **後方互換性**: `renderQRisk()` および既存 UI 要素を変更しない
- **テストカバレッジ**: AC-901〜AC-910 全件カバー必須（JavaScript 単体テスト + 手動確認）
- **アクセシビリティ**: パネルの色情報は必ずテキストでも提示する（`caution`, `high_risk` 等のラベルを表示）

---

## 11. Implementation Notes（Codex向け）

- **変更対象ファイル**: `mobile_app/index.html` のみ
- **パターン参照**: P2-003 の `renderQRisk()` 実装を参照し、同じ設計パターンで `renderCompetitorPressure()` を実装する
- **パネル追加位置**: Q_risk パネル（`id="q-risk-panel"`）の直後に `id="competitor-pressure-panel"` を追加する
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/q_risk.py`, `aurion/stealth_competitor.py`, `mobile_app/api.py`
- **テストファイル**: `tests/spec_phase3/test_P3-003.js` または `tests/spec_phase3/test_P3-003.py`（Playwright/Selenium 等）に作成

---

## 12. Test Plan

### 単体テスト（Codex が作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_901 | AC-901 | patterns=[] → パネル非表示 |
| test_902 | AC-902 | level=caution → 黄色配色で表示 |
| test_903 | AC-903 | level=high_risk → 赤配色で表示 |
| test_904 | AC-904 | score=40 → #competitor-pressure-score に "40" |
| test_905 | AC-905 | high severity → ⚠️ アイコン + メッセージ |
| test_906 | AC-906 | medium severity → 🔶 アイコン |
| test_907 | AC-907 | values が存在 → values 値が表示 |
| test_908 | AC-908 | undefined 入力 → 例外なし |
| test_909 | AC-909 | Q_risk パネル動作が変化しない（回帰） |
| test_910 | AC-910 | DOM 順: q-risk-panel → competitor-pressure-panel |

### 手動確認（実装後）
- [ ] 審査画面でステルス競合圧力パネルが caution/high_risk 時に表示される
- [ ] ok（パターンなし）のときパネルが表示されない
- [ ] Q_risk パネルとステルス競合圧力パネルが両方表示された場合にレイアウトが崩れない
- [ ] APIレスポンスの既存フィールドが壊れていない
