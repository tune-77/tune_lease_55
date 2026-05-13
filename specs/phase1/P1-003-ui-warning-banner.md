---
spec_id: P1-003
phase: 1
title: index.html — リースルール警告バナーUI
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-14
updated: 2026-05-14
depends_on: [P1-001, P1-002]
superseded_by: ""
---

# P1-003 — index.html — リースルール警告バナーUI

---

## 1. Goal

`mobile_app/index.html` に2つの変更を加える。①リースルール入力用オプションフォームカード（物件種別・保険情報）を追加し、APIリクエストに含める。② `POST /predict` レスポンスの `warnings[]` を黄色い警告バナーとして結果カード内に表示する。スコアバッジ（承認/条件付/否認）の表示ロジックは変更しない。

---

## 2. Scope

### In scope
- `mobile_app/index.html` への変更のみ
  - 入力フォームに「物件・保険情報（任意）」カードを追加（4フィールド）
  - 結果カードに警告バナーセクション（`#warnings-section`）を追加
  - `showResult()` 関数に `warnings` / `rule_check_status` の処理を追加
  - `runPredict()` 関数のリクエストボディに新規フィールドを追加
  - `clearAll()` 関数に新規フィールドのリセット処理を追加
  - 警告バナー用 CSS を追加

### Out of scope
- `mobile_app/api.py` の変更（P1-002 で対応）
- `mobile_app/lease_rule_checks.py` の変更（P1-001 で対応）
- 警告内容のローカル計算（JSでのルール実装）— APIから受け取るだけ
- Streamlit 画面（`tune_lease_55.py` 等）への警告表示
- 警告の永続化・保存

---

## 3. Inputs / Outputs

### 新規フォームフィールド（HTMLフォームへの追加）

| フィールドID | 要素型 | デフォルト | 説明 |
|------------|--------|----------|------|
| `asset_type` | `<select>` | `""`（選択なし） | 物件種別。LEGAL_USEFUL_LIFE_YEARS のキー名に対応するオプション＋「その他・不明」 |
| `is_re_lease` | `<input type="checkbox">` | 未チェック（false） | 再リース予定フラグ |
| `insurance_applicable` | `<select>` | `"不明"` | 動産保険付保状況 |
| `re_lease_insurance` | `<select>` | `"不明"` | 再リース保険付保状況。`is_re_lease` チェック時のみ有効（視覚的に有効/無効切り替え） |

### APIリクエスト追加フィールド

```js
{
  // 既存フィールド（変更なし）
  ...既存フィールド...,
  // 新規フィールド
  "asset_type":           <string>,   // 空文字またはselect値
  "is_re_lease":          <boolean>,  // チェックボックス値
  "insurance_applicable": <string>,   // "付保済" | "未付保" | "不明"
  "re_lease_insurance":   <string>,   // "付保済" | "未付保" | "不明"
}
```

---

## 4. Data Model

### 警告バナーアイテム（APIレスポンスから受け取る）

```js
{
  code: string,      // 例: "TERM_EXCEEDS_LEGAL_LIFE"
  severity: string,  // "high" | "medium" | "low"
  message: string,   // 日本語メッセージ
  source: string,    // 根拠文字列
}
```

### severity と表示スタイルの対応

| severity | アイコン | 背景色 | ボーダー色 | テキスト色 |
|---------|---------|--------|----------|---------|
| `"high"` | ⚠️ | `#fef3c7` | `#f59e0b` | `#78350f` |
| `"medium"` | ⚠️ | `#fef9c3` | `#d97706` | `#78350f` |
| `"low"` | ℹ️ | `#eff6ff` | `#93c5fd` | `#1e40af` |

---

## 5. API / Interface

### showResult 関数シグネチャ（変更後）

```js
function showResult({ score, probability, judgment,
                      recommended_rate, base_rate, spread_pred, rate_range,
                      warnings, rule_check_status }) {
  // ... 既存処理 ...
  renderWarnings(warnings || [], rule_check_status || "ok");
}
```

### renderWarnings 関数（新規）

```js
function renderWarnings(warnings, rule_check_status) {
  const section = document.getElementById("warnings-section");
  if (!warnings || warnings.length === 0) {
    section.style.display = "none";
    return;
  }
  section.style.display = "block";
  const list = document.getElementById("warnings-list");
  list.innerHTML = warnings.map(w => `
    <div class="warning-item warning-${w.severity}">
      <span class="warning-icon">${w.severity === "low" ? "ℹ️" : "⚠️"}</span>
      <div class="warning-body">
        <div class="warning-message">${escapeHtml(w.message)}</div>
        <div class="warning-source">${escapeHtml(w.source)}</div>
      </div>
    </div>
  `).join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
```

---

## 6. Business Rules

**BR-121**: 警告バナーは warnings[] が空の場合は非表示
- 条件：APIレスポンスの `warnings` が空配列 `[]` または undefined の場合
- 処理：`#warnings-section` の `display: none` を維持する
- 根拠：不要なUIノイズを避ける

**BR-122**: 警告バナーは results カードが非表示の間は表示しない
- 条件：審査未実行時または clearAll() 後
- 処理：`#warnings-section` を非表示にする
- 根拠：古い警告情報が表示されたままにならないようにする

**BR-123**: スコア・判定バッジは warnings の有無に関係なく変化しない
- 条件：warnings[] に何が入っていても
- 処理：`#score-val`, `#judgment-badge`, `#progress-bar` の値を warnings で変更しない
- 根拠：P1-002 BR-111「warnings はスコアに影響しない」をUIでも一貫させる

**BR-124**: XSS 対策：message と source フィールドをエスケープする
- 条件：warning.message / warning.source を innerHTML に挿入する前
- 処理：`escapeHtml()` を通して `<`, `>`, `&`, `"` をエンティティ変換する
- 根拠：APIレスポンスの文字列を直接 innerHTML に入れることによる XSS を防止する

**BR-125**: is_re_lease チェックオフ時は re_lease_insurance select を disabled にする
- 条件：`#is_re_lease` チェックボックスが未チェックの場合
- 処理：`#re_lease_insurance` select に `disabled` 属性を付与し、見た目をグレーアウトする
- 根拠：再リース予定がない場合、再リース保険の選択は無意味。誤入力防止のためのUX

**BR-126**: clearAll() は警告バナーも消去する
- 条件：`clearAll()` が呼ばれた場合
- 処理：`#warnings-section` を非表示にし、`#warnings-list` の innerHTML を空にする
- 根拠：フォームリセット後に古い警告が残ることを防ぐ

---

## 7. UI / UX

### 新規フォームカード「物件・保険情報（任意）」

入力フォームカード（財務情報カード）の直後、ボタン行の直前に配置する。

```
┌─────────────────────────────────────────────┐
│ 物件・保険情報（任意）                         │
│                                             │
│ 🏗 物件種別                                  │
│ [電子計算機              ▼]                   │
│                                             │
│ 🔄 再リース予定    [ ] チェックで有効           │
│                                             │
│ 🛡 動産保険                                  │
│ [不明                    ▼]                   │
│                                             │
│ 🔄 再リース保険（再リース予定時のみ）            │
│ [不明                    ▼] ← disabled時グレー │
└─────────────────────────────────────────────┘
```

### 警告バナーセクション

`#result` カード内の `#rate-section` の直後、免責文（`.disclaimer`）の直後に配置する。

```
┌─────────────────────────────────────────────┐
│ ⚠️ リースルール警告                          │
│                                             │
│  ⚠️ リース期間が法定耐用年数（4年）を超えています│
│     （リース期間: 6年）。減価償却上の問題が    │
│     生じる可能性があります。                  │
│     [法人税法施行令別表第一]                  │
│                                             │
│  ℹ️ 動産保険が未付保です。物件滅失リスクのため │
│     付保を推奨します。                       │
│     [リース会社内規]                         │
└─────────────────────────────────────────────┘
```

### CSS 追加仕様

```css
/* 物件・保険情報カード */
.card-optional .card-title {
  color: var(--muted);
}
.card-optional .card-title::after {
  content: " （任意）";
  font-weight: 400;
}

/* チェックボックス行 */
.field-checkbox {
  display: flex;
  align-items: center;
  gap: 10px;
}
.field-checkbox input[type="checkbox"] {
  width: 20px;
  height: 20px;
  cursor: pointer;
}

/* 警告バナー全体 */
#warnings-section {
  display: none;
  margin-top: 16px;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #f59e0b;
}
.warnings-header {
  background: #fef3c7;
  padding: 10px 14px;
  font-size: 13px;
  font-weight: 700;
  color: #78350f;
  border-bottom: 1px solid #f59e0b;
}

/* 警告アイテム */
.warning-item {
  display: flex;
  gap: 10px;
  padding: 12px 14px;
  border-bottom: 1px solid #fde68a;
}
.warning-item:last-child { border-bottom: none; }

.warning-high, .warning-medium {
  background: #fef9c3;
}
.warning-low {
  background: #eff6ff;
  border-bottom-color: #bfdbfe;
}
.warning-icon { font-size: 18px; flex-shrink: 0; }
.warning-body { flex: 1; }
.warning-message {
  font-size: 13px;
  font-weight: 600;
  color: #78350f;
  line-height: 1.5;
}
.warning-low .warning-message { color: #1e40af; }
.warning-source {
  font-size: 11px;
  color: var(--muted);
  margin-top: 3px;
}
```

---

## 8. Error Handling

| エラー条件 | 処理 |
|-----------|------|
| APIレスポンスに `warnings` キーがない（旧クライアント互換） | `warnings \|\| []` で空配列とし、バナーを非表示 |
| `warnings` が null の場合 | `warnings \|\| []` で空配列とし、バナーを非表示 |
| `message` / `source` に特殊文字が含まれる場合 | `escapeHtml()` でエスケープして XSS を防止 |
| `severity` が想定外の値の場合 | `.warning-${severity}` クラスが存在しないだけで、表示は崩れない |

---

## 9. Acceptance Criteria

**AC-301**: 警告ありの場合に黄色バナーが表示される
- Given: `POST /predict` が `warnings` に1件以上、`rule_check_status="high_risk"` を返す
- When: 審査実行ボタンを押し、結果が表示される
- Then: `#warnings-section` が表示され（`display: block`）、警告アイテムが1件以上表示される

**AC-302**: 警告なしの場合にバナーが非表示になる
- Given: `POST /predict` が `warnings=[]`, `rule_check_status="ok"` を返す
- When: 審査実行ボタンを押し、結果が表示される
- Then: `#warnings-section` が非表示（`display: none`）のまま

**AC-303**: clearAll() で警告バナーが消える
- Given: 警告バナーが表示されている状態
- When: 「データ抹消」ボタンを押す
- Then: `#warnings-section` が非表示になり、`#warnings-list` が空になる

**AC-304**: スコアバッジが警告の有無で変わらない
- Given: 同一の財務情報で `asset_type` の有無のみが異なる2回のリクエスト
- When: それぞれ審査実行
- Then: `#judgment-badge` のテキストと CSS クラスが変化しない

**AC-305**: 物件種別 select が存在する
- Given: ページが読み込まれた状態
- When: 「物件・保険情報（任意）」カードを確認する
- Then: `#asset_type` select 要素が存在し、「その他・不明」を含む10件以上のオプションが存在する

**AC-306**: is_re_lease チェック OFF で re_lease_insurance が disabled
- Given: ページが読み込まれた初期状態（`#is_re_lease` が未チェック）
- When: `#re_lease_insurance` select を確認する
- Then: `#re_lease_insurance` が `disabled` 属性を持ち、グレー表示される

**AC-307**: is_re_lease チェック ON で re_lease_insurance が有効
- Given: ページが読み込まれた初期状態
- When: `#is_re_lease` チェックボックスをチェックする
- Then: `#re_lease_insurance` select から `disabled` 属性が除去され、選択可能になる

**AC-308**: APIリクエストに新規フィールドが含まれる
- Given: `asset_type="電子計算機"`, `insurance_applicable="未付保"` を選択した状態
- When: 審査実行ボタンを押す
- Then: `fetch()` に渡すボディに `asset_type`, `is_re_lease`, `insurance_applicable`, `re_lease_insurance` が含まれる（DevTools Network で確認）

**AC-309**: XSS 対策 — message に HTML タグが含まれても無害化される
- Given: APIが `message: "<script>alert(1)</script>"` を返す状態（モック）
- When: 審査結果が表示される
- Then: スクリプトが実行されず、`&lt;script&gt;` としてテキスト表示される

**AC-310**: warning severity="low" が青系スタイルで表示される
- Given: `severity="low"` の warning が含まれるレスポンス
- When: 結果表示後に警告バナーを確認する
- Then: `.warning-low` クラスが適用され、背景が `#eff6ff`（青系）で表示される

---

## 10. Non-Functional Requirements

- **後方互換性**: `warnings` フィールドが APIレスポンスにない場合でも UI が壊れない
- **モバイル対応**: 警告バナーは最大幅 480px のラッパー内に収まり、テキスト折り返しが発生しても読みやすい
- **アクセシビリティ**: 警告バナーはスクリーンリーダー向けに `role="alert"` を付与する

---

## 11. Implementation Notes（Codex向け）

- **変更ファイル**: `mobile_app/index.html` のみ
- **触れてはいけないロジック**: `showResult()` 内のスコア・バッジ・プログレスバー・金利セクションの既存ロジック
- **物件種別 select のオプション値**: `LEGAL_USEFUL_LIFE_YEARS` のキーと完全一致させること。「その他・不明」オプションの value は `""` とし、これが送信された場合に API 側でチェックをスキップする（P1-001 BR-106）
- **asset_type の select オプション（value 順不同で全件）**:
  ```html
  <option value="">その他・不明</option>
  <option value="電子計算機">電子計算機（PC・サーバ）</option>
  <option value="複写機">複写機</option>
  <option value="ファクシミリ">ファクシミリ</option>
  <option value="複合機">複合機</option>
  <option value="通信機器">通信機器</option>
  <option value="工作機械">工作機械</option>
  <option value="印刷機械">印刷機械</option>
  <option value="農業機械">農業機械</option>
  <option value="建設機械">建設機械</option>
  <option value="フォークリフト">フォークリフト</option>
  <option value="自動車（普通）">自動車（普通）</option>
  <option value="自動車（小型）">自動車（小型）</option>
  <option value="トラック">トラック</option>
  <option value="バス">バス</option>
  <option value="医療機器">医療機器</option>
  <option value="歯科用機器">歯科用機器</option>
  <option value="エアコン">エアコン</option>
  <option value="冷凍・冷蔵機器">冷凍・冷蔵機器</option>
  <option value="厨房機器">厨房機器</option>
  <option value="運搬機具">運搬機具</option>
  <option value="自動販売機">自動販売機</option>
  <option value="カメラ">カメラ</option>
  ```
- **#warnings-section の配置**: 結果カード（`#result`）内の `<p class="disclaimer">` の直後（`#rate-section` より後）に配置する
- **re_lease_insurance の disabled 切り替え**: チェックボックスの `change` イベントで切り替える
  ```js
  document.getElementById("is_re_lease").addEventListener("change", function() {
    document.getElementById("re_lease_insurance").disabled = !this.checked;
  });
  ```
- **runPredict のリクエストボディ追加箇所**: `const vals = { ... }` ブロックに以下を追記
  ```js
  asset_type:           document.getElementById("asset_type").value,
  is_re_lease:          document.getElementById("is_re_lease").checked,
  insurance_applicable: document.getElementById("insurance_applicable").value,
  re_lease_insurance:   document.getElementById("re_lease_insurance").value,
  ```
- **テストファイル**: `tests/spec_phase1/test_P1-003.html`（Playwright または手動確認）

---

## 12. Test Plan

### 自動テスト（Playwright 推奨 / 手動でも可）
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_301 | AC-301 | モックAPIで warnings あり → バナー表示 |
| test_302 | AC-302 | モックAPIで warnings=[] → バナー非表示 |
| test_303 | AC-303 | clearAll() → バナー消去確認 |
| test_304 | AC-304 | asset_type有無でスコアバッジ不変 |
| test_305 | AC-305 | asset_type select のオプション数 >= 10 |
| test_306 | AC-306 | 初期状態: re_lease_insurance が disabled |
| test_307 | AC-307 | is_re_lease チェック → re_lease_insurance が有効 |
| test_308 | AC-308 | リクエストボディに4フィールドが含まれる |
| test_309 | AC-309 | XSS: message にスクリプトタグ → エスケープ確認 |
| test_310 | AC-310 | severity=low → blue 背景クラス確認 |

### 手動確認（実装後）
- [ ] スマホ（375px幅）で警告バナーのテキストが折り返して読める
- [ ] 「その他・不明」選択時に警告バナーが出ない（チェックスキップ）
- [ ] 保険フィールドを変えても既存スコア結果が変わらない
- [ ] clearAll() 後にバナーが消えること
