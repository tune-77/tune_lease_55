---
spec_id: P4-002
phase: 4
title: EDINET データ自動収集 — 法人財務データ取得モジュール
status: implemented
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-05-15
updated: 2026-05-15
depends_on: [P4-001]
superseded_by: ""
---

# P4-002 — EDINET データ自動収集 — 法人財務データ取得モジュール

---

## 1. Goal

法人番号をキーに EDINET API（金融庁）から有価証券報告書の財務データを自動取得し、審査入力フォームへの手入力を削減する。EDINET から取得できない場合は graceful degradation（手入力フォールバック）で審査フローを停止しない設計とする。

---

## 2. Scope

### In scope
- `edinet_collector.py` 新規作成（プロジェクトルート直下）
- EDINET API v2（`https://disclosure.edinet-api.go.jp/api/v2/`）を使った書類一覧取得・財務データ抽出
- 法人番号 → EDINET 企業コード（`edinetCode`）の解決（EDINET 企業情報 API 利用）
- 取得対象財務項目: 売上高（`nenshu`）、営業利益、当期純利益、総資産、自己資本比率
- Streamlit 審査フォームへの自動入力連携（`components/` 内のフォームコンポーネント）
- EDINET データのローカルキャッシュ（SQLite `edinet_cache` テーブル、TTL 24時間）
- `tests/spec_phase4/test_P4-002.py` のテスト作成（Codex担当）

### Out of scope
- XBRL パーサーの自前実装（`python-xbrl` ライブラリを利用）
- EDINET API のレート制限管理（簡易 sleep で対応、専用キューイングは対象外）
- 連結財務諸表と単体財務諸表の自動選択（連結優先で固定）
- 複数期分の財務データ取得（直近1期のみ対象）
- 既存スコアリングロジックの変更
- 法人番号の自動補完（入力支援は対象外、手入力された法人番号を利用）

---

## 3. Inputs / Outputs

### Inputs（`fetch_edinet_financials()` 引数）

| 項目名 | 型 | 必須/任意 | 説明 | 備考 |
|-------|-----|----------|------|------|
| `corporate_number` | str | 必須 | 法人番号（13桁数字） | チェックデジット検証あり |
| `fiscal_year` | int | 任意 | 取得対象の事業年度（例: 2024）| 省略時は直近完了年度を自動選択 |
| `use_cache` | bool | 任意（デフォルト `True`） | キャッシュを利用するか | |
| `api_key` | str | 任意 | EDINET API キー（環境変数 `EDINET_API_KEY` が優先） | |

### Outputs（`fetch_edinet_financials()` 戻り値）

| 項目名 | 型 | 説明 |
|-------|-----|------|
| `success` | bool | 取得成功フラグ |
| `source` | str | `"edinet"` / `"cache"` / `"fallback"` |
| `nenshu` | float \| None | 売上高（百万円） |
| `operating_profit` | float \| None | 営業利益（百万円） |
| `net_income` | float \| None | 当期純利益（百万円） |
| `total_assets` | float \| None | 総資産（百万円） |
| `equity_ratio` | float \| None | 自己資本比率（%） |
| `fiscal_year_retrieved` | int \| None | 取得できた事業年度 |
| `edinet_code` | str \| None | EDINET 企業コード |
| `error` | str \| None | エラーメッセージ（成功時は `None`） |

---

## 4. Data Model

### edinet_cache テーブル DDL

```sql
CREATE TABLE IF NOT EXISTS edinet_cache (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    corporate_number    TEXT    NOT NULL,
    fiscal_year         INTEGER NOT NULL,
    edinet_code         TEXT,
    nenshu              REAL,
    operating_profit    REAL,
    net_income          REAL,
    total_assets        REAL,
    equity_ratio        REAL,
    raw_json            TEXT,   -- EDINET APIレスポンスの生JSON（デバッグ用）
    fetched_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(corporate_number, fiscal_year)
);
```

### Python TypedDict

```python
from typing import TypedDict, Optional

class EdinetFinancialsResult(TypedDict):
    success: bool
    source: str                          # "edinet" | "cache" | "fallback"
    nenshu: Optional[float]              # 売上高（百万円）
    operating_profit: Optional[float]
    net_income: Optional[float]
    total_assets: Optional[float]
    equity_ratio: Optional[float]
    fiscal_year_retrieved: Optional[int]
    edinet_code: Optional[str]
    error: Optional[str]
```

---

## 5. API / Interface

### 主要関数シグネチャ

```python
def fetch_edinet_financials(
    corporate_number: str,
    fiscal_year: int | None = None,
    use_cache: bool = True,
    api_key: str | None = None,
    db_path: str = "data/lease_data.db",
) -> EdinetFinancialsResult:
    """
    法人番号をキーに EDINET API から財務データを取得する。

    取得できない場合は source="fallback" で全財務項目 None を返す。
    例外を外部に伝播させない。
    """

def resolve_edinet_code(
    corporate_number: str,
    api_key: str | None = None,
) -> str | None:
    """
    法人番号から EDINET 企業コードを解決する。
    見つからない場合は None を返す。
    """
```

### EDINET API エンドポイント（利用するもの）

| エンドポイント | 用途 |
|--------------|------|
| `GET /api/v2/companies.json?type=2` | 企業情報一覧（法人番号 → edinetCode 解決） |
| `GET /api/v2/documents.json?date={date}&type=2` | 書類一覧（有価証券報告書の docID 取得） |
| `GET /api/v2/documents/{docID}?type=5` | XBRL 書類ダウンロード |

---

## 6. Business Rules

**BR-411**: 法人番号フォーマット検証
- 条件：`corporate_number` が13桁の数字でない場合
- 処理：`success=False, error="invalid corporate_number format", source="fallback"` を返す
- 根拠：不正な法人番号で API を叩くと無駄なリクエストが発生する

**BR-412**: キャッシュ優先取得（TTL 24時間）
- 条件：`use_cache=True` かつ `edinet_cache` テーブルに `fetched_at` が現在時刻から24時間以内のレコードが存在する
- 処理：API を叩かずにキャッシュレコードを返す（`source="cache"`）
- 根拠：EDINET API は1日単位での更新のため、24時間 TTL で十分

**BR-413**: EDINET 取得失敗時の graceful degradation
- 条件：EDINET API が応答しない（タイムアウト）/ HTTP エラー / XBRL パースエラー が発生した場合
- 処理：`success=False, source="fallback"` で全財務項目を `None` として返す。Streamlit フォームは手入力モードで継続
- 根拠：EDINET 障害やメンテナンスで審査フロー全体が停止しないようにする

**BR-414**: 連結財務諸表優先
- 条件：連結財務諸表と単体財務諸表の両方が存在する
- 処理：連結財務諸表の値を優先して取得する
- 根拠：リース審査では企業グループ全体の財務状況の把握が重要

**BR-415**: API レート制限対応
- 条件：EDINET API 呼び出し前
- 処理：リクエスト間に最低 0.5秒 の `time.sleep()` を挿入する
- 根拠：EDINET API 利用規約のレート制限（詳細非公開）への配慮

**BR-416**: キャッシュへの書き込み
- 条件：EDINET API から正常に財務データを取得できた場合
- 処理：`edinet_cache` テーブルに `INSERT OR REPLACE` で保存する（同一法人番号・事業年度は上書き）
- 根拠：次回以降のリクエストでキャッシュを活用できるようにする

---

## 7. UI / UX

Streamlit 審査フォームに「EDINET から取得」ボタンを追加する。

```
┌─────────────────────────────────────────────────┐
│ 法人番号  [1234567890123         ] [EDINET取得]  │
│                                                   │
│ ▼ EDINET取得結果（2024年度）                      │
│   売上高:        1,234 百万円  ✓ 自動入力         │
│   営業利益:        123 百万円  ✓ 自動入力         │
│   当期純利益:       45 百万円  ✓ 自動入力         │
│   自己資本比率:    38.5%       ✓ 自動入力         │
│                                                   │
│ ⚠ 取得できない項目は手入力してください             │
└─────────────────────────────────────────────────┘
```

- 取得成功: フォームフィールドに自動入力し「✓ EDINET取得」バッジを表示
- 取得失敗（fallback）: `st.warning("EDINET からデータを取得できませんでした。手入力してください。")` を表示
- キャッシュ使用時: `st.info("キャッシュから取得しました（{fetched_at}）")` を表示

---

## 8. Error Handling

| エラー条件 | 処理 | ユーザー向け表示 |
|-----------|------|----------------|
| 法人番号フォーマット不正 | `source="fallback"` を返す（BR-411） | 「法人番号は13桁の数字で入力してください」 |
| EDINET API タイムアウト（10秒） | `source="fallback"` を返す（BR-413） | 「EDINET に接続できませんでした。手入力してください。」 |
| HTTP 4xx / 5xx エラー | `source="fallback"` を返す（BR-413） | 「EDINET からデータを取得できませんでした。」 |
| XBRL パースエラー | `source="fallback"` を返す（BR-413） | 「財務データの解析に失敗しました。」 |
| edinetCode が解決できない | `source="fallback"` を返す | 「EDINET に登録されていない法人番号です。」 |
| キャッシュ DB エラー | キャッシュを無視して API を叩く | （表示なし、ログにwarning） |

---

## 9. Acceptance Criteria

**AC-1101**: 有効な法人番号で財務データを取得できる（モック API 使用）
- Given: 有効な13桁の法人番号とモック EDINET API レスポンス
- When: `fetch_edinet_financials(corporate_number="1234567890123")` を呼ぶ
- Then: `success=True, source="edinet"` かつ `nenshu` が `float` で返る

**AC-1102**: キャッシュが有効期限内であれば API を叩かない
- Given: `edinet_cache` に `fetched_at` が1時間前のレコードが存在する
- When: `fetch_edinet_financials(corporate_number="1234567890123", use_cache=True)` を呼ぶ
- Then: `source="cache"` かつ HTTP リクエストが発行されない

**AC-1103**: キャッシュが25時間超過した場合は再取得する
- Given: `edinet_cache` に `fetched_at` が25時間前のレコードが存在する
- When: `fetch_edinet_financials(...)` を呼ぶ
- Then: `source="edinet"` が返る（API が呼ばれる）

**AC-1104**: 法人番号が12桁（フォーマット不正）で fallback を返す
- Given: `corporate_number="123456789012"`（12桁）
- When: `fetch_edinet_financials(...)` を呼ぶ
- Then: `success=False, source="fallback"` かつ `error` に `"invalid"` が含まれる

**AC-1105**: EDINET API タイムアウト時に例外が発生しない
- Given: API がタイムアウトするモック
- When: `fetch_edinet_financials(...)` を呼ぶ
- Then: 例外が発生せず `success=False, source="fallback"` が返る

**AC-1106**: HTTP 500 エラー時に fallback を返す
- Given: API が HTTP 500 を返すモック
- When: `fetch_edinet_financials(...)` を呼ぶ
- Then: `success=False, source="fallback"` が返る

**AC-1107**: EDINET 取得成功時にキャッシュが更新される
- Given: キャッシュが存在しない、有効な法人番号
- When: `fetch_edinet_financials(...)` を呼ぶ
- Then: `edinet_cache` テーブルに1件レコードが存在する

**AC-1108**: `use_cache=False` でキャッシュを無視して API を叩く
- Given: 有効なキャッシュが存在する
- When: `fetch_edinet_financials(..., use_cache=False)` を呼ぶ
- Then: `source="edinet"` が返る（API が呼ばれる）

**AC-1109**: resolve_edinet_code が不明な法人番号に対して None を返す
- Given: EDINET 企業情報に存在しない法人番号
- When: `resolve_edinet_code(corporate_number="9999999999999")` を呼ぶ
- Then: `None` が返り例外が発生しない

**AC-1110**: パフォーマンス要件（キャッシュヒット時 50ms 以内）
- Given: 有効なキャッシュが存在する
- When: `fetch_edinet_financials(..., use_cache=True)` を10回連続で呼ぶ
- Then: 合計処理時間が 500ms 以内

---

## 10. Non-Functional Requirements

- **パフォーマンス**: キャッシュヒット時 50ms 以内、API 取得時は最大 15秒（タイムアウト 10秒 + 処理 5秒）
- **後方互換性**: 既存スコアリングの出力値を変更しない
- **例外非伝播**: いかなる内部エラーも外部に raise しない（BR-413）
- **ログ**: API 取得時は `[edinet_collector] fetched corporate_number={} fiscal_year={}` を出力、fallback 時は `[edinet_collector] FALLBACK: {}` を出力
- **レート制限**: リクエスト間 0.5秒 sleep（BR-415）
- **テストカバレッジ**: AC-1101〜AC-1110 全件カバー必須
- **外部依存**: `requests`（HTTP）、`python-xbrl` または `lxml`（XBRL パース）を利用。`requirements.txt` に追記すること

---

## 11. Implementation Notes（Codex向け）

- **ファイル配置**:
  ```
  edinet_collector.py       （プロジェクトルート直下、新規作成）
  tests/
  └── spec_phase4/
      └── test_P4-002.py    （新規作成）
  ```
- **EDINET API キー**: `os.getenv("EDINET_API_KEY")` → 引数 `api_key` の優先順位で取得。キーなしでも多くのエンドポイントは利用可能だがレート制限が厳しい
- **法人番号チェックデジット検証（BR-411）**: 法人番号は13桁の数字で構成され、先頭1桁がチェックデジット。検証アルゴリズムは国税庁の mod-9 方式に従う。
  ```python
  def validate_corporate_number(cn: str) -> bool:
      if not cn.isdigit() or len(cn) != 13:
          return False
      # チェックデジット計算（国税庁方式）
      p_sum = sum(int(cn[i]) * (i % 2 + 1 if i < 12 else 0) for i in range(1, 12))
      # 正確には: Σ(奇数位置 × 1 + 偶数位置 × 2) を9で割った余りを9から引く
      total = sum(int(cn[i]) * (1 if (12 - i) % 2 == 0 else 2) for i in range(1, 13))
      check = (9 - (total % 9)) % 9
      return int(cn[0]) == check
  ```
  詳細は国税庁「法人番号の指定と公表」のチェックデジット説明を参照すること。
- **`resolve_edinet_code()` の企業情報一覧キャッシュ（BR-411補足）**: `GET /api/v2/companies.json?type=2` は全社データを一括返却するため大容量（数万社）。毎回取得すると実用上タイムアウトするリスクがある。以下の戦略でキャッシュすること:
  - `edinet_cache` DB に `edinet_company_list` テーブルを追加し企業情報（edinetCode, corporateNumber, filerName）を保存する
  - TTL: **7日**（EDINET の企業情報は日次更新だが変更頻度は低い）
  - `fetched_at` が7日以内のレコードがある場合は DB ルックアップのみで解決し API を呼ばない
  - キャッシュが存在しない場合のみ `companies.json` エンドポイントを叩き、全件を DB に INSERT OR REPLACE する
  - テストでは `companies.json` のモックレスポンスを使用し実際の API は叩かない
- **XBRL 財務項目の XPath**:
  - 売上高: `jppfs_cor:NetSales`（連結）/ `jpigp_cor:Revenue`（IFRS）
  - 営業利益: `jppfs_cor:OperatingIncome`
  - 当期純利益: `jppfs_cor:ProfitLossAttributableToOwnersOfParent`
  - 総資産: `jppfs_cor:Assets`
  - 自己資本比率: `jppfs_cor:EquityToAssetRatio`（なければ 自己資本/総資産 で計算）
- **金額単位**: XBRL 内は「円」単位のため、取得後に「百万円」に変換（÷1,000,000、小数点第1位まで）
- **equity_ratio の単位変換**: `jppfs_cor:EquityToAssetRatio` は XBRL の `decimals` 属性によって表現形式が異なる。取得時に以下のルールで % 値（例: 38.5）に統一すること:
  - `decimals="-4"` 以下（大きな絶対値）→ 円単位の金額フィールドと同じため無関係（本項目には非該当）
  - 値が 0.0〜1.0 の範囲内（小数表現）→ `× 100` して % に変換
  - 値が 1.0 超（百分率表現）→ そのまま使用
  - 自前計算（自己資本/総資産）の場合は結果を `× 100` すること
- **Streamlit 連携**: `components/` 内の審査フォームコンポーネントに「EDINET取得」ボタンを追加し、`st.session_state` 経由でフォーム値を更新する
- **テストでの HTTP モック**: `unittest.mock.patch("requests.get")` でモックする。実際の EDINET API は叩かない
- **触れてはいけないファイル**: `scoring_core.py`, `total_scorer.py`, `asset_scorer.py`, `quantum_analysis_module.py`, `aurion/`

---

## 12. Test Plan

### 単体テスト（Codexが作成）

| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_1101 | AC-1101 | モックAPIで正常取得 → success=True, source="edinet" |
| test_1102 | AC-1102 | TTL内キャッシュ → source="cache", HTTP未発行 |
| test_1103 | AC-1103 | TTL超過キャッシュ → source="edinet"（再取得） |
| test_1104 | AC-1104 | 12桁法人番号 → success=False, source="fallback" |
| test_1105 | AC-1105 | タイムアウトモック → 例外なし, fallback |
| test_1106 | AC-1106 | HTTP 500モック → fallback |
| test_1107 | AC-1107 | 正常取得後にキャッシュレコードが1件存在 |
| test_1108 | AC-1108 | use_cache=False → APIが呼ばれる |
| test_1109 | AC-1109 | 不明法人番号 → None, 例外なし |
| test_1110 | AC-1110 | キャッシュヒット10回 → 500ms以内 |

### 回帰テスト
- 既存スコアリング出力が変化しないこと
- Streamlit 審査フローが EDINET 取得失敗時も正常動作すること

### 手動確認（実装後）
- [ ] Streamlit 審査フォームで「EDINET取得」ボタンが表示される
- [ ] 有効な法人番号を入力して取得ボタンを押すと、財務フィールドに値が自動入力される
- [ ] EDINET 取得失敗時に手入力フォームが表示される
- [ ] キャッシュ使用時に `st.info` メッセージが表示される
