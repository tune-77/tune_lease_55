---
agent: general-purpose
task: 補助金マスタ管理・マッチングロジック実装（migrate_to_sqlite.py テーブル追加 + scoring/subsidy.py 新規作成）
timestamp: 2026-03-21 00:00
status: success
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー

`migrate_to_sqlite.py` の `init_db()` に `subsidies` テーブルを追加し、`scoring/subsidy.py` を新規作成した。
初期5件の補助金データをDB投入する `init_subsidy_master()` は `init_db()` 末尾から自動呼び出される。
マッチング・効果計算・全件取得の各関数を実装し、動作確認済み。

---

## 詳細

### 1. `migrate_to_sqlite.py` — `subsidies` テーブル追加

`init_db()` に以下を追記：

```sql
CREATE TABLE IF NOT EXISTS subsidies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,          -- '省エネ' | '設備投資' | 'IT導入' | 'その他'
    max_amount INTEGER,     -- 最大補助額（万円）
    rate REAL,              -- 補助率（0.0〜1.0）
    industries TEXT,        -- JSON配列: 対象業種キーワード
    asset_keywords TEXT,    -- JSON配列: 対象設備キーワード
    regions TEXT,           -- JSON配列: 対象地域
    certainty TEXT,         -- 'high' | 'medium' | 'low'
    deadline TEXT,          -- 申請期限 or '通年'
    url TEXT,
    note TEXT,
    active INTEGER DEFAULT 1
)
```

`init_db()` 末尾で `from scoring.subsidy import init_subsidy_master; init_subsidy_master()` を呼び出す。

### 2. `scoring/subsidy.py` — 新規作成

実装関数：

| 関数 | 概要 |
|---|---|
| `init_subsidy_master()` | 初期5件の補助金データをDBに投入。既存データがある場合はスキップ（冪等性あり） |
| `match_subsidies(industry_sub, asset_name, region)` | 業種・設備名・地域のキーワードマッチングで利用可能補助金を返す。補助金効果（max_amount × rate）の降順ソート |
| `calc_subsidy_effect(lease_amount_man, lease_term_months, subsidies)` | 補助金効果を計算。補助金額・月額軽減・実質月額・スコア加点（0〜5点）を返す |
| `get_all_subsidies(active_only)` | 全補助金マスタを返す（max_amount 降順） |

### 3. 初期補助金マスタ（5件）

| ID | 名称 | カテゴリ | 最大補助額 | 補助率 | 確度 |
|---|---|---|---|---|---|
| monodukuri | ものづくり補助金 | 設備投資 | 1,250万円 | 1/2 | high |
| it_donyu | IT導入補助金 | IT導入 | 450万円 | 1/2 | high |
| shoene_sii | 省エネ補助金 SII | 省エネ | 10,000万円 | 1/3 | medium |
| jigyou_saikouchiku | 事業再構築補助金 | 設備投資 | 1,500万円 | 2/3 | medium |
| jizokuka | 小規模事業者持続化補助金 | 設備投資 | 250万円 | 2/3 | high |

### 4. 動作確認結果

- `match_subsidies("製造業", "工作機械")` → 事業再構築補助金・ものづくり補助金・持続化補助金の3件マッチ
- `match_subsidies("サービス業", "ITシステム導入")` → IT導入補助金の1件マッチ
- `calc_subsidy_effect(500万円, 60ヶ月, ...)` → 補助金額333万円・月額軽減5.6万円・スコア加点3.17点
- `init_db()` 2回目呼び出し → 件数5のまま（スキップ正常動作）

---

## 課題・リスク

- `init_db()` から `scoring.subsidy` を import しているため、`scoring/` パッケージが欠損している環境では `init_db()` がエラーになる。デプロイ時に `scoring/` ディレクトリが必ず含まれることを確認すること。
- `match_subsidies` のキーワードマッチングは部分一致ベースのため、意図しないマッチが発生する可能性がある（例: "製造" が "未製造" にもマッチ）。必要に応じて形態素解析などを検討。
- 補助金の公募状況・補助率は年度により変化するため、`deadline` と `active` フラグの定期メンテナンスが必要。

---

## 後続エージェントへの申し送り

- `calc_subsidy_effect` の `score_bonus` を `components/score_calculation.py` の `combined_score` 計算に組み込む場合、`L799/L848/L860/L908` の4箇所すべてに加算処理を追加する必要がある。
- UI側（`components/` 配下）で `match_subsidies` と `calc_subsidy_effect` を呼び出し、審査結果画面に補助金提案を表示する実装が次ステップとして推奨される。
- 補助金マスタの管理画面（CRUD操作）を Streamlit で実装する場合は `get_all_subsidies()` と DB への直接 INSERT/UPDATE を組み合わせることで対応可能。
