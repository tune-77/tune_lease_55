---
agent: general-purpose
task: components/report.py に補助金シミュレーションパネル実装（render_subsidy_panel）
timestamp: 2026-03-21 00:00
status: success
reads_from:
  - .claude/reports/file-searcher/latest.md
  - .claude/reports/agent-team/subsidy_plan.md
---

## サマリー

`scoring/subsidy.py`（既存）の `match_subsidies` / `calc_subsidy_effect` を使い、
`components/report.py` に `render_subsidy_panel(res)` 関数を追加した。
`render_report()` 末尾から呼び出す形で組み込み済み。

---

## 詳細

### 事前調査で判明した事実

- `scoring/subsidy.py` は既に存在し、SQLite `subsidies` テーブルに5件の補助金マスタが投入済み。
- 実際のシグネチャは仕様書と一部異なる:
  - `match_subsidies(industry_sub, asset_name, region="全国")` — DB から取得
  - `calc_subsidy_effect(lease_amount_man, lease_term_months, subsidies)` — 戻り値キー: `effective_monthly`, `monthly_reduction`, `score_bonus`, `best_subsidy`
- これに合わせてパネル実装を調整した（spec の `lease_term` → `lease_term_months` など）。

### 実施変更

**`components/report.py`**

1. `_REPORT_CSS` の末尾（`</style>` の直前）に補助金パネル用 CSS を追記:
   - `.subsidy-panel-header` — 緑グラデーションのパネルヘッダー
   - `.subsidy-card` — 白背景・緑左ボーダー・shadow の補助金カード
   - `.subsidy-badge-high/medium/low` — 確度バッジ（緑/黄/赤）
   - `.subsidy-sim-grid/.subsidy-sim-box.before/.after` — 月額比較グリッド
   - `.subsidy-reduction-chip` — 軽減額チップ
   - `.subsidy-score-badge` — スコア加点バッジ
   - レスポンシブ対応（`@media max-width: 600px`）

2. `render_subsidy_panel(res: dict) -> None` 関数を追加（行 895、`_badge` ヘルパーの直前）:
   - `scoring.subsidy` が import できない場合は静かにスキップ（`try/except ImportError: return`）
   - `match_subsidies` / `calc_subsidy_effect` でのエラーも `try/except Exception: return` で保護
   - 補助金0件の場合は「該当する補助金情報がありません」と表示し、シミュレーションをスキップ
   - 補助金カードは最大3件を表示（補助金効果の高い順、match_subsidies 内でソート済み）
   - 月額シミュレーション: 元月額 vs 補助後月額を横並び比較、月額軽減額・総軽減額を chip 表示
   - スコア加点バッジを表示

3. `render_report()` 末尾（行 1796）に `render_subsidy_panel(res)` の呼び出しを追加

### データフロー

```
res["industry_sub"] + res["asset_name"]
  → match_subsidies(industry_sub, asset_name)  # SQLite subsidies テーブル参照
  → calc_subsidy_effect(lease_amount_man, lease_term_months, subsidies)
  → render_subsidy_panel が HTML 生成・表示
```

月額の計算:
- `acquisition_cost`（千円）/ 10 = `lease_amount_man`（万円）
- `lease_amount_man / lease_term` = 月あたり原価（万円）

---

## 課題・リスク

- `subsidies` テーブルは `match_subsidies` が DB 接続を毎回 open/close するため、Streamlit のリロードが多い環境では軽微な I/O オーバーヘッドがある（実害は小さい）
- `calc_subsidy_effect` の `effective_monthly` は「補助金を総額で受け取りリース期間に均等割」という簡易計算であり、実際の補助金は申請・採択後の一時払いが多い点に注意が必要
- 補助金マスタ（`subsidies` テーブル）の鮮度維持は手動運用のため、制度変更への追随が必要

---

## 後続エージェントへの申し送り

- **管理UI**: `components/admin_subsidy.py`（補助金マスタ編集画面）が未実装。次の Phase 2 タスク。
- **スコアリング組み込み**: `scoring/score_engine.py` への補助金加点統合は未実施。`render_subsidy_panel` の `effect["score_bonus"]` を表示するのみで、実スコアには反映されていない。
- **PDF 出力**: 補助金パネルの印刷対応は Phase 3 課題。
