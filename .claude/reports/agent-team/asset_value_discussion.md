---
agent: agent-team（田辺・ダッシュ・鈴木・プランナー）
task: 物件資産価値スコアリングの改善点議論
timestamp: 2026-03-21 17:30
status: success
reads_from: [.claude/reports/file-searcher/latest.md]
---

## サマリー

4エージェントが物件資産価値スコアリングの現状を精査し、合計22件の改善提案を提出した。
最重要課題は「情報欠如がリスクとして扱われない」「Slack・バッチ・個別で3系統のロジックが分裂している」「asset_scorer.pyの豊富な出力（grade・warnings・recommendation）がUIで完全に死んでいる」の3点で全員一致。

---

## 🔴 即効性が高い改善（優先度：高）

### A-1. Slack ↔ ASSET_ID_TO_CATEGORY のマッピング不整合解消（鈴木：Sサイズ 1日）
- `slack_screening.py` の ASSET_LIST（9種類）に対し、`category_config.py` の ASSET_ID_TO_CATEGORY は4種類のみ
- Slack審査では machinery/construction/food/office/solar にカテゴリ別スコアリングが動かない
- **改善案:** `ASSET_ID_TO_CATEGORY` に5種類を追加。`category_config.py` に5行 + `_store_value()` を10行修正

### A-2. 情報欠如がリスクとして扱われない問題（田辺：Sサイズ 1日）
- `lease_assets.json` 未配置・未入力項目があると `asset_score=50` 固定。情報不足案件が中立スコアを得る
- **改善案:** 入力完備率（completeness_ratio）をペナルティ係数として導入。
  `total_score × completeness_ratio + 50 × (1 - completeness_ratio)` 形式で補正

### A-3. `scoring_core.py` のデフォルト50をトレーサブルに（鈴木：Sサイズ 1日）
- `asset_score=None` のときに50が入っても「計算不能」か「本当に中立」か区別できない
- **改善案:** `warnings` に「物件スコア未入力のためデフォルト値(50)を使用」を追記。`used_default_asset_score: True` フラグを返却dictに追加

### A-4. warnings・推奨リース条件をUIに表示する（ダッシュ：即日実装可・計15行）
- `total_scorer.py` はすでに `warnings`・`recommendation`・`grade` を正しく返しているが `analysis_results.py` で**一切表示していない**
- **改善案（5行）:** `st.warning(w)` で warnings をスコアカード直後に表示（C=amber/D=red）
- **改善案（10行）:** `st.info()` + `st.columns(3)` で「最長リース年数 / 推奨残価率 / 備考」を固定表示
- ロジック変更ゼロ・即日実装

### A-5. lease_assets.json とカテゴリIDの命名規則を統一（鈴木：Sサイズ 1日）
- `lease_assets.json` の id（例: `construction_machine`）と `ASSET_ID_TO_CATEGORY` のkey（例: `construction`）が不一致
- **改善案:** `ASSET_ID_TO_CATEGORY` を正規ID辞書とし、`lease_assets.json` のidをそちらに統一。エイリアスマッピングも可

---

## 🟠 審査精度・UI品質を上げる改善（優先度：中）

### B-1. 物件グレードバッジをスコアカード最上部に表示（ダッシュ：20行）
- 物件グレードが C/D でも総合スコアが B に見えることがある。物件リスクが視界から消える
- **改善案:** `st.markdown()` + インラインHTMLでバッジを表示。物件グレード ≠ 総合グレードの場合は `st.warning()` で自動アラート

### B-2. カテゴリ別5項目レーダーチャート（ダッシュ：15行・既存関数流用）
- `item_scores`（各項目のスコア・ウェイト・寄与度）が返却されているが、UIで一切描画されていない
- **改善案:** `charts.py` の `plot_radar_chart_plotly()` を流用。`st.plotly_chart()` で即表示できる

### B-3. 動的ウェイト調整の差分テーブル表示（ダッシュ：20行）
- `weight_adjusted=True` でもUI上に何も表示されず、稟議書に根拠を書けない
- **改善案:** `st.expander("⚙️ 動的ウェイト調整が適用されました")` 内に基本/調整後ウェイトの差分テーブルを表示

### B-4. 車両カテゴリにEV残価リスクの動的係数を追加（田辺：Mサイズ 3日）
- `ev_tech_risk`（重み10）が静的入力依存。バッテリー劣化カーブ・補助金継続可能性が未反映
- **改善案:** `contract` に `vehicle_fuel_type`（EV/HEV/ICE）を追加。純EV + 48ヶ月超でウェイトを1.5倍引き上げ。`max_lease_years` をEVは4年以内に制限

### B-5. 産業機械の「カスタマイズ度」評価軸を新設（田辺：Mサイズ 3日）
- `versatility`（汎用性）が主観頼み。専用ライン組み込み型と汎用旋盤で担保価値は10倍以上異なる
- **改善案:** `customization_level`（カスタマイズ度の低さ、重み15）を新設。「標準品/一部カスタム/専用品」の3択。専用品なら `resale_market` にペナルティ（×0.5）

### B-6. 医療機器の「再販適格性リスク」を評価軸に追加（田辺：Mサイズ 3日）
- `regulatory_risk`（重み30）は「現在の借手が使えるか」の評価。「次の買い手が見つかるか」が未評価
- **改善案:** `resale_eligibility`（再販適格性、重み15）を追加。「一般管理/管理/高度管理・特定保守管理」の3段階スコア

### B-7. batch_scoring.py の物件スコアを calc_asset_score() に統合（鈴木：Mサイズ 3日）
- `(term_ok + cost_ok) / 2.0 × 100` という独自計算。バッチと個別で同一案件の判定が逆転しうる
- **改善案:** CSVに「物件カテゴリ」列を追加（任意）。入力あれば `calc_asset_score()` 呼び出し、なければフォールバック

---

## 🟡 精度向上・中期施策（優先度：中長期）

### C-1. 中古市場価格の「時点性」をスコアに反映（田辺）
- `lease_assets.json` のスコアは取得時点固定。5年リースのITサーバーなら満了時は20〜30%以下
- **改善案:** カテゴリ別残価率テーブルで「リース満了時推定スコア」を算出。`static_data/useful_life_equipment.json` が既存で接続できる

### C-2. 補助金マスタとの連動（プランナー）
- `subsidy_master.py`（既実装）が `calc_asset_score()` と接続されていない
- **改善案:** `calc_asset_score()` に `subsidy_info` 引数を追加。補助金対象品は `residual_value` タグに最大+8点ボーナス。`match_subsidies()` の返り値をそのまま渡せるインターフェース設計

### C-3. 物件評価確認チェックボックスによる審査記録の担保（ダッシュ：10行+PDF連携）
- 担当者が「物件価値を確認した」という証跡が残らない
- **改善案:** `st.checkbox("✅ 物件評価を確認しました")` を追加。PDFレポートに「確認済み: YYYY-MM-DD HH:MM 担当: ○○」を差し込む

### C-4. Slackセッションの排他制御（鈴木：Sサイズ 1日）
- `data/slack_sessions.json` の read-modify-write 方式は並行書き込みでデータ上書きが発生する
- **改善案:** `filelock` で排他ロック（短期）。SQLite WALモードへ移行（中期）

### C-5. GX・規制適合性スコアを全カテゴリに追加（プランナー：Mサイズ 3日）
- `ev_tech_risk` が車両のみ。産業機械（排ガス規制）・IT機器（電力規制）・医療機器（冷媒規制）が未反映
- **改善案:** 全カテゴリ共通タグ `policy_sensitivity` を新設。「GX認定済み/計画中/未対応/規制直撃品目」の4段階入力

---

## 🔵 将来の新評価軸・アーキテクチャ（優先度：長期）

### D-1. 市場実勢価格連動スコアの自動更新レイヤー（プランナー：Lサイズ 1週間〜）
- 車両はJAA公開指数、IT機器はリサイクル業者API、産業機械は中古機械業者データを週次ポーリング
- `fetch_estat_annual.py` の基盤を流用。CAGRが負なら `residual_value` タグのウェイトを最大1.4倍引き上げ

### D-2. 流動性スコアの独立評価軸化（プランナー：Mサイズ 3日〜）
- 現在は `market_liquidity`（IT機器15点）等として他軸に埋没している
- **改善案:** 独立項目化。「想定売却日数/最低買取価格の予見可能性/売却チャネル数」の3軸加重平均。30点未満なら総合スコアに-5点のハードルール

### D-3. スコア時系列劣化モデル（プランナー：Lサイズ 1週間〜）
- 技術陳腐化は指数減衰 `S(t) = S₀ × exp(-λt)`、物理耐久は線形減衰で別々にモデル化
- `calc_residual_trajectory()` 関数を追加。`st.line_chart` でリース期間中のスコア推移を可視化

### D-4. サブカテゴリ階層の導入（プランナー：Lサイズ 1週間〜）
- 現行4分類では「トヨタハイエース（180万）」と「中古軽トラ（30万）」が同じウェイトで評価される
- **改善案:** `SUBCATEGORY_SCORE_ITEMS` を追加。物件名NERによる自動サブカテゴリ推定（将来）

### D-5. asset_finance.py と asset_scorer.py の用途分離明文化（鈴木：Sサイズ 1日）
- 両者は統合せず、docstringで「使うべき場面・使うべきでない場面」を明記するだけで保守コストが下がる

---

## 📊 優先度マトリクス（全員合意）

| 優先度 | 改善 | 担当視点 | 工数 |
|:---:|---|---|---|
| 🔴 | Slack/ASSET_ID不整合解消 | 鈴木 | S(1日) |
| 🔴 | 情報欠如ペナルティ係数 | 田辺 | S(1日) |
| 🔴 | scoring_core.py デフォルト50トレーサビリティ | 鈴木 | S(1日) |
| 🔴 | warnings・推奨条件のUI表示 | ダッシュ | 即日(15行) |
| 🔴 | lease_assets.json ID命名統一 | 鈴木 | S(1日) |
| 🟠 | 物件グレードバッジ + 乖離アラート | ダッシュ | S(1日) |
| 🟠 | 5項目レーダーチャート | ダッシュ | S(1日) |
| 🟠 | EV残価リスク動的係数 | 田辺 | M(3日) |
| 🟠 | 産業機械カスタマイズ度評価軸 | 田辺 | M(3日) |
| 🟠 | 医療機器再販適格性評価軸 | 田辺 | M(3日) |
| 🟠 | batch_scoring統合 | 鈴木 | M(3日) |
| 🟡 | 満了時推定スコア（useful_life連動） | 田辺 | M(3日) |
| 🟡 | subsidy_master連動加点 | プランナー | M(3日) |
| 🟡 | 確認チェックボックス+PDF記録 | ダッシュ | S(1日) |
| 🟡 | Slackセッション排他制御 | 鈴木 | S(1日) |
| 🟡 | GX規制適合性スコア | プランナー | M(3日) |
| 🔵 | 市場実勢価格連動スコア | プランナー | L(1週間〜) |
| 🔵 | 流動性スコア独立化 | プランナー | M〜L |
| 🔵 | スコア時系列劣化モデル | プランナー | L(1週間〜) |
| 🔵 | サブカテゴリ階層導入 | プランナー | L(1週間〜) |
| 🔵 | 用途分離docstring明文化 | 鈴木 | S(1日) |

---

## 後続エージェントへの申し送り

- **実装担当**: 🔴の5件（Slack不整合・情報欠如ペナルティ・deafault50・warningsUI・ID統一）を最初のスプリントで実装。合計4〜5日
- **code-reviewer**: `category_config.py` への評価軸追加はウェイト合計100維持の確認が必要
- **test-runner**: `calc_asset_score()` への引数追加（subsidy_info等）は既存テストに影響する可能性あり
