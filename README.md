# リース審査AI システム

リース会社向けの社内審査支援ツール。財務データを入力するだけで審査スコアを算出し、金利サジェスト・軍師コメント・成約予測・改善提案まで一気通貫で行う Streamlit アプリ。

---

## 主な機能

| 機能 | 概要 |
|------|------|
| **審査スコアリング** | 単体LightGBM＋ベイズ推論で審査スコアを算出（承認ライン: 71点） |
| **限界改善シミュレーター** | ボーダーライン案件に「どの指標をいくら改善すれば承認圏内か」を提示 |
| **軍師コメント** | ベイズ推論＋LLM（Gemini）による審査所見の自動生成 |
| **金利サジェスト** | 過去の成約データから最適なリースレートを提案 |
| **基準金利マスタ** | 月次の基準金利を管理・参照（社内決定金利を毎月登録） |
| **競合関係グラフ** | 業種×競合他社の競合関係を D3.js で可視化 |
| **案件結果登録** | 審査後の成約/失注・獲得レート・競合情報を記録 |
| **自動係数最適化** | 成約実績50件到達後、以降20件ごとに回帰係数を自動更新 |
| **バッチ審査** | Excel アップロードで複数案件を一括スコアリング |
| **PDF出力** | 審査結果レポートを PDF で出力 |

---

## 起動方法

```bash
cd /path/to/tune_lease_55
./run_lease_app.sh
```

または:

```bash
streamlit run tune_lease_55.py
```

APIキーの設定（初回のみ）:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-gemini-api-key"
SLACK_BOT_TOKEN = "your-slack-bot-token"
ANYTHING_LLM_API_KEY = "your-anything-llm-key"
```

環境変数でも設定可能（優先順位: 環境変数 > secrets.toml > session_state）。

---

## ML モデルの仕組み

### スコアリングパイプライン

```
財務入力
  ↓
① ロジスティック回帰（LR）        ← 係数更新用
② LightGBM 分類器（LGB）          ← data/lgb_main_model.joblib / data/lgb_main_model_new.joblib
定量スコア（現在は LGB 単体を採用）
  ↓ + ベイズ推論 + 定性評価 + 物件スコア + 直感補正
最終スコア（0〜100点）
```

### DSCR 特徴量（キャッシュフロー系）

| 特徴量 | 定義 | 意味 |
|--------|------|------|
| `dscr_approx` | 営業利益 ÷ (減価償却費 + 賃借料) | 固定費に対する利益カバー力 |
| `interest_coverage` | 営業利益 ÷ 支払利息 | 利息支払い余力 |

### モデル再学習（ローカルで実行可能）

```bash
# 1. DB からトレーニングデータを抽出
python export_cases_for_colab.py   # → data/cases_for_colab.json

# 2. 学習（data/ 直下にモデルファイルを自動保存）
python train_lgb_colab.py
```

生成されるファイル:
- `data/lgb_main_model.joblib` — 既存先向け定量モデル
- `data/lgb_main_model_new.joblib` — 新規先向け定量モデル
- `data/lgb_qual_model.joblib` — 定性モデル

---

## 限界改善シミュレーター

スコアが承認ライン付近（71点 − 2以内）の案件に対し、改善提案パネルを自動表示する。

```
現在スコア: 67.4 → 要審議

✅ 単独改善で承認ラインに到達できる案
┌────────────┬──────────────────┬────────┬────────────┐
│ 指標       │ 改善量           │ 推定   │ 効果       │
├────────────┼──────────────────┼────────┼────────────┤
│ 売上高     │ +15%（約360万増）│ 72.1   │ 承認圏内   │
│ 信用格付   │ 要注意 → 4-6    │ 74.5   │ 承認圏内   │
└────────────┴──────────────────┴────────┴────────────┘
```

実装: `components/marginal_improvement.py`

---

## プロジェクト構造

```
tune_lease_55/
├── tune_lease_55.py              # エントリポイント・ページルーティング
├── scoring_core.py               # スコア計算コア（LR・LGB・DSCR・最適閾値）
├── coeff_definitions.py          # 全業種×既存先/新規先の回帰係数定義
├── asset_scorer.py               # 物件スコアリング
├── total_scorer.py               # 合計スコア集約
├── category_config.py            # 業種カテゴリ設定
├── bayesian_engine.py            # ベイズ推論エンジン
├── quantum_analysis_module.py    # 量子干渉スコア（財務矛盾検出）
├── data_cases.py                 # 案件データの読み書き
├── auto_optimizer.py             # 自動最適化トリガー
├── analysis_regression.py        # 回帰分析・混合重み最適化
├── export_cases_for_colab.py     # 学習データ抽出スクリプト
├── train_lgb_colab.py            # LightGBM 再学習スクリプト（ローカル実行可）
├── components/
│   ├── chat_wizard.py            # リースくんウィザード（対話型入力）
│   ├── form_apply.py             # 審査入力フォーム
│   ├── score_calculation.py      # スコア計算・ログ保存
│   ├── analysis_results.py       # 分析結果表示
│   ├── marginal_improvement.py   # 限界改善シミュレーター
│   ├── shinsa_gunshi.py          # 軍師コメント生成
│   ├── rate_suggestion.py        # 金利サジェスト
│   ├── graph_view.py             # 競合関係グラフ（D3.js）
│   ├── form_status.py            # 案件結果登録
│   ├── dashboard.py              # ダッシュボード
│   ├── batch_scoring.py          # バッチ審査
│   ├── report.py                 # レポート生成
│   ├── sidebar.py                # サイドバー UI
│   └── settings.py               # 設定画面
├── scoring/                      # スコアリングサブモジュール
│   └── feature_engineering_custom.py
├── slack_bot.py                  # Slack ボット
├── slack_screening.py            # Slack 審査フロー
├── data/                         # SQLite DB・モデルファイル（コミット禁止）
└── .streamlit/secrets.toml       # 秘密情報（コミット禁止）
```

---

## スコアリングの仕組み（詳細）

### 3モデル加重平均（LR ベース）

| モデル | 係数キー例 | デフォルト重み |
|--------|-----------|--------------|
| ① 全体モデル | `全体_既存先` / `全体_新規先` | 50% |
| ② 指標モデル | `指標_既存先` / `指標_新規先` | 30% |
| ③ 業種別モデル | `運送業_既存先` など | 20% |

- 顧客区分（既存先/新規先）で係数セットを自動切り替え
- 重みは50件蓄積後にクロスバリデーション（StratifiedKFold）で自動最適化

#### 最適化ロジック（3モデル）

- 実装関数: `analysis_regression.py::optimize_model_blend_weights()`
- 入力: `score_borrower`（全体）, `bench_score`（指標）, `ind_score`（業種別）と成約/失注ラベル
- 手順: 各foldで `StandardScaler + LogisticRegression` を学習 → 係数平均 → 非負化 → 合計1へ正規化
- 出力: `w_main`, `w_bench`, `w_ind`, `auc_cv`, `n_cases`, `n_scored_oof`
- 保存先: `auto_optimizer.py` 経由で `data/coeff_auto.json` の `_auto_blend_w_*` に保存

#### 実務での回し方（推奨）

1. 成約/失注の確定データを最低50件以上ためる（両クラス必須）
2. `auto_optimizer.run_auto_optimization(force=False)` を実行
3. `blend_weights.auc_cv` が直近運用より改善しているか確認
4. 月次で再学習し、重み推移（`w_main/w_bench/w_ind`）を監視

> 補足: 行数不足（<20件）・片側クラスのみの場合は `None` を返して更新をスキップします。


### 金利サジェスト

- 過去の成約データ（スプレッド・スコア・競合情報）から推奨金利レンジを算出
- 「競合なし失注」（現金購入・銀行融資への切り替え）はサンプルから除外
- スコアに応じた調整・競合他社情報がある場合は競合スプレッドも反映

---

## 運用ガードレール

### 自動係数更新の安全条件

- 成約/失注の登録済み案件が **50件** を超えた時点で初回実行
- 以降 **20件ごと** に自動実行
- AUC・判定乖離率・業種別精度を同時監視
- 結果は `data/coeff_auto.json` に保存、次回起動から自動反映

### 監査ログの最小セット

案件ごとに以下を保存する運用を推奨:

- 入力値（匿名ID・業種・主要財務指標）
- 使用係数バージョン（manual/auto と更新日時）
- 判定スコアと承認可否
- 金利提案の根拠（スプレッド/競合情報の有無）

### 運用 KPI（週次・月次）

- 承認率・成約率
- スコア帯別成約率
- 提示金利と獲得金利のギャップ
- モデル更新前後の性能差（AUC/成約率）

---

## データファイル

| ファイル | 内容 |
|---------|------|
| `data/lease_data.db` | 全案件ログ（SQLite）。スコア・成約/失注・金利・競合情報を記録 |
| `data/screening_db.sqlite` | 軍師モード用 DB（ベイズ証拠重み） |
| `data/coeff_overrides.json` | 係数の手動上書き・モデル混合重みの手動設定 |
| `data/coeff_auto.json` | 自動最適化で算出した係数の保存先 |
| `data/lgb_main_model.joblib` | LightGBM 定量モデル |
| `data/lgb_main_model_new.joblib` | LightGBM 定量モデル（新規先） |
| `data/lgb_qual_model.joblib` | LightGBM 定性モデル |
| `data/business_rules.json` | 業種別ビジネスルール |
| `data/industry_benchmarks.json` | 業種別財務指標ベンチマーク |

> `data/` 配下の `.db` / `.sqlite` / `.joblib` / `.jsonl` は Git で管理しない（`.gitignore` 参照）。

---

## 注意事項

- `coeff_definitions.py` はリポジトリルートに配置（直接参照のため）
- Gemini API キーは `.streamlit/secrets.toml` で管理（Git にコミットしない）
- 基準金利は毎月末に翌月分を `📅 基準金利マスタ` 画面から登録する
- 数値の単位は **千円**（スコアリングモジュール内は円単位に変換して計算）
