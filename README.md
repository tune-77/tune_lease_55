# lease_logic_sumaho12

リース審査AI。回帰係数ベースの3モデル加重スコアリング・軍師コメント・自動最適化を統合した版。

---

## 起動方法

```bash
cd /Users/kobayashiisaoryou/clawd/lease_logic_sumaho12
streamlit run lease_logic_sumaho12.py --server.port 8502
```

---

## データファイル

| ファイル | 内容 |
|---------|------|
| `data/lease_data.db` | 案件データ（SQLite）。過去案件・スコア結果・成約/失注を記録 |
| `data/screening_db.sqlite` | 軍師モード用DB（gunshi_cases テーブル） |
| `data/coeff_overrides.json` | 係数の手動上書き・モデル混合重みの手動設定 |
| `data/coeff_auto.json` | 自動最適化で算出した係数・混合重みの保存先 |
| `data/business_rules.json` | 業種別ビジネスルール |
| `data/industry_benchmarks.json` | 業種別財務指標ベンチマーク |

---

## スコアリングの仕組み

### 3モデル加重平均

| モデル | 係数キー例 | デフォルト重み |
|-------|-----------|--------------|
| ①全体モデル | `全体_既存先` / `全体_新規先` | 50% |
| ②指標モデル | `指標_既存先` / `指標_新規先` | 30% |
| ③業種別モデル | `運送業_既存先` / `運送業_新規先` | 20% |

- 重みはデータが50件蓄積されると **クロスバリデーション（StratifiedKFold）で自動最適化**
- 手動設定は `data/coeff_overrides.json` の `model_blend_weights` で上書き可能

### 既存先・新規先の切り替え

`customer_type`（既存先 / 新規先）に応じて係数セットを自動切り替え。
新規先は `lease_credit_log=0`, `contracts=0` が設計上の初期値（取引実績なし）。

### 承認ライン

総合スコア **71以上** で「承認圏内」。

---

## 自動最適化（auto_optimizer.py）

- 成約/失注の登録済み案件が **50件到達**、以降 **20件ごと** に自動実行
- 最適化内容：回帰係数の更新 ＋ 3モデル混合重みのクロスバリデーション
- 結果は `data/coeff_auto.json` に保存

---

## フォルダ構成

```
lease_logic_sumaho12/
├── lease_logic_sumaho12.py      # メインアプリ（起動エントリーポイント）
├── coeff_definitions.py         # 全業種×既存先/新規先の回帰係数定義
├── scoring_core.py              # スコア計算コア
├── data_cases.py                # 案件データの読み書き・混合重み取得
├── auto_optimizer.py            # 自動最適化トリガー
├── analysis_regression.py       # 回帰分析・混合重み最適化
├── components/
│   ├── form_apply.py            # 審査入力フォーム（担当者直感スコア含む）
│   ├── score_calculation.py     # 3モデル加重スコア計算
│   ├── analysis_results.py      # 分析結果表示
│   ├── shinsa_gunshi.py         # 軍師コメント生成（ベイズ推論＋LLM）
│   ├── sidebar.py               # サイドバーUI（API設定等）
│   ├── dashboard.py             # ダッシュボード
│   └── settings.py              # 設定画面
├── data/                        # データ・設定ファイル
└── docs/                        # ドキュメント（スコア計算式まとめ等）
```

---

## 軍師コメント（shinsa_gunshi.py）

- ベイズ推論で承認確率を算出し、LLM（Gemini / Ollama）で推薦コメントを生成
- **担当者の直感スコア**（1〜5）を入力フォームで受け取り、ベイズ計算に反映
  - 1=かなり懸念 / 3=ニュートラル（デフォルト）/ 5=強い確信
- GeminiのAPIキーは `.streamlit/secrets.toml` に `GEMINI_API_KEY = "..."` で設定

---

## 注意

- `coeff_definitions.py` はリポジトリルートに置いてください（直接参照）。
- `data/` 配下の `.db` / `.sqlite` はGitで管理しないことを推奨（`.gitignore` 参照）。
