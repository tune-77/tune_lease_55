# リース審査AI システム

リース審査の入力、スコアリング、金利提案、軍師AI、過去案件分析、Obsidian 知識活用をまとめた社内向け審査支援システムです。

現在の主系統は **Next.js + FastAPI** です。Streamlit 版は既存機能の参照・一部運用のために残していますが、日常利用と外部公開は `run_next_stable.sh` を使います。

---

## 現在の仕様

| 領域 | 現仕様 |
|------|--------|
| フロントエンド | Next.js production (`next build` -> `next start`) |
| API | FastAPI。Next の `/api/*` から FastAPI へプロキシ |
| 主起動 | `run_next_stable.sh` |
| 外部公開 | Cloudflare quick tunnel (`PUBLIC_TUNNEL=1`) |
| 主スコアAPI | `POST /api/score/full` |
| 借手モデル | 既存先: RandomForest / 新規先: LogisticRegression |
| 定性モデル | LR と LightGBM の比較表示 |
| Q_risk | 財務データの矛盾・歪みを示す補助指標。自動減点ではなく深掘り対象 |
| 知識連携 | iCloud 上の Obsidian Vault のニュース、審査メモ、判断変更ログを利用 |

---

## 主な画面

| 画面 | 役割 |
|------|------|
| `/home` | ホームダッシュボード。KPI、注目論点、最新リースニュース、ニュースダイジェストを表示 |
| `/` | 審査入力と分析結果。左は数値・モデル根拠、右は軍師AI |
| `/lease-kun` | 入力を絞ったスマホ向け審査導線 |
| `/quantitative` | 定量要因分析。LR / RandomForest / LGBM の指標・重要度を比較 |
| `/qualitative` | 定性因子分析。定性LR・LightGBM を比較 |
| `/history-dash` | 過去案件から成約ドライバー、平均財務、タグ傾向を確認 |
| `/finance` | 物件ファイナンス審査。Obsidian 関連メモと稟議条件案を利用 |
| `/chat` | AIチャット。Obsidian 文脈、ニュース論点、直近審査文脈を参照 |
| `/debate` | 境界案件の討論・軍師裁定 |
| `/report` | 審査レポート出力 |

### 審査分析画面の役割分担

`/` の分析画面は情報過多を避けるため、役割を分けています。

- 左カラム常時表示: 主要指標、重大警告、金利提案
- 左カラム折りたたみ: スコアDAG・Q_risk・類似度、詳細グラフ・情報源、稟議書・レポート出力
- 右カラム: 軍師AI。逆転戦略、審査部のツッコミ予測、顧客に聞くこと、稟議メモ、追加相談

軍師AIと重複する文章系ブロック（条件付き承認アクション、入力反映メモ、稟議コメント案、AI審査アドバイス、高度シミュレーション）は、分析画面の初期表示から外しています。詳細なモデル根拠は必要なときだけ開く設計です。

通常分析タブの名称は「数値分析」です。軍師AI側ではベイズゲージや類似案件一覧を再表示せず、数値分析の結果を受けて「次に何をするか」を提示します。

---

## 起動方法

### 通常起動

```bash
cd /path/to/tune_lease_55
bash run_next_stable.sh
```

起動後:

- Next: `http://127.0.0.1:3000`
- FastAPI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

### Cloudflare Tunnel 付き

```bash
PUBLIC_TUNNEL=1 bash run_next_stable.sh
```

起動ログに `https://xxxx.trycloudflare.com` が表示されます。quick tunnel の URL は使い捨てなので、毎回最新の `logs/next/tunnel_*.log` または起動ログを確認してください。

### 状態確認・部分再起動

フル再起動の前に `RESTART_SCOPE=status` を使います。

```bash
RESTART_SCOPE=status bash run_next_stable.sh
RESTART_SCOPE=api bash run_next_stable.sh
RESTART_SCOPE=next bash run_next_stable.sh
RESTART_SCOPE=tunnel bash run_next_stable.sh
```

`curl 200` だけで正常判断しないでください。特に `/home` は API 集計中でも本体を先に描画する設計なので、画面がローディングだけになっていないかも確認します。

### Streamlit 版

```bash
bash run_streamlit_stable.sh
```

Streamlit は旧導線・一部管理機能のために残しています。新規の画面改善は原則 Next 側に入れます。

---

## 環境変数・秘密情報

API キーは環境変数または `.streamlit/secrets.toml` で管理します。秘密情報は Git にコミットしません。

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-gemini-api-key"
SLACK_BOT_TOKEN = "your-slack-bot-token"
ANYTHING_LLM_API_KEY = "your-anything-llm-key"
```

主な環境変数:

| 変数 | 用途 |
|------|------|
| `PUBLIC_TUNNEL=1` | Cloudflare quick tunnel を起動 |
| `RESTART_SCOPE=status/api/next/tunnel` | 状態確認・部分再起動 |
| `API_HOST=127.0.0.1` | FastAPI bind host |
| `NEXT_HOST=127.0.0.1` | Next bind host |
| `ENABLE_GUNSHI_RAG=1` | 軍師AIでローカルRAGを有効化 |
| `ENABLE_OBSIDIAN_INDEXING=true` | FastAPI 起動時の Obsidian index を有効化 |

---

## スコアリング仕様

### スコア構成

```
入力値
  ↓
単位変換・特徴量生成
  ↓
借手モデル
  - 既存先: RandomForest
  - 新規先: LogisticRegression
  ↓
bench_score / ind_score / Q_risk / 定性評価 / 物件スコア / 直感補正
  ↓
最終スコア・判定・金利提案・軍師AI文脈
```

### 主要スコア

| 項目 | 意味 |
|------|------|
| `score_borrower` | 借手モデルの基礎スコア |
| `bench_score` | 業界ベンチマーク比較用の参考指標 |
| `ind_score` | 業種別比較用の参考指標 |
| `asset_score` | 物件の保全性・汎用性・残価リスク等 |
| `score` / `score_base` | 画面表示用の総合スコア |
| `ai_prob` | RandomForest 由来の PD 表示。失敗時のみフォールバック |
| `quantum_risk` | Q_risk。財務入力の矛盾・歪みを示す補助指標 |

`bench_score` と `ind_score` はブレンド前提ではなく、乖離アラート・参考比較に使います。

### モデルファイル

| ファイル | 内容 |
|---------|------|
| `data/lgb_main_model.joblib` | 既存先向け借手モデル。現在は RandomForest バンドル |
| `data/lgb_main_model_new.joblib` | 新規先向け借手モデル。現在は LogisticRegression バンドル |
| `data/ml_rf_v4.pkl` | 既存導線で利用する RandomForest 主モデル |
| `data/lgb_qual_model.joblib` | 定性モデル |

> `data/` 配下の DB、モデル、jsonl、キャッシュは原則 Git 管理しません。

---

## Obsidian・ニュース連携

既定の保存先は iCloud 上の通常の Obsidian Vault です。

```text
/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault
```

`lease-wiki-vault` は、明示的に指定された場合だけ使います。

### ニュース

最新リースニュースは主に以下を参照します。

```text
Obsidian Vault/05-クリップ_記事/リースニュース
```

`/api/lease-news/recent` はこのフォルダを主系統として読み、旧 `Obsidian Vault/リースニュース` も互換パスとして扱います。

ニュースは保存するだけでなく、`/home`、`/chat`、`/debate`、審査コメントの注目論点として再利用します。

`/debate` と審査分析画面の軍師AIチャットで、AI判断を担当者が変更した場合は、担当者の最終判断、変更理由、審査入力、裁定・軍師回答の根拠を `judgment_feedback` テーブルへ保存します。ニュースは必須ではなく、表示されている場合だけ補足根拠として保存します。保存時点ではモデル改善候補であり、レビューで `approved` になったデータだけを承認判断モデルの教師データとして取り出せます。担当者判断を延滞・デフォルト実績と同一視せず、既存のデフォルト予測再学習へ直接混ぜません。

関連API:

- `POST /api/lease-news/judgment-change`
- `POST /api/judgment-feedback`
- `GET /api/judgment-feedback/summary`
- `GET /api/judgment-feedback/candidates`
- `POST /api/judgment-feedback/{record_id}/review`

### AIチャットの Obsidian 検索ルール

Obsidian 検索は共通経路を使います。

- 検索語分解: `obsidian_query.py`
- AIプロンプト用文脈: `obsidian_ai_context.py`
- Vault検索本体: `mobile_app/obsidian_bridge.py`

チャット実装ごとに `vault.rglob("*.md")` を直接呼ばないでください。

---

## プロジェクト構造

```text
tune_lease_55/
├── api/
│   ├── main.py                  # FastAPI 本体、/api/score/full、ニュース、Obsidian連携
│   ├── gunshi_gemini.py          # 軍師AI SSE / Gemini / 代替戦略
│   └── knowledge/                # Obsidian index / vector store
├── frontend/
│   └── src/
│       ├── app/                  # Next.js App Router 各画面
│       ├── components/analysis/  # GunshiAdvice, Q_risk, レポート等
│       ├── components/form/      # 審査入力フォーム
│       └── lib/                  # API / 単位変換
├── components/                   # Streamlit 由来の審査・分析ロジック
│   ├── score_calculation.py      # run_scoring。APIモードでは副作用フックを抑制
│   ├── analysis_results.py       # Streamlit 分析表示
│   └── dashboard.py              # Streamlit ダッシュボード
├── scoring/                      # スコアリングサブモジュール
├── data/                         # DB・モデル・ログ。原則コミット禁止
├── logs/next/                    # 起動ログ・ビルドログ・tunnelログ
├── memory/                       # 日次作業メモ
├── run_next_stable.sh            # FastAPI + Next + tunnel 安定起動
└── run_streamlit_stable.sh       # Streamlit 安定起動
```

---

## 開発・検証

### Frontend build

```bash
cd frontend
npm run build
```

### Python syntax check

```bash
python -m py_compile api/main.py api/gunshi_gemini.py components/score_calculation.py
```

### API / Next 疎通

```bash
curl --max-time 10 -sS http://127.0.0.1:3000/ >/dev/null && echo NEXT_OK
curl --max-time 10 -sS http://127.0.0.1:8000/docs >/dev/null && echo API_OK
```

sandbox 内で `listen EPERM` や localhost 接続失敗が出る場合があります。`next-server` が生きているのに curl だけ失敗する場合は、権限付きで再確認してからアプリ障害と判断します。

---

## Git 運用

- `data/` 配下は原則コミットしない
- `.streamlit/secrets.toml` はコミットしない
- `frontend/.next/`、`node_modules/`、ログ、キャッシュはコミットしない
- 変更後は `npm run build` または該当する Python syntax check を実行する
- `git-ship` 時は `data/` と secrets を除外して commit/push する

---

## 運用上の注意

- ホームが開かないように見える場合、`curl 200` だけではなく、全画面ローディングで止まっていないか確認する
- Cloudflare quick tunnel の URL は毎回変わる。古い trycloudflare URL を信用しない
- `/api/score/full` は FastAPI プロセス安定性のため、APIモードでは Streamlit 側の recorder / FluidPipeline フックを抑制している
- 軍師AIの RAG は通常無効。必要時だけ `ENABLE_GUNSHI_RAG=1`
- Q_risk は「否決理由」ではなく「確認すべき歪みの候補」として扱う
- 画面改善では、情報を増やすより審査判断に必要な情報へ絞る
