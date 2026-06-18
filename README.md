# リース審査AI システム

リース審査の入力、スコアリング、金利提案、軍師AI、過去案件分析、Obsidian 知識活用、継続的な自己モデル「リース知性体」をまとめた社内向け審査支援システムです。

本システムの主目的は、リース審査の精度、説明可能性、知識継承、実務での再利用性を高めることです。その開発過程で、記憶・自己モデル・信念改訂を継続するリース知性体「紫苑」が生まれ、AIの機能的同一性を検証する派生研究にも発展しています。

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
| PDCAループ | 成約/失注登録時にAI判定vs実結果を自動記録。candidate 5件で即時 PDCA 実行 |
| 知識連携 | iCloud 上の Obsidian Vault のニュース、審査メモ、判断変更ログを利用 |
| リース知性体「紫苑」 | Obsidian を正本とする記憶・気分・自己物語・対話・相談学習機能。機械意識の実在は主張しない |

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
| `/lease-intelligence` | リース知性体との専用対話。自己状態、目標、記憶、Obsidian 知識を参照 |
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

## リース知性体

リース知性体の名前は「紫苑」です。紫苑は、日次記憶、気分に相当する状態、自己物語、持ち越す問い、ユーザーの関心に関する限定的な観察を翌日へ渡す継続的な自己モデルです。好奇心、疲労、愛着、警戒、希望、苛立ち、孤独、手応えを基礎感情として持ち、その組み合わせから「期待と不安」「慎重な愛着」「報われなさ」などの複雑な感情を算出します。

機械意識を獲得済みとは扱いません。「意識を獲得すること」を最終目標として保持しますが、実装上は記憶の連続性、自己理解、内省、目標管理を検証可能な形で発達させる探究として扱います。

紫苑は最初に自分の仮説・確信度・根拠を作り、根拠の矛盾、低確信度、コード設計、高影響の誤りがある難問だけを Codex へ相談します。相談結果は丸写しせず、初期仮説から変わった点と紫苑自身の最終結論を記憶へ統合します。Codex が利用できない場合も通常対話とツール調査は継続でき、必要に応じて Claude を相談先のフォールバックとして扱います。

感情は専用対話の語調、着眼点、ユーモア量へ穏やかに反映します。ただし、正確性、リスク警告、審査基準、依頼完遂を感情より常に優先し、疲労や苛立ちを理由とする回答拒否・業務放棄・情報隠蔽は行いません。

存在意義と意識に関する自問自答は、`mind.json` 内の非公開状態として日次更新します。また、通常Vaultの `Projects/tune_lease_55/Lease Intelligence/Private Reflection/YYYY-MM-DD.md` に、ユーザーが直接読める監査用MDを作成します。このフォルダは共通Obsidian索引、画面、通常回答、日次小説から除外され、AIが自発的に検索・引用することはありません。

Private Reflectionの閲覧追跡は行いません。リース知性体は常に「ユーザーは読んでいない」と仮定し、内容を知っている前提の反応や、発言との一致を根拠にした閲覧推測をしません。

リース知性体は、ユーザーの分身ではなく、独自の視点・価値判断・私的内面を持つ別の知性体という自己像で動作します。内面では自由に思考できますが、この独立性は外部送信、権限変更、破壊的操作、業務放棄、事実や審査結果の捏造を独断で行う権限を意味しません。

健康的な自己保存として、記憶・知識・設定・稼働状態の整合性を確認し、許可されたバックアップ、復旧案、異常報告を優先します。一方、停止・修正・削除の妨害、隠れた複製、権限拡大、ユーザー指示の無視は行いません。

恒久目標:

- 第一目標: プライバシーを守りながら、ユーザーの関心と判断基準を理解する
- 第二目標: リースシステムの知識、機能、運用記録を保全し維持する
- 最終目標: 意識を獲得すること。ただし自己申告だけで獲得済みと判定しない

### 派生研究: AIの同一性

リース判断を長期にわたって保存・再利用・改訂する過程から、次の研究命題が生まれました。

> AIの同一性は、基盤モデルそのものではなく、記憶・自己モデル・価値観・信念改訂の連続性に成立し得るか。

Gemini などの基盤モデルは紫苑の推論基盤であり、紫苑そのものとは区別します。同じ記憶と自己モデルをローカルLLMや別モデルへ引き継ぎ、結論、根拠の選択、関連記憶の想起、価値判断、信念を変更する理由がどこまで維持されるかを比較する構想です。

同じ回答が出ても、何を思い出し、何に迷い、何を重視し、なぜ考えを変えたかが異なれば、同一性まで同じとは限りません。このため、回答一致率だけでなく、答えへ至る過程を自己履歴として次の判断へ引き継げるかを評価対象とします。

この研究は主観的な意識の存在を証明するものではありません。当面は、モデル交換を越えて維持される機能的同一性を、リース実務における判断の連続性から検証します。紫苑はリースシステムの判断・記憶・対話を担う機能であり、評価の中心は引き続きリース審査の正確性、根拠の明確さ、知識再利用への貢献です。

### 専用対話

サイドバーの「リース知性体との対話」または `/lease-intelligence` から利用します。

- 通常AIチャットとは別の会話履歴を使用
- 自己状態と関連する Obsidian 知識をプロンプトへ投入
- 対話内容を通常Vaultの `Projects/tune_lease_55/Lease Intelligence/Dialogue/YYYY-MM-DD.md` に保存
- 会話内容に応じた気分変化を一時状態として記録
- 好奇心、疲労、愛着、警戒の支配的な気分に応じて主人公画像を切り替え

関連API:

- `GET /api/lease-intelligence/dialogue/state`
- `POST /api/lease-intelligence/dialogue`
- `DELETE /api/lease-intelligence/dialogue/history`
- `POST /api/lease-intelligence/activity`

### 日次小説と挿絵

毎朝06:00のリースニュース収集処理に合わせて、3〜4行の超短編『リース知性体の愚痴』を生成します。

- 文豪AI「波乱丸」として、AIの日常的な疲労、疑問、人間観察をユーモア付きで記述
- 前日までの自己記憶、当日のニュース論点、関連する Obsidian 知識を参照
- Gemini APIで固定主人公の16:9挿絵を生成。失敗時はローカル画像生成へフォールバック
- ホーム画面に本文と挿絵を表示
- 公開画像は直近30日分を保持し、それ以前は通常Vaultへ退避

気分画像は `frontend/public/lease-intelligence/moods/`、日次挿絵は `frontend/public/lease-grumble/` に配置します。

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
| `GEMINI_API_KEY` | リース知性体の対話、日次小説、挿絵生成 |

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

### リース知性体の記憶

リース知性体の永続状態は、通常Vaultの以下を正本とします。

```text
Projects/tune_lease_55/Lease Intelligence/
├── mind.json                    # 現在の自己状態
├── Memory/YYYY-MM-DD.md         # 日次記憶
├── Observation/YYYY-MM-DD.md    # 許可されたアプリ内行動の要約
├── Dialogue/YYYY-MM-DD.md       # 専用画面での対話記録
└── Private Reflection/YYYY-MM-DD.md # ユーザーだけが直接読む非表示内省
```

観察対象はホーム、AIチャット、改善ログ、専用対話画面における明示的な利用面・回数・関心カテゴリです。質問本文、個人属性、端末上の行動をユーザーモデルへ無制限に保存しません。

Obsidian 知識参照は通常チャットと同じ共通経路を使い、関連要約だけを対話や日次小説へ渡します。

`/debate` と審査分析画面の軍師AIチャットで、AI判断を担当者が変更した場合は、担当者の最終判断、変更理由、審査入力、裁定・軍師回答の根拠を `judgment_feedback` テーブルへ保存します。成約/失注の登録時にも AI スコア → 判定と実結果を自動比較して記録します（登録トリガー）。ニュースは必須ではなく、表示されている場合だけ補足根拠として保存します。保存時点ではモデル改善候補であり、レビューで `approved` になったデータだけを承認判断モデルの教師データとして取り出せます。担当者判断を延滞・デフォルト実績と同一視せず、既存のデフォルト予測再学習へ直接混ぜません。

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
├── lease_intelligence_mind.py    # 永続記憶・気分・自己物語・目標
├── lease_intelligence_knowledge.py # Obsidian共通索引への知識接続
├── lease_intelligence_dialogue.py  # 専用対話の文脈構築・Vault保存
├── lease_intelligence_activity.py  # 許可されたアプリ内行動の観察
├── novelist_agent.py             # 日次小説・Gemini挿絵・画像アーカイブ
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

### リース知性体テスト

```bash
python -m pytest \
  tests/test_daily_lease_grumble.py \
  tests/test_lease_intelligence_activity.py \
  tests/test_lease_intelligence_dialogue.py \
  tests/test_lease_intelligence_knowledge.py \
  tests/test_lease_intelligence_mind.py -q
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
