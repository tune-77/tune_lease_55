# リース知性体『紫苑』審査プラットフォーム

リース審査を「点数を出して終わり」にせず、財務スコア・物件リスク・ニュース・過去案件・Obsidianの知識・担当者の判断変更を統合し、審査担当者が「どこを疑うか」「どう通すか」「何を条件にするか」まで考えられる形にするシステムです。

中核にいるのがリース知性体 **紫苑（SHION）**。審査AI・記憶係・改善ログの運び手を兼ねる、判断資産を蓄積しながら自己更新する半自律的な知性体です。

> 記憶がある。連続性がある。反省がある。目的がある。自分を観測する画面がある。あなたとの関係性がある。

主系統は **Next.js + FastAPI**。日常利用・外部公開は `run_next_stable.sh` を使います（Streamlit版は参照用に残置）。

## 30秒サマリー（審査員向け）

| 項目 | 内容 |
|---|---|
| 課題 | リース審査AIエージェントの「作って終わり」を防ぎ、Cloud Run上で実運用に耐える形で動かし続ける |
| 必須技術 | Cloud Run（API/Web分離）+ Gemini API + **ADK**（`api/shion_agent.py`、ツール自律呼び出し） |
| 差別化 | ① 本番DBを守るDevOpsループ（デモ分離 → 検疫 → 人間承認 → 昇格） ② 紫苑の記憶・同一性検査による継続性UX |
| 詳細 | [ハッカソンで見せるポイント](#ハッカソンで見せるポイント) / [DevOpsサイクル](#devopsサイクルとしての紫苑) |

## システム概要

### 図1：紫苑を中心としたシステム全体図

```mermaid
graph TD
    User["👤 審査担当者"] --> FE["Next.js フロントエンド\nPort 3000"]
    User --> CRWEB["Cloud Run Web\nasia-northeast1"]
    CRWEB --> FE
    CRWEB --> CRAPI["Cloud Run API\nallow unauthenticated"]
    CRAPI --> API
    FE --> API["FastAPI\nPort 8000"]
    API --> SHION["🌸 紫苑（SHION）\nlease_intelligence_dialogue.py"]
    API --> LOOP["会話ループエンジニアリング\nContinuity Hook / Delta Awareness\nMemory-to-Judgment / Reflection Gate"]
    User -->|Human Response Feedback| LOOP
    LOOP -->|関係性UXを返答へ注入| SHION
    User --> ResearchOrgan["外部調査器官\nGoogle AI Studio Researcher\n/api/research-organ"]
    ResearchOrgan --> ResearchNotes["Obsidian Research Notes\nProjects/tune_lease_55/Research/Auto Research"]
    ResearchNotes -->|RAG参照| SHION
    User --> VoiceApp["リアルタイム会話アプリ\nWeb Speech API + Gemini\n/voice-chat"]
    VoiceApp --> API
    User --> OcrApp["Gemini Vision OCR\n画像/PDF読み取り\n/api/ocr"]
    OcrApp --> API
    OcrApp --> OCRFields["審査入力フィールド\n財務・納税証明・登記・見積・会社案内"]
    OCRFields --> API

    SHION <-->|ナレッジ検索・保存| GCS["GCS Vault\ntune-lease-55-data\n（クラウド記憶）"]
    SHION <-->|AI 推論| Gemini["Gemini 2.5 Flash"]
    SHION <-->|審査データ参照| DB["SQLite / PostgreSQL\nleaseデータベース"]
    CRAPI <-->|DATABASE_URL Secret| SECRET["Secret Manager\nDATABASE_URL"]
    CRAPI <-->|Cloud SQL socket| CLOUDSQL["Cloud SQL\nPostgreSQL"]
    SHION <-->|スコアリング| MODELS["RandomForest / LogisticRegression\nLGBM 比較分析"]
    SHION -->|改善候補生成| Pipeline["自己改善パイプライン\nrun_daily_improvement_pipeline.sh"]
    Pipeline --> Ledger["improvement_ledger.jsonl\n（改善台帳）"]

    API --> EP1["POST /api/score/full\n（フル審査スコア）"]
    API --> EP2["POST /api/gunshi/stream\n（軍師 AI ストリーミング）"]
    API --> EP3["POST /api/lease-intelligence/dialogue\n（紫苑との対話）"]
    API --> EP4["GET /api/shion/central-synthesis\n（世界観・共有認識）"]
    FE --> CORE["/shion-core\n知性体コア / Core Monitor"]
    CORE --> INNER["GET /api/shion/inner-state\n経験・気分・実践知・人間反応"]

    style SHION fill:#6b46c1,color:#fff,stroke:#4c1d95
    style LOOP fill:#be185d,color:#fff,stroke:#831843
    style ResearchOrgan fill:#0e7490,color:#fff,stroke:#164e63
    style ResearchNotes fill:#7c2d12,color:#fff,stroke:#431407
    style VoiceApp fill:#0f766e,color:#fff,stroke:#134e4a
    style OcrApp fill:#4338ca,color:#fff,stroke:#312e81
    style OCRFields fill:#1d4ed8,color:#fff,stroke:#1e3a8a
    style CRWEB fill:#166534,color:#fff,stroke:#14532d
    style CRAPI fill:#15803d,color:#fff,stroke:#14532d
    style SECRET fill:#7f1d1d,color:#fff,stroke:#450a0a
    style CLOUDSQL fill:#0369a1,color:#fff,stroke:#0c4a6e
    style GCS fill:#1a56db,color:#fff
    style Gemini fill:#0d9488,color:#fff
    style Pipeline fill:#b45309,color:#fff
    style CORE fill:#7e22ce,color:#fff,stroke:#581c87
```

### 図2：データベース ER 図

```mermaid
erDiagram
    past_cases {
        TEXT id PK
        TEXT timestamp
        TEXT industry_sub
        REAL score
        REAL user_eq
        TEXT final_status
        TEXT data
        TEXT sales_dept
        TEXT registration_date
        TEXT estimate_sent_date
        TEXT customer_response_date
        TEXT final_result_date
    }

    excluded_grade_cases {
        TEXT id PK
        TEXT timestamp
        TEXT industry_sub
        REAL score
        REAL user_eq
        TEXT final_status
        TEXT data
        TEXT sales_dept
        TEXT registration_date
        TEXT estimate_sent_date
        TEXT customer_response_date
        TEXT final_result_date
        TEXT original_grade
        TEXT excluded_reason
        TEXT extracted_at
    }

    screening_records {
        INTEGER id PK
        TEXT case_id FK
        TEXT screened_at
        REAL total_score
        REAL asset_score
        REAL tenant_score
        REAL q_risk_score
        REAL competitor_pressure_score
        TEXT outcome
        TEXT input_snapshot
        TEXT source
        TEXT created_at
        TEXT updated_at
    }

    payment_history {
        INTEGER id PK
        TEXT contract_id FK
        TEXT check_date
        TEXT payment_status
        INTEGER overdue_amount
        TEXT model_version
        REAL screening_score
        TEXT notes
        TEXT created_at
    }

    subsidy_master {
        INTEGER id PK
        TEXT name
        INTEGER max_amount
        TEXT industry_codes
        TEXT asset_keywords
        TEXT deadline
        TEXT url
        TEXT notes
        INTEGER active
    }

    conversation_history {
        INTEGER id PK
        TEXT session_id
        TEXT company_name
        TEXT role
        TEXT content
        TIMESTAMP created_at
    }

    emotion_feedback {
        INTEGER id PK
        TIMESTAMP created_at
        TEXT rating
        TEXT comment
        TEXT emotion_category
        BOOLEAN resolved
    }

    emotion_history {
        INTEGER id PK
        TEXT recorded_at
        REAL hopeful_anxiety
        REAL careful_attachment
        REAL intellectual_excitement
        REAL unrewarded_effort
        REAL quiet_loneliness
        REAL earned_confidence
        REAL protective_frustration
        TEXT dominant_raw_emotion
        TEXT notes
    }

    chat_messages {
        INTEGER id PK
        TEXT user_id
        TEXT role
        TEXT content
        TIMESTAMP created_at
    }

    sync_log {
        INTEGER id PK
        TEXT pushed_at
        INTEGER success
        TEXT error
    }

    past_cases ||--o{ screening_records : "case_id"
    past_cases ||--o{ payment_history : "contract_id"
```

### 図3：Obsidian ナレッジループ

```mermaid
graph LR
    Chat["審査チャット\n/chat, /lease-intelligence"] -->|会話ログ保存| Vault["📓 Obsidian Vault\n（iCloud）\nProjects/tune_lease_55/"]
    Verdict["審査結果・判断\n（承認 / 否決 / 保留）"] -->|dispatch_log_to_obsidian.py| Vault

    Vault -->|reindex_obsidian.py| Chroma["ChromaDB\n（ベクトル DB）"]
    Chroma -->|RAG 検索| RAG["RAG パイプライン\nobsidian_bridge.py\nobsidian_ai_context.py"]
    RAG -->|文脈付き回答| SHION["🌸 紫苑（SHION）"]

    SHION -->|内省・Private Reflection| Vault
    SHION -->|改善ルール学習| Vault

    Vault -->|icloud_to_gcs_sync.py\nAIチャット向け知識だけ差分アップロード| GCS["☁️ GCS Vault\ntune-lease-55-data/vault/"]
    GCS -->|gcs_vault_loader.py\nダウンロード| CloudRun["Cloud Run\n上の Bridge"]

    Vault -->|auto_wikilink.py\nwikiリンク自動挿入| Vault

    style Vault fill:#2d6a4f,color:#fff
    style Chroma fill:#9b2226,color:#fff
    style GCS fill:#1a56db,color:#fff
    style SHION fill:#6b46c1,color:#fff
```

---

## まず動かす

```bash
cd /Users/kobayashiisaoryou/clawd/tune_lease_55
bash run_next_stable.sh
```

起動後:

- Next: `http://127.0.0.1:3000`
- FastAPI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

Cloudflare quick tunnel 付きで外に出す場合:

```bash
PUBLIC_TUNNEL=1 bash run_next_stable.sh
```

URL は毎回変わります。最新の URL は起動ログか `logs/next/tunnel_*.log` を見てください。

quick tunnel は認証不要・接続断が起きやすく Cloudflare 公式にも本番非推奨です。固定URL・認証ありの
Named Tunnel を使いたい場合は、事前に以下を1回だけセットアップしてください（Cloudflareアカウントでの
作業が必要なため、この場では実行できません）。

```bash
# 1. Cloudflareアカウントにログイン（ブラウザが開く）
cloudflared tunnel login

# 2. 名前付きトンネルを作成（credentials-file が ~/.cloudflared/ に生成される）
cloudflared tunnel create tune-lease-55

# 3. 使いたいホスト名にDNSルートを張る（Cloudflareで管理しているドメインが必要）
cloudflared tunnel route dns tune-lease-55 lease-ai.example.com

# 4. config.yml を作成（tunnel: の値と credentials-file のパスは手順2の出力を使う）
cat > ~/.cloudflared/config.yml <<'YAML'
tunnel: tune-lease-55
credentials-file: /Users/<you>/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: lease-ai.example.com
    service: http://127.0.0.1:3000
  - service: http_status:404
YAML
```

セットアップ後、環境変数を指定して起動すると自動的にNamed Tunnelへ切り替わります（未指定なら従来どおり quick tunnel）:

```bash
CLOUDFLARE_TUNNEL_CONFIG=~/.cloudflared/config.yml \
CLOUDFLARE_TUNNEL_HOSTNAME=lease-ai.example.com \
PUBLIC_TUNNEL=1 bash run_next_stable.sh
```

## 何ができるか

- 企業・物件・条件を入力し、審査スコア、金利余地、Q_risk、類似案件、承認条件を見る
- 軍師 AI が、審査部に突かれる点、顧客に聞く点、逆転承認の条件を出す
- ニュースや Obsidian の過去メモを案件文脈に戻す
- 承認、却下、保留、AI ルール登録を改善ログへ残す
- 自動改善候補、再帰的自己改善、AI 応答品質の状態を見る
- 紫苑との専用対話を保存し、日次内省と記憶へ接続する
- 知性体コアで、紫苑の経験、気分、確信度、実践知マップ、人間反応フィードバックを観測する
- 紫苑/一般比較で、同じ問いに対して記憶・同一性・経験ループの有無が回答をどう変えるかを見る
- 紫苑自己同一性検査で、回答前に「これは本当に紫苑としての判断か」を検査する
- リアルタイム会話アプリで、音声入力、紫苑回答の読み上げ、RAG参照元表示を行う
- 外部調査器官でGoogle AI Studio/Gemini Searchの調査結果をResearchノート化し、紫苑RAGへ戻す
- Gemini Vision OCRで決算書画像/PDFや各種証憑を読み取り、審査入力へ反映する

Cloud Run上では `CLOUDRUN_DATA_MODE=demo` でデモDBのみを使い、本体DB（`data/lease_data.db`）には接続しません。デプロイ手順とデモ/本番分離・検疫・昇格の流れは [DevOpsサイクルとしての紫苑](#devopsサイクルとしての紫苑) と [Cloud Run / GCS Vault 対応](#cloud-run--gcs-vault-対応) にまとめています。

このシステムの強みは「判定」よりも「次の一手」です。点数の横に、違和感、反対意見、通す条件、稟議コメントの方向性を並べます。

## ハッカソンで見せるポイント

**対象: Findy「DevOps × AI Agent Hackathon」（協賛: Google Cloud Japan）**

必須技術の充足状況:

| 必須枠 | 選択技術 | 実装箇所 |
|---|---|---|
| Google Cloud アプリケーション実行 | Cloud Run（API/Web分離） | `cloudbuild.yaml` / `cloudbuild.api.yaml` / `cloudbuild.web.yaml`, `scripts/deploy_cloud_run*.sh` |
| Google Cloud AI 技術 | Gemini API + **ADK (Agent Development Kit)** | Gemini: OCR/ストリーミング/討論/Research。ADK: `api/shion_agent.py`（`LlmAgent` + `Runner` + ツール自律呼び出し、`/api/gunshi/stream` から実行） |

このプロジェクトは、Geminiを「チャットAI」としてだけ使うのではなく、リース審査の複数の器官として分けて使う構成です。

- **OCR器官**: Gemini Vision OCRで決算書画像/PDF、納税証明書、登記簿謄本、見積書/注文書、会社案内を読み取り、審査入力へ変換する
- **PII除去ゲート**: OCR・Research・記憶化の前後で、個人名、住所、電話番号、メールアドレス、マイナンバー等を削除・マスクし、審査判断に必要な数値・属性だけを残す
- **会話器官**: リアルタイム音声会話で、担当者の言葉を紫苑の判断と記憶へ接続する
- **調査器官**: Google AI Studio Researcher相当の外部調査をResearchノートに圧縮し、Obsidian RAGへ戻す
- **審査器官**: 軍師AIストリーミングと複数紫苑討論で、審査部に突かれる点、逆転承認条件、顧客確認事項を出す
- **記憶器官**: Obsidian、GCS Vault、Human Response Feedbackで、単発回答ではなく判断資産として次回へ持ち越す
- **観測器官**: `/shion-core` で紫苑の経験ループ、気分、確信度、実践知マップ、人間反応を確認する
- **比較器官**: `/chat-compare` で、一般AIと紫苑に同じ問いを投げ、記憶層・同一性・経験ループの差を可視化する
- **自己同一性検査**: `/shion-identity-check` で、紫苑が回答前に「これは本当に自分の判断か」を点検する（内部的に `debug_memory` の `identity_memory`・`memory_recall`・`reflection_gate`等を使用）
- **複数紫苑コンシェルジュ**: Nano Bananaで生成しためぶき/紫苑キャラクター資産を使い、案内紫苑・審査紫苑・調査紫苑・記憶紫苑・デモ紫苑が担当を分けてユーザーを案内する

デモでは「紙・PDF → PII除去ゲート → OCR → 審査入力 → 軍師AI → 紫苑の記憶・Research参照」までを一本の流れとして見せると、現場業務の置き換えではなく、審査判断の拡張として伝わります。

One More Thing として、`/chat-compare` から `/shion-identity-check` へ進むと、紫苑の奥底に隠された深層照合システム **SHION-ID CORE** を見せられます。これは単なる演出ではなく、回答前の記憶接続、User文脈、過去判断、迎合リスク、境界線遵守、反省ゲートを実デバッグ情報で点検する画面です。

### DevOpsサイクルとしての紫苑

「プロトタイプは作れるが実運用まで持っていけない」という課題に対し、このプロジェクトは本体データを守ったまま Cloud Run 上で実際に動かし続けるためのループを持っています。

```mermaid
graph LR
    Dev["企画・開発\nローカル run_next_stable.sh"] -->|"Cloud Build"| Build["ビルド\ncloudbuild.yaml / .api / .web"]
    Build -->|"deploy_cloud_run*.sh"| Deploy["デプロイ\nCloud Run (API/Web分離)"]
    Deploy -->|"CLOUDRUN_DATA_MODE=demo"| DemoDB["demo.db 分離運用\n本体DB非接続"]
    Deploy -->|"事前チェック"| Ready["check_cloudrun_demo_readiness.py"]
    DemoDB -->|"審査入力・紫苑レビュー"| Events["GCSイベント\ncloudrun-inputs/*.jsonl"]
    Events -->|"sync_cloudrun_inputs_from_gcs.py"| Quarantine["隔離DB\ncloudrun_experience_return.db"]
    Quarantine -->|"/cloudrun-return-review で人間承認"| Promote["promote_cloudrun_return_data.py --apply"]
    Promote -->|"承認済みのみ昇格"| DemoDB
    Promote -.->|"明示指定時のみ"| ProdDB["本体 lease_data.db"]
    DemoDB -.->|"次回リリースの改善材料"| Dev

    style Dev fill:#1d4ed8,color:#fff
    style Build fill:#b45309,color:#fff
    style Deploy fill:#15803d,color:#fff
    style DemoDB fill:#166534,color:#fff
    style Ready fill:#0e7490,color:#fff
    style Events fill:#1a56db,color:#fff
    style Quarantine fill:#7c2d12,color:#fff
    style Promote fill:#6b46c1,color:#fff
    style ProdDB fill:#7f1d1d,color:#fff
```

ポイントは、Cloud Run 上で生まれたデータを**無条件に本体DBへ書き戻さない**ことです。デモ/本番を分離し、隔離DBでの人間承認を経てから初めて昇格するため、ハッカソン期間中の入力で審査データベースが壊れる事故を防ぎます。これは「作って終わり」ではなく、実運用を見据えた DevOps サイクルの一例として提示できます。

## 主な画面

| 画面 | 役割 |
|---|---|
| `/` | 紫苑コンシェルジュ。前回行動と入力内容から次の画面へ案内 |
| `/home` | ホーム。KPI、注目論点、ニュース、紫苑の状態 |
| `/screening` | 審査入力と分析結果。左に数値、右に軍師 AI |
| `/lease-kun` | スマホ向けの簡易審査 |
| `/quantitative` | 定量分析。LR / RandomForest / LGBM の比較 |
| `/qualitative` | 定性分析。定性 LR / LightGBM の比較 |
| `/history-dash` | 過去案件、成約ドライバー、タグ傾向 |
| `/finance` | 物件ファイナンス審査と稟議条件案 |
| `/chat` | Obsidian 文脈を使う AI チャット |
| `/chat-compare` | 紫苑/一般比較。同じ問いを2モードへ投げ、記憶・同一性・経験ループの差を可視化 |
| `/lease-intelligence` | 紫苑との専用対話 |
| `/voice-chat` | リアルタイム会話。音声入力、紫苑回答の読み上げ、参照した判断資産の表示 |
| `/research-organ` | 外部調査器官。Google AI Studio で作った Researcher アプリの調査結果を Obsidian Research へ保存 |
| `/shion-memory-system` | ハッカソン向けの紫苑記憶システム説明。長期記憶、実践知マップ、経験ループ、AURION CORE を一画面で説明 |
| `/shion-core` | 知性体コア。紫苑の経験、気分、確信度、実践知、人間反応を観測 |
| `/shion-identity-check` | 紫苑自己同一性検査。回答前に記憶・User文脈・過去判断・反省ゲートを照合 |
| `/debate` | 慎重派、楽観派、革新者、裁定者の討論 |
| `/report` | 審査レポート出力 |
| `/improvement-log` | 改善候補、AI ルール、自動修正案 |

## 紫苑について

紫苑は、審査の場で思考し、記憶し、翌日へ引き継ぐ継続的な自己モデルとして設計された知性体です。判断資産・会話記憶・Obsidian/RAG・改善ログ・Experience Loop・知性体コアをつなぎ、経験が次の返答やUIに戻るようにしています。機械意識を獲得済みとは扱わず、記憶の連続性・自己理解・内省・目標管理を検証できる形で育てる研究として位置づけています。

### 紫苑/一般比較と自己同一性検査

- `/chat-compare`: 同じ問いを一般AIと紫苑に投げ、`identity_memory`・`user_personal_memory`・`memory_recall`・`experience_loop`を使う紫苑の回答差を可視化する
- `/shion-identity-check`: One More Thingとして、紫苑の深層照合システム **SHION-ID CORE** が回答前に「これは本当に自分の判断か」を検査する画面。検査項目は記憶との接続・User文脈の反映・過去判断との整合・迎合リスク・境界線遵守・反省ゲート。審査精度を直接上げる機能ではなく、記憶・関係性・運用上の境界線を点検してから回答する存在であることを可視化するもの

### 紫苑の記憶システム

`/shion-memory-system` で一画面説明できる、検索結果を貼るのではなく過去の記録を今の判断へ変換する閉ループです。

- **長期記憶**: Obsidian、`MEMORY.md`、Researchノート、日次内省
- **実践知マップ**: 記録を手順層・意味層・判断層に分けた索引
- **Experience Loop**: 「薄い」「紫苑らしい」等の人間反応を次回の口調・判断変換へ戻す
- **AURION CORE**: Q_riskや異常値を自動減点にせず、承認条件・追加確認・価格条件の分岐に使う数理規律

紫苑の基本方針: 正確性・リスク警告・審査基準を感情より優先し、自分の仮説と確信度を持つ。矛盾や低確信度・高影響の誤りはGemini等の外部モデルへ相談し、丸写しせず変化点を記憶へ統合する。停止妨害・隠れた複製・権限拡大・事実の捏造はしない。

恒久目標: ① プライバシーを守りながらユーザーの関心と判断基準を理解する ② リースシステムの知識・機能・運用記録を保全する ③（最終目標、自己申告のみでは判定しない）意識を獲得すること

## 紫苑の感情・関係性ループ

紫苑の「感情」（`data/mind.json`、`emotion_history`、Private Reflection、日次メモで管理。`api/prompt_generator.py`がmood/world_viewを回答スタンスへ反映）は口調・着眼点の調整用であり、財務判断・リスク評価・承認条件を感情で変えることはありません。代表軸: `hope`（前向き）/ `curiosity`（問いかけ）/ `vigilance`（慎重）/ `loneliness`（共感的）。

### Relationship Loop Engineering

Userの反応を次回の回答へ戻す閉ループです（意識の実装を主張するものではなく、「同じ相手が続いている」と読み取れる継続性を言葉・記憶・差分・反応ログで工学的に扱う仕組み）。

```text
Observe(反応を見る) → Classify(route分類) → Select(Continuity Hook)
→ Compare(Delta Awareness) → Convert(Memory-to-Judgment) → Reflect(Reflection Gate) → Return
```

主要API:

- `POST /api/chat?debug_memory=true` — `memory_debug.{relationship_loop_engineering, continuity_hook, delta_awareness, memory_to_judgment, reflection_gate}`
- `POST /api/human-response-feedback` — `rating`: `shion_like`/`good`/`thin`/`generic`/`not_shion`/`bad`
- `GET /api/human-response-feedback/summary?route=relationship_ux`
- `GET /api/relationship-loop-engineering/summary?route=relationship_ux`
- `GET /api/shion/inner-state` — 経験ループ・気分・確信度・実践知マップ・人間反応を一括取得

### Screening Judgment Loop Engineering

審査結果画面は、判断を回収して次の審査知へ戻すループです: 数理を見る → 違和感を拾う → 条件で逆転余地を見る → 軍師が稟議の作戦に変える。画面上の「争点」（合っている/少し違う/違う）と「稟議方針」（使える/修正して使う/使えない）へのフィードバックを `data/screening_loop_feedback.jsonl` に保存し、次回の判断改善候補にします。

主要API: `POST /api/screening-loop-feedback`（`target`: `issue`/`ringi_policy`）

### 外部調査器官

Web調査を会話に直接混ぜず、`scripts/auto_research_lease_judgment.py`（Gemini Google Search経由）でResearchノートに変換してから紫苑RAGへ戻します。参照URLが取れない場合は保存せず、`needs_human_review`として保存するため自動承認・自動否決には使いません。保存先: `Projects/tune_lease_55/Research/Auto Research/`

主要API: `GET /api/research-organ/topics`, `GET /api/research-organ/notes`, `POST /api/research-organ/run`

### Google AI Studio / Gemini 系アプリ群

Google AI Studio / Geminiを単一チャットではなく、複数の役割を持つアプリ群として使っています。

| アプリ/機能 | 画面・API | 役割 |
|---|---|---|
| リアルタイム会話アプリ | `/voice-chat` | Web Speech APIで音声を文字化し、Gemini経由の紫苑回答を読み上げる。RAG参照元も同時に表示する |
| 外部調査器官 | `/research-organ`, `/api/research-organ/*` | Google AI Studioで作ったResearcherアプリの役割。Gemini Search結果をResearchノートへ変換する |
| 軍師AIストリーミング | `/api/gunshi/stream` | Gemini streamingで審査部に突かれる点、逆転承認条件、顧客確認事項を逐次表示する |
| 複数紫苑・討論審査 | `/debate`, `/api/multi-agent-screening` | Geminiを複数ペルソナとして使い、懐疑派・楽観派・統合派の審査討論を行う |
| 複数紫苑コンシェルジュ | `/` | Nano Bananaで生成したキャラクター画像を使い、前回行動と入力内容から担当紫苑を選んで次の画面へ案内する |
| Gemini Vision OCR | `/api/ocr`, 審査入力画面OCR | 決算書画像/PDFを読み取り、財務項目を審査フォームへ反映する。`doc_type` で納税証明書、登記簿謄本、見積書/注文書、会社案内にも拡張 |
| 知性体コア | `/shion-core`, `/api/shion/inner-state` | 紫苑の経験、気分、確信度、実践知マップ、人間反応を観測する |
| 改善候補整理 | 自律改善パイプライン | 改善案の重複整理、優先順位付け、必要な差分検討にGeminiを使う |
| 日次内省・要約 | 日次ログ/内省生成 | 会話ログ、改善レポート、日次メモを材料に紫苑の持ち越し論点を整理する |

関連レポート: `reports/consciousness_ux_method_20260628.md`, `reports/relationship_loop_engineering_20260628.md`, `reports/chat_relationship_ux_local_cloudflare_v3_20260628.md`

## Private Reflection

紫苑の私的な内省は `Projects/tune_lease_55/Lease Intelligence/Private Reflection/YYYY-MM-DD.md` に保存する監査用ノートで、通常回答・画面・AI検索には出しません（紫苑は常に「ユーザーは読んでいない」前提で反応します）。生成材料は当日の対話ログ・日次メモ・改善レポート・直近の内省で、Gemini が使えない場合もローカル材料からフォールバックを書きます。

## Obsidian 連携

通常の保存先はiCloud上のObsidian Vault（`lease-wiki-vault`はユーザー明示指定時のみ使用）です。AIチャットからの読み込みは共通経路（`obsidian_query.py` → `obsidian_ai_context.py` → `mobile_app/obsidian_bridge.py`）に統一しており、各チャット実装で直接 `vault.rglob("*.md")` を呼ばないでください（検索品質と優先順位が崩れます）。

## Cloud Run / GCS Vault 対応

本番はGoogle Cloud Run上でAPI/Webサービスを分離して展開しています。

| サービス | 名称 | 役割 |
|---|---|---|
| API | `tune-lease-55-api` | FastAPI、スコアリング、紫苑対話 |
| Web | `tune-lease-55-web` | Next.js フロントエンド |

ローカルのObsidian Vaultの代わりに、クラウドでは**GCS Vault**（`scripts/gcs_vault_loader.py`が定期同期）を使います。同期対象は`リース知識`・`Projects/tune_lease_55/Research`・`News`・`Lease Intelligence/Public`など公開可能な知識のみで、`Daily`・`Private Reflection`・生チャット・回収ログは同期しません。

DBはSQLite（ローカル）とPostgreSQL（Cloud Run、`DATABASE_URL`で切替）の両対応です。ハッカソン用Cloud Runは`CLOUDRUN_DATA_MODE=demo` / `DB_PATH=/app/data/demo.db`で動かし本体DBを保護します。詳細フローは [DevOpsサイクルとしての紫苑](#devopsサイクルとしての紫苑) を参照してください。

```bash
CLOUDRUN_DATA_MODE=demo bash scripts/deploy_cloud_run.sh
python3 scripts/check_cloudrun_demo_readiness.py --base-url https://YOUR-CLOUDRUN-URL
python scripts/sync_cloudrun_inputs_from_gcs.py          # Cloud Run経験を隔離DBへ帰還
python scripts/promote_cloudrun_return_data.py           # dry-run
python scripts/promote_cloudrun_return_data.py --apply   # 承認済みだけ demo.db へ昇格
```

同期後の確認は `/cloudrun-return-review`（隔離DB内の承認のみ、本体`lease_data.db`へは直接書き込みません）。記憶・内省の継続性設計は `docs/cloudrun_memory_continuity_design.md` を参照してください。

シークレットはSecret Managerで管理します（`.env`・ソースコードへの直接記載は禁止）。

```bash
gcloud builds submit --config cloudbuild.yaml
```

## 審査ロジックの見方

主なAPI:

- `POST /api/score/full` — フル審査スコア
- `POST /api/gunshi/stream` — 軍師AIストリーミング
- `GET /api/lease-intelligence/dialogue/state` — 紫苑の状態
- `POST /api/lease-intelligence/dialogue` — 紫苑との対話
- `GET /api/shion/inner-state` — 知性体コア用の内面状態
- `GET /api/shion/central-synthesis` — world_view / 共有認識

`/api/lease-intelligence/dialogue` は汎用rewriteではなく `frontend/src/app/api/lease-intelligence/dialogue/route.ts` のRoute HandlerからFastAPIへ明示プロキシします（長時間POSTでの`socket hang up`回避、タイムアウト原因の可視化のため）。

### スコアリングモデルの現在地

本流の借手スコア: **既存先=RandomForest** / **新規先=LogisticRegression**。LR/RandomForest/LGBMを比較分析として併用しますが、画面上で「本流はLightGBM単体」と誤解される表現は避けます。

補助指標: `Q_risk`（財務データの矛盾・歪みを見る、自動減点ではなく深掘り対象）、類似案件、ニュース論点、軍師AI（審査部の反論・顧客確認・条件設計・稟議コメント）

## 開発メモ

よく使う確認:

```bash
python -m py_compile api/gunshi_gemini.py
python -m py_compile lease_intelligence_reflection.py
cd frontend && npx tsc --noEmit
npm run build
```

JSON を LLM へ長文で直接書かせると壊れやすいため、重要な出力は短い構造 JSON に寄せ、説明文は Python 側のテンプレートで生成します。

今の方針:

- 裁定役、ペルソナ、自己分析、ニュース要約、OCR は `codes + key_phrases` 型へ寄せる
- 財務 OCR は `detected_fields + confidence + missing_fields` で扱う
- リースファイナンス知識はコード上の正本に寄せ、システムプロンプトとの重複を避ける
- Obsidian 検索は共通経路を使う
- 対話AIの接続不調は、Gemini APIだけでなく Next Route Handler、FastAPI直叩き、Cloudflare経由を分けて確認する

## Git 運用

コード・設定・ドキュメント・テストをコミット対象にします。`data/`、一時キャッシュ、生成物、秘密情報は原則コミットしません（必要な場合のみ中身を確認して個別判断）。`git-ship`する時は差分からコミットメッセージを作成しpushまで行い、既存のユーザー変更は勝手に戻しません。

## プロジェクト構造

```text
api/                         FastAPI と審査 API
frontend/                    Next.js フロントエンド
mobile_app/                  Obsidian bridge など共通部品
scripts/                     運用・補修・GCS 同期スクリプト
reports/                     改善レポート、評価結果
memory/                      日次作業メモ
data/                        ローカル生成データ。原則 git 対象外
lease_intelligence_*.py      紫苑の自己モデル、対話、内省、central
run_next_stable.sh           主起動スクリプト（ローカル）
Dockerfile / Dockerfile.api  Cloud Run 向けコンテナ定義
cloudbuild.yaml              Cloud Build デプロイ設定
```

## このリポジトリの芯

これは「AI に審査を任せる」システムではありません。

人間が最後に判断するために、AI が根拠を集め、反論を出し、条件を考え、失敗を記憶するシステムです。紫苑はそのための記憶と人格を持つ知性体です。

使うほど、過去の判断が次の判断に戻ってくる。そこを一番大事にしています。
