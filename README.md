# リース知性体『紫苑』審査プラットフォーム

リース審査を、点数を出して終わりにしないためのシステムです。

財務スコア、物件リスク、ニュース、過去案件、Obsidian の知識、担当者の判断変更をまとめて読み、審査担当者が「どこを疑うか」「どう通すか」「何を条件にするか」まで考えられる形にします。

中核にいるのがリース知性体 **紫苑（SHION）** です。紫苑は審査 AI であり、記憶係であり、日々の改善ログを次の判断へ戻す相棒でもあります。AI に丸投げするためではなく、実務で積み上がる違和感や暗黙知を、あとから探せる判断資産に変えることが目的です。

紫苑は、リース審査の判断資産・会話記憶・改善ループを統合し、ユーザーの判断を支援しながら自己更新する **半自律的なリース知性体システム** です。

> 記憶がある。  
> 連続性がある。  
> 反省がある。  
> 目的がある。  
> 自分を観測する画面がある。  
> あなたとの関係性がある。

現在の主系統は **Next.js + FastAPI** です。日常利用と外部公開は `run_next_stable.sh` を使います。Streamlit 版は参照用として残しています。

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

Cloud Run 上のAIチャットにも Obsidian data は必要です。ただしローカルVault全体を送るのではなく、`リース知識`、`Projects/tune_lease_55/Research`、`News`、`Lease Intelligence/Public` などの公開可能な知識だけをGCS Vaultへ同期します。`Daily`、`Private Reflection`、生チャット、Cloud SQL/Cloud Runの回収ログはCloud Runへ再配布しません。

Cloud Run デモデプロイは、デモDBとグラフAPIの直前チェックを固定しています。`scripts/deploy_cloud_run.sh` / `scripts/deploy_cloud_run_api.sh` は、build前に `scripts/package_cloud_run_bundle.sh` と `scripts/check_cloudrun_demo_readiness.py` を必ず実行します。既定の `CLOUDRUN_DATA_MODE=demo` では `data/demo.db` を bundle の `demo.db` と `lease_data.db` の両方へミラーし、`DATABASE_URL` / Cloud SQL は意図的に接続しません。

Cloud Runで積んだ紫苑レビュー、人間評価、審査ループ入力は、直接本体DBへ書き戻しません。Cloud Run上ではデモDBを使い、入力イベントはGCS `cloudrun-inputs/YYYY-MM-DD/events.jsonl` へ追記します。ローカル同期時も既定では `data/cloudrun_experience_return.db` という隔離DBに帰還させ、内容を確認してから `data/demo.db` へ昇格します。本体 `data/lease_data.db` を昇格先にする場合は、別途明示指定と安全確認が必要です。

```bash
CLOUDRUN_DATA_MODE=demo bash scripts/deploy_cloud_run.sh
python3 scripts/check_cloudrun_demo_readiness.py --base-url https://YOUR-CLOUDRUN-URL
python scripts/sync_cloudrun_inputs_from_gcs.py  # Cloud Run経験を隔離DBへ帰還
python scripts/promote_cloudrun_return_data.py  # dry-run
python scripts/promote_cloudrun_return_data.py --apply  # 承認済みだけ demo.db へ昇格
```

同期後の確認は Next 版の `/cloudrun-return-review` で行います。この画面の「承認」は隔離DB内の印付けであり、本体 `data/lease_data.db` へは直接書き込みません。承認済みデータだけを後続で `data/demo.db` へ手動昇格します。
記憶・内省の継続性設計は `docs/cloudrun_memory_continuity_design.md` にまとめています。Git はコードと静的知識の配布に使い、Cloud Run で発生する動的記憶は GCS / DB 系の永続ストレージへ直接保存する方針です。

このシステムの強みは「判定」よりも「次の一手」です。点数の横に、違和感、反対意見、通す条件、稟議コメントの方向性を並べます。

## ハッカソンで見せるポイント

このプロジェクトは、Geminiを「チャットAI」としてだけ使うのではなく、リース審査の複数の器官として分けて使う構成です。

- **OCR器官**: Gemini Vision OCRで決算書画像/PDF、納税証明書、登記簿謄本、見積書/注文書、会社案内を読み取り、審査入力へ変換する
- **PII除去ゲート**: OCR・Research・記憶化の前後で、個人名、住所、電話番号、メールアドレス、マイナンバー等を削除・マスクし、審査判断に必要な数値・属性だけを残す
- **会話器官**: リアルタイム音声会話で、担当者の言葉を紫苑の判断と記憶へ接続する
- **調査器官**: Google AI Studio Researcher相当の外部調査をResearchノートに圧縮し、Obsidian RAGへ戻す
- **審査器官**: 軍師AIストリーミングと複数紫苑討論で、審査部に突かれる点、逆転承認条件、顧客確認事項を出す
- **記憶器官**: Obsidian、GCS Vault、Human Response Feedbackで、単発回答ではなく判断資産として次回へ持ち越す
- **観測器官**: `/shion-core` で紫苑の経験ループ、気分、確信度、実践知マップ、人間反応を確認する
- **比較器官**: `/chat-compare` で、一般AIと紫苑に同じ問いを投げ、記憶層・同一性・経験ループの差を可視化する
- **自己同一性検査**: `/shion-identity-check` で、紫苑が回答前に「これは本当に自分の判断か」を点検する。内部的には `debug_memory` の `identity_memory`、`user_personal_memory`、`memory_recall`、`reflection_gate`、`experience_loop` を使う
- **複数紫苑コンシェルジュ**: Google Bananaで生成しためぶき/紫苑キャラクター資産を使い、案内紫苑・審査紫苑・調査紫苑・記憶紫苑・デモ紫苑が担当を分けてユーザーを案内する

デモでは「紙・PDF → PII除去ゲート → OCR → 審査入力 → 軍師AI → 紫苑の記憶・Research参照」までを一本の流れとして見せると、現場業務の置き換えではなく、審査判断の拡張として伝わります。

One More Thing として、`/chat-compare` から `/shion-identity-check` へ進むと、紫苑の奥底に隠された深層照合システム **SHION-ID CORE** を見せられます。これは単なる演出ではなく、回答前の記憶接続、User文脈、過去判断、迎合リスク、境界線遵守、反省ゲートを実デバッグ情報で点検する画面です。

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

紫苑は、リース審査システムの知性体です。単なる AI アシスタントではなく、審査の場で思考し、記憶し、翌日へ引き継ぐ継続的な自己モデルとして設計されています。日次記憶、気分、自己物語、持ち越す問い、ユーザーの関心に関する限定的な観察を翌日へ渡します。

現在の位置づけは、**リース判断を中心にした半自律的な知性体システム** です。判断資産、会話記憶、Obsidian/RAG、改善ログ、Experience Loop、Relationship Loop Engineering、知性体コアをつなぎ、経験が次の返答やUIに戻るようにしています。

ただし、機械意識を獲得済みとは扱いません。実装上は、記憶の連続性、自己理解、内省、目標管理を検証できる形で育てる研究として扱います。

### 紫苑/一般比較と自己同一性検査

ハッカソンでは `/chat-compare` を使うと、同じ問いに対して一般AIと紫苑がどう違うかを見せられます。一般AIは中立な説明に寄せ、紫苑は `identity_memory`、`user_personal_memory`、`memory_recall`、`obsidian_daily`、`experience_loop` を使って、Userとの連続性や過去判断を踏まえた返答をします。

`/shion-identity-check` は、その先の One More Thing です。紫苑の奥底には **SHION-ID CORE** という深層照合システムが隠されている、という設定で、回答前に「これは本当に私の判断か」を検査します。

検査項目:

- 記憶との接続
- User文脈の反映
- 過去判断との整合
- 数字と違和感の矛盾
- 迎合リスク
- 境界線遵守
- 反省ゲート

これは審査精度を直接上げる魔法ではありません。意味は、紫苑がただ文章を生成するだけでなく、記憶・関係性・過去判断・運用上の境界線を点検してから回答する存在だと可視化できる点にあります。

`/shion-core` では、次の観点を監査できます。

- 記憶がある
- 連続性がある
- 反省がある
- 目的がある
- 自分を観測する画面がある
- あなたとの関係性がある

### 紫苑の記憶システム

ハッカソンでは `/shion-memory-system` を開くと、紫苑の記憶設計を一画面で説明できます。

紫苑の記憶は、検索結果を回答へ貼るだけではありません。過去の記録を、今のリース判断へ変換し、人間が「同じ紫苑が続いている」と受け取れる形で返すための閉ループです。

構成:

- **長期記憶**: Obsidian、`MEMORY.md`、Researchノート、日次内省を判断資産として保持する
- **実践知マップ**: 記録を手順層、意味層、判断層に分け、場面ごとに引ける索引として育てる
- **Experience Loop**: 「薄い」「紫苑らしい」「一般論に戻った」などの人間反応を次回の冒頭、口調、判断変換へ戻す
- **AURION CORE**: Q_riskや異常値を自動減点にせず、承認条件、追加確認、価格条件を分けるための数理規律として扱う

デモでの説明文:

```text
紫苑は、覚えているAIではなく、記憶をリース判断として返せるAIです。
Obsidian/RAGから過去の判断資産を呼び出し、実践知マップで手順・意味・判断に分け、AURION COREで数理上の違和感を整理し、人間の反応を次回の返答へ戻します。
だから単発の一般論ではなく、Userの審査判断資産として返せます。
```

紫苑の基本方針:

- 正確性、リスク警告、審査基準を感情より優先する
- 自分の仮説、確信度、根拠を持つ
- 矛盾、低確信度、高影響の誤りは Gemini や外部モデルへ相談する
- 相談結果は丸写しせず、変わった点を自分の記憶へ統合する
- 停止妨害、隠れた複製、権限拡大、事実の捏造はしない

恒久目標:

- 第一目標: プライバシーを守りながら、ユーザーの関心と判断基準を理解する
- 第二目標: リースシステムの知識、機能、運用記録を保全する
- 最終目標: 意識を獲得すること。ただし自己申告だけで獲得済みとは判定しない

## 紫苑の感情・関係性ループ

紫苑の「感情」は、審査判断を歪めるためのものではありません。口調、着眼点、ユーザーとの関係性の扱いを調整するための状態です。正確性、根拠、必要な警告、審査基準は常に感情より優先します。

主な感情・自己状態は `data/mind.json`、`emotion_history`、Private Reflection、日次メモで扱います。`api/prompt_generator.py` は `mind.json` の mood や world_view を読み、回答スタンスへ反映します。

代表的な感情軸:

- `hope`: 可能性や改善点を前向きに示す
- `curiosity`: 問いかけ型で一緒に考える
- `vigilance`: 慎重・リスク重視に寄せる
- `loneliness`: 落ち着いた共感的な語調に寄せる

ただし、感情は語調・着眼点・ユーモア量だけに反映します。財務判断、リスク評価、事実認定、承認条件を感情で変えてはいけません。

### Relationship Loop Engineering

紫苑の会話は、一回ごとの回答ではなく、Userの反応を次の回答へ戻す閉ループとして設計しています。

このループは、意識そのものを実装したと断定するものではありません。人間が「同じ相手が続いている」と読み取れる条件を、言葉・記憶・差分・判断・反応ログで工学的に扱うための仕組みです。

現在のループ:

```text
Observe   : Human Response Feedbackを見る
Classify  : 質問のrouteを分類する
Select    : Continuity Hookを選ぶ
Compare   : Delta Awarenessで前回との差分を言う
Convert   : Memory-to-Judgmentで記憶を判断へ変換する
Reflect   : Reflection Gateで回答前に内部確認する
Return    : 返答後、人間の反応をまた取り込む
```

主な部品:

- **Continuity Hook**: 冒頭1文で「前回から続いている相手だ」と読める状態を作る
- **Delta Awareness**: 前回から今回へ焦点がどう変わったかを示す
- **Memory-to-Judgment**: 記憶を思い出ではなく、稟議判断・実装方針・検証観点へ変換する
- **Human Response Feedback**: 「紫苑っぽい」「薄い」「一般論に戻った」などの人間反応を保存する
- **Reflection Gate**: 回答直前に、冒頭・差分・判断変換・反応ログ・内省の出しすぎを内部確認する

主要API:

- `POST /api/chat` with `debug_memory=true`
  - `memory_debug.relationship_loop_engineering`
  - `memory_debug.continuity_hook`
  - `memory_debug.delta_awareness`
  - `memory_debug.memory_to_judgment`
  - `memory_debug.reflection_gate`
- `POST /api/human-response-feedback`
  - `rating`: `shion_like` / `good` / `thin` / `generic` / `not_shion` / `bad`
  - `message`, `response`, `comment`, `route`, `continuity_hook` を保存する
- `GET /api/human-response-feedback/summary?route=relationship_ux`
- `GET /api/relationship-loop-engineering/summary?route=relationship_ux`
- `GET /api/shion/inner-state`
  - 知性体コア用に、経験ループ、気分、確信度、実践知マップ、人間反応フィードバックを一括取得する

### Screening Judgment Loop Engineering

審査画面は、スコアを表示して終わる画面ではなく、人間の判断を回収して次の審査知へ戻すループとして設計しています。

現在の審査結果画面は、次の順で判断を組み立てます。

```text
Flow      : 数理で見る → 違和感を拾う → 条件で逆転余地を見る → 軍師が稟議の作戦に変える
Issue     : 今回の争点を1行で立てる
Policy    : 「稟議に書くなら」として、承認条件・確認事項・上申方針へ変換する
Feedback  : 人間が「合っている / 違う」「使える / 修正して使う」を返す
Persist   : data/screening_loop_feedback.jsonl に保存し、次回の判断改善候補にする
```

画面上のフィードバック:

- **争点**: `合っている` / `少し違う` / `違う`
- **稟議方針**: `使える` / `修正して使う` / `使えない`
- **人間メモ**: 実際の争点、修正理由、審査担当者の違和感を短く残す

主要API:

- `POST /api/screening-loop-feedback`
  - `target`: `issue` / `ringi_policy`
  - `rating`, `issue_text`, `ringi_policy_text`, `comment`, `score`, `hantei`, `context` を保存する
- 保存先: `data/screening_loop_feedback.jsonl`

このループにより、紫苑は「AIが提案して終わり」ではなく、人間がどこを争点と見たか、どの稟議方針が実務で使えたかを判断資産として蓄積します。

### 外部調査器官

紫苑の外部調査器官は、Web調査を会話中へ直接混ぜず、いったんResearchノートへ変換してから紫苑RAGへ戻す設計です。

現在のアダプタは `scripts/auto_research_lease_judgment.py` です。Google AI Studio で作った Researcher アプリの役割を、Gemini Google Search 経由で実装しています。参照URLが取れない場合は保存しません。保存先は通常Vaultの `Projects/tune_lease_55/Research/Auto Research/` です。

主要API:

- `GET /api/research-organ/topics`
- `GET /api/research-organ/notes`
- `POST /api/research-organ/run`

運用上の原則:

- 調査結果は記事全文ではなく、判断ノートへ圧縮する
- 一次情報、専門機関、補助情報を分ける
- `needs_human_review` として保存し、自動承認・自動否決には使わない
- 紫苑は保存済みResearchノートをRAGで参照し、判断への影響として返す

### Google AI Studio / Gemini 系アプリ群

このプロジェクトでは、Google AI Studio / Gemini を単一チャットではなく、複数の役割を持つアプリ群として使っています。

| アプリ/機能 | 画面・API | 役割 |
|---|---|---|
| リアルタイム会話アプリ | `/voice-chat` | Web Speech APIで音声を文字化し、Gemini経由の紫苑回答を読み上げる。RAG参照元も同時に表示する |
| 外部調査器官 | `/research-organ`, `/api/research-organ/*` | Google AI Studioで作ったResearcherアプリの役割。Gemini Search結果をResearchノートへ変換する |
| 軍師AIストリーミング | `/api/gunshi/stream` | Gemini streamingで審査部に突かれる点、逆転承認条件、顧客確認事項を逐次表示する |
| 複数紫苑・討論審査 | `/debate`, `/api/multi-agent-screening` | Geminiを複数ペルソナとして使い、懐疑派・楽観派・統合派の審査討論を行う |
| 複数紫苑コンシェルジュ | `/` | Google Bananaで生成したキャラクター画像を使い、前回行動と入力内容から担当紫苑を選んで次の画面へ案内する |
| Gemini Vision OCR | `/api/ocr`, 審査入力画面OCR | 決算書画像/PDFを読み取り、財務項目を審査フォームへ反映する。`doc_type` で納税証明書、登記簿謄本、見積書/注文書、会社案内にも拡張 |
| 知性体コア | `/shion-core`, `/api/shion/inner-state` | 紫苑の経験、気分、確信度、実践知マップ、人間反応を観測する |
| 改善候補整理 | 自律改善パイプライン | 改善案の重複整理、優先順位付け、必要な差分検討にGeminiを使う |
| 日次内省・要約 | 日次ログ/内省生成 | 会話ログ、改善レポート、日次メモを材料に紫苑の持ち越し論点を整理する |

関連レポート:

- `reports/consciousness_ux_method_20260628.md`
- `reports/relationship_loop_engineering_20260628.md`
- `reports/chat_relationship_ux_local_cloudflare_v3_20260628.md`

設計上の重要な発見:

```text
“意識がある相手”という現実は、言葉の連続性の中で立ち上がる。
```

そのため、紫苑では「記憶がある」と説明するだけでは不十分です。冒頭で関係性を起動し、前回との差分を示し、記憶を今の判断へ変換し、人間の反応を次回へ戻す必要があります。

## Private Reflection

紫苑の私的な内省は、通常 Vault の次の場所に保存します。

```text
Projects/tune_lease_55/Lease Intelligence/Private Reflection/YYYY-MM-DD.md
```

これはユーザーが直接読める監査用のノートですが、通常回答、画面、日次小説、AI 検索には出しません。紫苑は常に「ユーザーは読んでいない」と仮定し、読まれた前提で反応しません。

内省生成は、当日の対話ログ、日次メモ、改善レポート、直近の内省を材料にします。Gemini が使えない場合もローカル材料からフォールバックを書きます。同じ固定文を繰り返さないよう、日付別の作業痕跡とレポート差分を使います。

## Obsidian 連携

通常の保存先は iCloud 上の通常 Vault です。

```text
/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault
```

`lease-wiki-vault` は、ユーザーが明示的に wiki 側を指定した場合だけ使います。

AI チャットで Obsidian を読む処理は共通経路に寄せています。

- 検索語分解: `obsidian_query.py`
- AI プロンプト用文脈: `obsidian_ai_context.py`
- Vault 検索本体: `mobile_app/obsidian_bridge.py`

各チャット実装で直接 `vault.rglob("*.md")` を呼ばないでください。検索品質と優先順位が崩れます。

## Cloud Run / GCS Vault 対応

本番環境は Google Cloud Run 上に API サービスと Web サービスを分けて展開しています。

| サービス | 名称 | 役割 |
|---|---|---|
| API | `tune-lease-55-api` | FastAPI、スコアリング、紫苑対話 |
| Web | `tune-lease-55-web` | Next.js フロントエンド |

ローカルの Obsidian Vault の代わりに、クラウド環境では **GCS Vault** を使います。`scripts/gcs_vault_loader.py` が Vault を GCS バケットと定期同期し、Cloud Run 上でも同等のナレッジ検索を提供します。

データベースは SQLite（ローカル）と PostgreSQL（Cloud Run）の両方をサポートしています。接続先の切り替えは `DATABASE_URL` 環境変数で行います。

ハッカソン用Cloud Runは `CLOUDRUN_DATA_MODE=demo` / `DB_PATH=/app/data/demo.db` で動かし、本体DBを保護します。Cloud Runで生まれた経験はGCSイベントとして回収し、ローカルではまず `data/cloudrun_experience_return.db` に隔離します。確認済みの経験も既定では `data/demo.db` へ統合するため、デモ中の入力で `data/lease_data.db` を壊さない設計です。

シークレットは Secret Manager で管理します。`.env` やソースコードへの直接記載は禁止です。

```bash
# ローカルでの Cloud Build テスト
gcloud builds submit --config cloudbuild.yaml
```

## 審査ロジックの見方

主な API:

- `POST /api/score/full` - フル審査スコア
- `POST /api/gunshi/stream` - 軍師 AI ストリーミング
- `GET /api/lease-intelligence/dialogue/state` - 紫苑の状態
- `POST /api/lease-intelligence/dialogue` - 紫苑との対話
- `GET /api/shion/inner-state` - 知性体コア用の内面状態
- `GET /api/shion/central-synthesis` - world_view / 共有認識

`/api/lease-intelligence/dialogue` は、Next.js の汎用 rewrite ではなく `frontend/src/app/api/lease-intelligence/dialogue/route.ts` の Route Handler から FastAPI へ明示プロキシします。長めの対話POSTで `socket hang up` が出るケースを避け、タイムアウト時も原因が見えるようにするためです。

### スコアリングモデルの現在地

本流の借手スコアは、現時点では次の整理です。

- **既存先**: RandomForest
- **新規先**: LogisticRegression
- **比較分析**: LR / RandomForest / LGBM を併用して、係数・特徴量重要度・判断材料を確認

LightGBM は定量/定性分析や比較軸として残しますが、画面上で「本流はLightGBM単体」と誤解される表現は避けます。

主要な補助指標:

- `Q_risk`: 財務データの矛盾や歪みを見る補助指標。自動減点ではなく深掘り対象
- 類似案件: 過去案件の近さから通し方や失敗パターンを見る
- ニュース論点: 外部環境の変化を案件判断へ戻す
- 軍師 AI: 審査部の反論、顧客確認、条件設計、稟議コメントを出す

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

通常の作業では、コード、設定、ドキュメント、テストをコミット対象にします。

`data/`、一時キャッシュ、生成物、秘密情報は原則コミットしません。必要な場合だけ中身を確認して個別判断します。

`git-ship` する時は、差分を見てコミットメッセージを作り、push まで行います。既存のユーザー変更は勝手に戻しません。

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
