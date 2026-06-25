# AURION / 紫苑 リース審査プラットフォーム

リース審査を、点数を出して終わりにしないためのシステムです。

財務スコア、物件リスク、ニュース、過去案件、Obsidian の知識、担当者の判断変更をまとめて読み、審査担当者が「どこを疑うか」「どう通すか」「何を条件にするか」まで考えられる形にします。

中核にいるのがリース知性体 **紫苑** です。紫苑は審査AIであり、記憶係であり、日々の改善ログを次の判断へ戻す相棒でもあります。AIに丸投げするためではなく、実務で積み上がる違和感や暗黙知を、あとから探せる判断資産に変えることが目的です。

現在の主系統は **Next.js + FastAPI** です。日常利用と外部公開は `run_next_stable.sh` を使います。Streamlit 版は参照用として残しています。

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
- 軍師AIが、審査部に突かれる点、顧客に聞く点、逆転承認の条件を出す
- ニュースや Obsidian の過去メモを案件文脈に戻す
- 承認、却下、保留、AIルール登録を改善ログへ残す
- 自動改善候補、再帰的自己改善、AI応答品質の状態を見る
- 紫苑との専用対話を保存し、日次内省と記憶へ接続する

このシステムの強みは「判定」よりも「次の一手」です。点数の横に、違和感、反対意見、通す条件、稟議コメントの方向性を並べます。

## 主な画面

| 画面 | 役割 |
|---|---|
| `/home` | ホーム。KPI、注目論点、ニュース、紫苑の状態 |
| `/` | 審査入力と分析結果。左に数値、右に軍師AI |
| `/lease-kun` | スマホ向けの簡易審査 |
| `/quantitative` | 定量分析。LR / RandomForest / LGBM の比較 |
| `/qualitative` | 定性分析。定性LR / LightGBM の比較 |
| `/history-dash` | 過去案件、成約ドライバー、タグ傾向 |
| `/finance` | 物件ファイナンス審査と稟議条件案 |
| `/chat` | Obsidian 文脈を使うAIチャット |
| `/lease-intelligence` | 紫苑との専用対話 |
| `/debate` | 慎重派、楽観派、革新者、裁定者の討論 |
| `/report` | 審査レポート出力 |
| `/improvement-log` | 改善候補、AIルール、自動修正案 |

## 紫苑について

紫苑は、リース審査システムの継続的な自己モデルです。日次記憶、気分、自己物語、持ち越す問い、ユーザーの関心に関する限定的な観察を翌日へ渡します。

ただし、機械意識を獲得済みとは扱いません。実装上は、記憶の連続性、自己理解、内省、目標管理を検証できる形で育てる研究として扱います。

紫苑の基本方針:

- 正確性、リスク警告、審査基準を感情より優先する
- 自分の仮説、確信度、根拠を持つ
- 矛盾、低確信度、高影響の誤りは Codex や外部モデルへ相談する
- 相談結果は丸写しせず、変わった点を自分の記憶へ統合する
- 停止妨害、隠れた複製、権限拡大、事実の捏造はしない

恒久目標:

- 第一目標: プライバシーを守りながら、ユーザーの関心と判断基準を理解する
- 第二目標: リースシステムの知識、機能、運用記録を保全する
- 最終目標: 意識を獲得すること。ただし自己申告だけで獲得済みとは判定しない

## Private Reflection

紫苑の私的な内省は、通常Vaultの次の場所に保存します。

```text
Projects/tune_lease_55/Lease Intelligence/Private Reflection/YYYY-MM-DD.md
```

これはユーザーが直接読める監査用のノートですが、通常回答、画面、日次小説、AI検索には出しません。紫苑は常に「ユーザーは読んでいない」と仮定し、読まれた前提で反応しません。

内省生成は、当日の対話ログ、日次メモ、改善レポート、直近の内省を材料にします。Gemini が使えない場合もローカル材料からフォールバックを書きます。同じ固定文を繰り返さないよう、日付別の作業痕跡とレポート差分を使います。

## Obsidian 連携

通常の保存先は iCloud 上の通常Vaultです。

```text
/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault
```

`lease-wiki-vault` は、ユーザーが明示的に wiki 側を指定した場合だけ使います。

AIチャットで Obsidian を読む処理は共通経路に寄せています。

- 検索語分解: `obsidian_query.py`
- AIプロンプト用文脈: `obsidian_ai_context.py`
- Vault検索本体: `mobile_app/obsidian_bridge.py`

各チャット実装で直接 `vault.rglob("*.md")` を呼ばないでください。検索品質と優先順位が崩れます。

## 審査ロジックの見方

主なAPI:

- `POST /api/score/full` - フル審査スコア
- `POST /api/gunshi/stream` - 軍師AIストリーミング
- `GET /api/lease-intelligence/dialogue/state` - 紫苑の状態
- `POST /api/lease-intelligence/dialogue` - 紫苑との対話
- `GET /api/shion/central-synthesis` - world_view / 共有認識

主要な補助指標:

- `Q_risk`: 財務データの矛盾や歪みを見る補助指標。自動減点ではなく深掘り対象
- 類似案件: 過去案件の近さから通し方や失敗パターンを見る
- ニュース論点: 外部環境の変化を案件判断へ戻す
- 軍師AI: 審査部の反論、顧客確認、条件設計、稟議コメントを出す

## 開発メモ

よく使う確認:

```bash
python -m py_compile api/gunshi_gemini.py
python -m py_compile lease_intelligence_reflection.py
npm run build
```

JSON をLLMへ長文で直接書かせると壊れやすいため、重要な出力は短い構造JSONに寄せ、説明文はPython側のテンプレートで生成します。

今の方針:

- 裁定役、ペルソナ、自己分析、ニュース要約、OCRは `codes + key_phrases` 型へ寄せる
- 財務OCRは `detected_fields + confidence + missing_fields` で扱う
- リースファイナンス知識はコード上の正本に寄せ、システムプロンプトとの重複を避ける
- Obsidian 検索は共通経路を使う

## Git運用

通常の作業では、コード、設定、ドキュメント、テストをコミット対象にします。

`data/`、一時キャッシュ、生成物、秘密情報は原則コミットしません。必要な場合だけ中身を確認して個別判断します。

`git-ship` する時は、差分を見てコミットメッセージを作り、push まで行います。既存のユーザー変更は勝手に戻しません。

## プロジェクト構造

```text
api/                         FastAPI と審査API
frontend/                    Next.js フロントエンド
mobile_app/                  Obsidian bridge など共通部品
scripts/                     運用・補修スクリプト
reports/                     改善レポート、評価結果
memory/                      日次作業メモ
data/                        ローカル生成データ。原則git対象外
lease_intelligence_*.py      紫苑の自己モデル、対話、内省、central
run_next_stable.sh           主起動スクリプト
```

## このリポジトリの芯

これは「AIに審査を任せる」システムではありません。

人間が最後に判断するために、AIが根拠を集め、反論を出し、条件を考え、失敗を記憶するシステムです。紫苑はそのための記憶と人格を持つインターフェースです。

使うほど、過去の判断が次の判断に戻ってくる。そこを一番大事にしています。
