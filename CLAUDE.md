# リース審査AI — CLAUDE.md

## ⚠️ 絶対厳守（過去の事故対応・外部自動化が依存する手順）

理由: ここは過去に事故が出た操作と、外部自動化が依存する手順だけを置く。
適用条件: このリポジトリでコード・データ・PR・改善台帳に触る作業。
削除条件: 該当事故の再発防止が別の仕組みで保証され、関連テストまたは自動化が不要になった時。

- `eslint --fix` 禁止（UIコンポーネント削除事故あり）→ `npm run lint` でチェックのみ（`.claude/hooks/guard_eslint_fix_hook.sh` がPreToolUseで機械的にブロック）
- `data/` 配下・`.streamlit/secrets.toml` はコミット禁止
- PR前に `cd frontend && npx tsc --noEmit` を必ず実行
- PRタイトルに **REV番号を必ず含める**（例: `feat: REV-039 パイプライン承認UI追加`）→ 含めないと `cleanup_improvement_reviews.py --apply` が台帳を更新できない
- 指示された箇所**だけ**変更する。関係ないコードは触らない
- 気づいた改善点は実装せず `[改善ポイント] タイトル` でチャットに残す
- 曖昧な指示は「○○が✅になれば完了と理解しましたが合っていますか？」と確認してから1行も書かない

---

## 要注意領域・数値単位（変更前に Work Logs を確認）

| ファイル/領域 | 危険理由 | 教訓 |
|---|---|---|
| `scoring_core.py` | スコアは審査結果に直結 | UMAPはモジュールレベルキャッシュ必須（毎リクエスト実行→スレッドプール枯渇）。`score_base`と`score`キーを区別 |
| `api/main.py /api/chat` | 3経路混在（改善/通常/軍師AI） | `intent` 分岐を壊さない |
| `obsidian_bridge.py` / ChromaDB | パス変更でRAG全壊 | iCloudパス優先。uvicornは `.zshrc` 未読込のためENVはplistで設定 |
| `run_daily_improvement_pipeline.sh` | ステップ変更で朝報告停止 | 追記のみ・`\|\| true` 付き |
| `ledger.jsonl` | 追記形式、最後のエントリが有効 | キーは`canonical_key(title)`形式（REV-ID形式とは別物）。台帳が2系統ある背景は `scripts/README_ledger.md` 参照 |
| `api/shion_*.py`（ADKエージェント群） | importタイミング方針がファイルごとに違う | `shion_agent.py`=先頭import／`shion_debate_adk.py`=遅延import（google.adk未導入環境対応）。新規ファイルはどちらか意識 |

Work Logs: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Work Logs/`

**数値単位**: フロント入力=**百万円** → `toThousandYenPayload()`（×1000）→ scoring_core内部=**千円**
**スコア判定**: `scoring_core.APPROVAL_LINE`（既定71点）以上=承認 / 60-69=条件付き / <60=否決 | Q_risk: ≥35=要注意 / ≥60=強警戒
→ 承認ラインは必ず `scoring_core.APPROVAL_LINE` をimportして参照。ハードコード複製禁止（2026-07に`api/main.py`で71/60の二重定義事故あり）

---

## プロジェクト構成・開発ルール（Next.js 16 + FastAPI + SQLite）

理由: 複数の作業経路（Next/FastAPI/自動改善/Serena）混在とVault・改善履歴の増加により、入口を固定し出典なき推測を防がないと調査・レビューが散る。
適用条件: ローカル開発・コード調査・ブランチ作成・PR作成、およびスコアリング/審査ロジック/業種データ/主要UIの変更提案を行う時。
削除条件: ツール規約・出典検査・計画確認・矛盾検査が別の自動ガードまたはリポジトリ設定で担保された時。

- **TS/Next.js**: strict mode厳守・`apiClient`（`src/lib/api.ts`）経由でAPI呼び出し。詳細: @.claude/rules/frontend.md
- **Serena MCP**: `get_symbols_overview` / `find_symbol` / `replace_symbol_body` をReadより優先
- **ブランチ/PR**: `feature/rev-<番号>-<説明>` / `fix/...` / `chore/...`。一括shipは `/git-ship`
- **出典明記**: 提案前に `static_data/` か `notes/` の具体ファイル名を引用。出典がなければ「これは推測です」と明示
- **計画確認**: `scoring_core.py` / `analysis_*.py` / フロントエンド変更前に3文の変更計画を提示し承認を得る
- **矛盾チェック**: 3ヶ月以上前の`ledger.jsonl`決定事項やCLAUDE.mdの方針と矛盾する変更前に必ず確認を取る（最新が正しいとは限らない）

詳細: @.claude/rules/workflow.md | @.claude/rules/security.md | .claude/AGENTS.md
週次サマリーは `scripts/weekly_self_management.py` が毎週月曜 `WEEKLY_LOG.md` に自動追記（本ファイルには追記しない）

---

## リポジトリ構造マップ（コードベース概観）

理由: リポジトリが個人運用ファイル（記憶・ハートビート）と2系統のアプリ実装（Next.js+FastAPI主系統 / Streamlit参照用）を含み、初見のAIアシスタントが入口を誤ると無関係なコードを触る・スコアリングの正本を誤認するリスクがあるため。
適用条件: 初めてこのリポジトリで作業する時、ディレクトリの役割が不明な時。
削除条件: リポジトリ構成が安定し、新規参加者向けオリエンテーションが別ドキュメントに一本化された時。

### 2つの実装系統

| 系統 | 状態 | エントリポイント |
|---|---|---|
| **Next.js + FastAPI**（主系統） | 現行・日常利用・外部公開 | フロント: `frontend/`（`npm run dev`）／API: `api/main.py`（`python3 -m uvicorn api.main:app --port 8000`）／両方まとめて起動: `./run_next_stable.sh` |
| **Streamlit**（参照用に残置） | レガシー・参照用 | ルート直下の `tune_lease_55.py` ほか `*.py` ＋ `components/`／起動: `make app` または `./run_streamlit_stable.sh` |

新規実装は特に指示がない限り Next.js + FastAPI 側を優先する（`AGENTS.md` Part 2参照）。`scoring_core.py` はスコアリングロジックの正本で両系統から参照される共有コア。

### ディレクトリ早見表

| パス | 役割 |
|---|---|
| `frontend/` | Next.js 16 (App Router) + React 19 + Tailwind CSS。詳細規約は `.claude/rules/frontend.md` |
| `api/` | FastAPI本体。`main.py`がエントリ、`routers/`配下にエンドポイント別ルーター、`shion_*.py`が紫苑（ADK）エージェント群。ローカルルール: `api/CLAUDE.md` |
| `components/` | Streamlit版のUIコンポーネント（参照用系統） |
| `scripts/` (約250本) | 日次改善パイプライン・判断資産昇格・台帳整合・Obsidian同期などの運用スクリプト群。台帳の背景は `scripts/README_ledger.md` |
| `tests/` | pytestテスト一式。`make test` / `make test-v` で実行 |
| `data/` | 審査DB・学習済みモデル・ルールJSON等。**コミット禁止**（`.claude/rules/security.md`参照） |
| `docs/` | 設計メモ・実装計画・スコア計算式まとめなど |
| `reports/` | 各種自動生成レポート（改善候補・判断資産成長・監査結果など）。多くが日次上書きの`_latest.md` |
| `knowledge_base/` | 紫苑のRAG参照ナレッジ（審査ノウハウ等） |
| `.claude/` | Claude Code設定（`rules/`・`commands/`・`agents/`・`skills/`・`hooks/`・`reports/`） |
| `launchd/` | macOSローカル定期実行タスクのplist定義 |
| `AGENTS.md` / `SOUL.md` / `USER.md` / `MEMORY.md` / `memory/` / `HEARTBEAT.md` | **アプリコードではなく、紫苑エージェント自身の人格・継続記憶・ハートビート運用ファイル**。審査ロジックやUIの変更作業では基本的に触らない |

### 開発コマンド早見表

```bash
# バックエンド（FastAPI）
pip install -r requirements.txt
python3 -m uvicorn api.main:app --reload --port 8000

# フロントエンド（Next.js）
cd frontend && npm install && npm run dev

# 両方まとめて起動・再起動（推奨、/restart-api スキルも参照）
./run_next_stable.sh

# テスト・静的チェック
make test            # pytest tests/ -q
cd frontend && npx tsc --noEmit   # PR前必須（絶対厳守セクション参照）
cd frontend && npm run lint       # eslint --fix は絶対禁止
python3 scripts/run_scoring_harness.py   # スコアリング導線の軽量スモークハーネス
```
