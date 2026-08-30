# リポジトリ構造マップ（コードベース概観）

理由: リポジトリが個人運用ファイル（記憶・ハートビート）と2系統のアプリ実装（Next.js+FastAPI主系統 / Streamlit参照用）を含み、初見のAIアシスタントが入口を誤ると無関係なコードを触る・スコアリングの正本を誤認するリスクがあるため。
適用条件: 初めてこのリポジトリで作業する時、ディレクトリの役割が不明な時。
削除条件: リポジトリ構成が安定し、新規参加者向けオリエンテーションが別ドキュメントに一本化された時。

## 2つの実装系統

| 系統 | 状態 | エントリポイント |
|---|---|---|
| **Next.js + FastAPI**（主系統） | 現行・日常利用・外部公開 | フロント: `frontend/`（`npm run dev`）／API: `api/main.py`（`python3 -m uvicorn api.main:app --port 8000`）／両方まとめて起動: `./run_next_stable.sh` |
| **Streamlit**（参照用に残置） | レガシー・参照用 | ルート直下の `tune_lease_55.py` ほか `*.py` ＋ `components/`／起動: `make app` または `./run_streamlit_stable.sh` |

新規実装は特に指示がない限り Next.js + FastAPI 側を優先する（`AGENTS.md` Part 2参照）。`scoring_core.py` はスコアリングロジックの正本で両系統から参照される共有コア。

## ディレクトリ早見表

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

## 開発コマンド早見表

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
cd frontend && npx tsc --noEmit   # PR前必須（CLAUDE.md 絶対厳守セクション参照）
cd frontend && npm run lint       # eslint --fix は絶対禁止
python3 scripts/run_scoring_harness.py   # スコアリング導線の軽量スモークハーネス
```
