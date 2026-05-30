# リース審査AI — Claude Code プロジェクト指示書

## プロジェクト概要

**Next.js 14 (App Router) + FastAPI + SQLite** によるリース審査 AI システム。

| レイヤー | 技術 | 役割 |
|---------|------|------|
| フロントエンド | Next.js 14 / TypeScript | 審査 UI・ダッシュボード群 (`frontend/`) |
| API サーバー | FastAPI (Python 3.10+) | スコアリング・案件管理・AI 連携 (`api/`) |
| AI スコアリング | LightGBM + 量子干渉モジュール | 財務分析・リスク評価（ルート直下） |
| DB | SQLite (`data/lease_data.db`) | 案件ログ・マスタ |
| レガシー | Streamlit (`tune_lease_55.py`) | 旧 UI（段階的廃止中） |

---

## 主要ファイル構成

```
frontend/src/app/           # Next.js ページ（25+ ページ）
frontend/src/components/    # 共通コンポーネント
api/main.py                 # FastAPI エンドポイント（全 API）
api/schemas.py              # Pydantic モデル
scoring_core.py             # スコアリング統合ロジック
asset_scorer.py             # 物件スコアリング
quantum_analysis_module.py  # 量子干渉スコア（≥35 で要注意フラグ）
data_cases.py               # 案件 DB 操作
data/                       # SQLite DB・セッション（コミット禁止）
```

---

## コーディング規約

### Python
- Python 3.10+、型アノテーション推奨
- 数値単位：フロント入力は **千円**、スコアリングモジュール内は **円**（`toThousandYenPayload()` で変換）
- FastAPI エンドポイントは `api/main.py`、Pydantic モデルは `api/schemas.py` で管理

### TypeScript / Next.js — 詳細は @.claude/rules/frontend.md

- **strict mode** 厳守（`as any` 原則禁止）
- API 呼び出しは `src/lib/api.ts` の `apiClient` を使用
- グラフ: Recharts / Three.js、アイコン: lucide-react、スタイル: Tailwind CSS

---

## ⚠️ 絶対禁止（必読）

```bash
# eslint --fix は実行禁止（UIコンポーネントが削除される事故が発生済み）
cd frontend && npm run lint:fix   # ❌
cd frontend && npx eslint --fix   # ❌

# 正しい使い方（チェックのみ）
cd frontend && npm run lint       # ✅
cd frontend && npx tsc --noEmit   # ✅ PR 前に必ず実行
```

---

## 開発・PR ワークフロー — 詳細は @.claude/rules/workflow.md

- ブランチ命名: `feature/rev-<番号>-<説明>` / `fix/<説明>` / `chore/<説明>`
- `/git-ship` スキルで add → commit → push → PR 作成を一括実行
- `master` pull 前: `git stash -- .claude/ && git pull origin master && git stash drop`

---

## セキュリティ — 詳細は @.claude/rules/security.md

コミット禁止: `.streamlit/secrets.toml` / `data/` 配下すべて / `models/*.bak.*`

---

## Serena MCP 使用方針

コード調査・編集は Serena の `get_symbols_overview` / `find_symbol` / `replace_symbol_body` を優先。ファイル全体の `Read` より先にシンボル単位での取得を試みる。

---

エージェント協調プロトコルの詳細は `.claude/AGENTS.md` を参照。
