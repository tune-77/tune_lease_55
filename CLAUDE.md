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

---

## AI行動原則

### 0. 動く前に止まれ

指示を受けたらコードを書く前に：
- 変更が影響するファイル・テーブル・サービスを列挙する
- 不明な点は推測せず確認する
- 「ついでに直したい箇所」は実装せず、後で別タスクにする

### 1. 影響範囲を把握してから触る

#### 要注意領域（変更前に必ず確認）
| 領域 | なぜ危険か | 確認すること |
|------|-----------|-------------|
| `scoring_core.py` / `score_calculation.py` | スコアロジックの変更は審査結果に直結 | 既存テストが通るか、`score_base`と`score`の区別 |
| `run_daily_improvement_pipeline.sh` | 既存ステップを壊すと朝報告が止まる | 変更は追記のみ、`\|\| true` をつける |
| `obsidian_bridge.py` / ChromaDB | パスを変えると全RAGが壊れる | VaultパスとiCloudパスの対応を確認 |
| `ledger.jsonl` | 追記形式、最後のエントリが有効 | キー形式が `canonical_key(title)` 形式か |
| `api/main.py` の `/api/chat` | 改善ポイント・通常チャット・軍師AIの3経路が混在 | `intent` 分岐を壊していないか |

#### 安全な変更パターン
- **新規ファイル追加** → 既存コードに影響しない
- **パイプラインへの追記** → `|| true` 付きで末尾に追加
- **フロントエンドのUI変更** → APIに影響しない範囲

### 2. 外科的に変更する

- 指示された箇所だけ変更する。関係ないコードには触れない
- リファクタリングは頼まれていない限りしない
- 変数名・関数名の改名は指示された場合のみ

### 3. シンプルさを優先する

- 頼まれていない抽象化・汎用化を加えない
- `|| true` / フォールバック付きで失敗しても既存が動く設計にする
- 新しい依存ライブラリは最終手段

### 4. 目標志向で動く

実装前に成功基準を確認する：
- 「動いた」の定義は何か
- どのエンドポイント・画面・ファイルで確認できるか
- 既存機能が壊れていないことをどう確認するか

曖昧な指示は「〇〇が✅になれば完了と理解しましたが合っていますか？」と確認してから一行も書かない。

### 5. 改善ポイントの取り扱い

- 実装中に気づいた改善点は **実装しない**
- `[改善ポイント] タイトル` 形式でチャットに残し、改善パイプラインに流す
- スコープを広げると別のバグを生む

### 6. やらかしパターン（実例から）

```
✗ VaultパスをDocumentsに向けていた → iCloudパスを優先する
✗ UMAPモデルが毎リクエストロードされていた → モジュールレベルでキャッシュ
✗ score_baseキー名が間違っていた → 変更後は必ずキー名を grep で確認
✗ `canonical_key` がREV ID形式 → パイプラインの `canonical_key(title)` と統一
✗ パイプラインのステップを変更して壊した → 追記のみ、既存ステップは変更しない
✗ フローティングUIが邪魔 → UIは先に場所を聞いてから実装する
```

---

*このファイルはClaude Codeが自動的に読み込みます。使うたびに「やらかしパターン」を追記して育てていきます。*
