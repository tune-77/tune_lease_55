# リース審査AI — CLAUDE.md

## ⚠️ 最優先ルール

### 絶対禁止
- `eslint --fix` 禁止（UIコンポーネント削除事故あり）→ `npm run lint` でチェックのみ
- `data/` 配下・`.streamlit/secrets.toml` はコミット禁止
- PR前: `cd frontend && npx tsc --noEmit` 必須

### スコープ厳守
- 指示された箇所**だけ**変更する。関係ないコードは触らない
- 実装中に気づいた改善点は実装せず `[改善ポイント] タイトル` でチャットに残す
- 曖昧な指示は「○○が✅になれば完了と理解しましたが合っていますか？」と確認してから1行も書かない

### PR命名（自動化に必須）
PRタイトルに **REV番号を必ず含める** 例: `feat: REV-039 パイプライン承認UI追加`
→ 含めないと `cleanup_improvement_reviews.py --apply` が台帳を更新できない

---

## 要注意領域（変更前に Work Logs を確認）

| ファイル/領域 | 危険理由 | 注意点・やらかし教訓 |
|---|---|---|
| `scoring_core.py` | スコアは審査結果に直結 | UMAPはモジュールレベルキャッシュ必須（毎リクエスト実行→スレッドプール枯渇）。`score_base`と`score`キーを区別 |
| `api/main.py /api/chat` | 3経路混在（改善/通常/軍師AI） | `intent` 分岐を壊さない |
| `obsidian_bridge.py` / ChromaDB | パス変更でRAG全壊 | iCloudパス優先。uvicornは `.zshrc` を読まないので ENV は plist で設定 |
| `run_daily_improvement_pipeline.sh` | ステップ変更で朝報告停止 | 追記のみ・`\|\| true` 付き |
| `ledger.jsonl` | 追記形式、最後のエントリが有効 | キーは必ず `canonical_key(title)` 形式（CLI の REV-ID 形式とは別物） |

Work Logs: `~/Documents/Obsidian Vault/Projects/tune_lease_55/Work Logs/`

---

## 数値単位（バグの温床）

フロント入力: **千円** → `toThousandYenPayload()` → スコアリングモジュール内: **円**

スコア判定: ≥70=承認 / 60-69=条件付き / <60=否決 | Q_risk: ≥35=要注意 / ≥60=強警戒

---

## プロジェクト構成

**Next.js 14 + FastAPI + SQLite**

```
frontend/src/app/     # UI（25+ ページ）
api/main.py           # FastAPI エンドポイント（全API）
api/schemas.py        # Pydantic モデル
scoring_core.py       # スコアリング（LightGBM + 量子干渉）
data/lease_data.db    # SQLite（コミット禁止）
```

---

## 規約・ツール

- **TS/Next.js**: strict mode厳守・`apiClient`（`src/lib/api.ts`）経由でAPI呼び出し。詳細: @.claude/rules/frontend.md
- **Serena MCP**: `get_symbols_overview` / `find_symbol` / `replace_symbol_body` を優先（Read より先）
- **ブランチ**: `feature/rev-<番号>-<説明>` / `fix/...` / `chore/...`
- **一括ship**: `/git-ship` で add→commit→push→PR作成

詳細: @.claude/rules/workflow.md | @.claude/rules/security.md | .claude/AGENTS.md

---

## Freshman Rules（Vault成長に伴う品質維持）

### Cite the Source
スコアリング・審査ロジック・業種データに関する提案を出す前に、必ず
`static_data/` または `notes/` の特定ファイル名を引用すること。
Vault に出典がない場合は「これは推測です」と明示する。

### Plan-First Checkpoint
`scoring_core.py` / `analysis_*.py` / フロントエンドコンポーネントを
変更し始める前に、CLAUDE.md と関連ファイルを読み、
3文の変更計画を提示して承認を得てから着手すること。

### Kill the Assumptions
3ヶ月以上前の `ledger.jsonl` の決定事項や `CLAUDE.md` の設計方針と
矛盾する変更を提案する前に、必ず確認を取ること。最新が正しいとは限らない。
