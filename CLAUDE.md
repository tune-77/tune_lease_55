# リース審査AI — CLAUDE.md

## ⚠️ 最優先ルール

理由: ここは過去に事故が出た操作と、外部自動化が依存する手順だけを置く。
適用条件: このリポジトリでコード・データ・PR・改善台帳に触る作業。
削除条件: 該当事故の再発防止が別の仕組みで保証され、関連テストまたは自動化が不要になった時。

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
| `api/shion_*.py`（ADKエージェント定義ファイル群） | google.adk等のimportタイミング方針がファイルごとに違う | `shion_agent.py`はモジュール先頭import、`shion_debate_adk.py`は遅延import（google.adk未導入環境でも読み込み可能にするため）。新規ファイルを書く際はどちらの方針を採るか意識すること |

Work Logs: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Work Logs/`

---

## 数値単位（バグの温床）

フロント入力: **百万円**（UI表示ラベル参照）→ `toThousandYenPayload()`（×1000）→ スコアリングモジュール内: **千円**（`scoring_core.py` 内コメント参照）

スコア判定: `scoring_core.APPROVAL_LINE`（既定71点）以上=承認 / 60-69=条件付き / <60=否決 | Q_risk: ≥35=要注意 / ≥60=強警戒
→ 承認ラインを参照・複製する箇所は必ず `scoring_core.APPROVAL_LINE` を import すること。ハードコードした別定数を置くと審査結果がモジュールごとに食い違う（2026-07 レビューで `api/main.py` に71と60の二重定義が見つかった実例あり）

---

## プロジェクト構成

**Next.js 16 + FastAPI + SQLite**

```
frontend/src/app/     # UI（25+ ページ）
api/main.py           # FastAPI エンドポイント（全API）
api/schemas.py        # Pydantic モデル
scoring_core.py       # スコアリング（RandomForest主モデル + 量子干渉）
data/lease_data.db    # SQLite（コミット禁止）
```

---

## 規約・ツール

理由: このリポジトリは Next/FastAPI/自動改善/Serena など複数の作業経路が混在するため、入口を固定しないと調査と編集が散る。
適用条件: ローカル開発、コード調査、ブランチ作成、PR作成を行う時。
削除条件: ツール選択とブランチ規約が別の自動ガードまたはリポジトリ設定で強制された時。

- **TS/Next.js**: strict mode厳守・`apiClient`（`src/lib/api.ts`）経由でAPI呼び出し。詳細: @.claude/rules/frontend.md
- **Serena MCP**: `get_symbols_overview` / `find_symbol` / `replace_symbol_body` を優先（Read より先）
- **ブランチ**: `feature/rev-<番号>-<説明>` / `fix/...` / `chore/...`
- **一括ship**: `/git-ship` で add→commit→push→PR作成

詳細: @.claude/rules/workflow.md | @.claude/rules/security.md | .claude/AGENTS.md

---

## Freshman Rules（Vault成長に伴う品質維持）

理由: Vaultと改善履歴が増えるほど、根拠なしの推測や古い方針の丸呑みが起きやすくなるため。
適用条件: スコアリング・審査ロジック・業種データ・主要UIを変更または提案する時。
削除条件: 出典検査・計画確認・古い決定の矛盾検査が別の自動ガードで安定して担保された時。

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

## Weekly Log

`scripts/weekly_self_management.py` が毎週月曜に自動生成する週次サマリーは
`WEEKLY_LOG.md` に追記される（このファイルには追記しない）。

