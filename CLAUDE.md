# リース審査AI — CLAUDE.md

## ⚠️ 絶対厳守（過去の事故対応・外部自動化が依存する手順）

理由: ここは過去に事故が出た操作と、外部自動化が依存する手順だけを置く。
適用条件: このリポジトリでコード・データ・PR・改善台帳に触る作業。
削除条件: 該当事故の再発防止が別の仕組みで保証され、関連テストまたは自動化が不要になった時。

- `eslint --fix` 禁止（UIコンポーネント削除事故あり）→ `npm run lint` でチェックのみ
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

```
frontend/src/app/     # UI（25+ ページ）
api/main.py           # FastAPI エンドポイント（全API）
api/schemas.py        # Pydantic モデル
scoring_core.py       # スコアリング（RandomForest主モデル + 量子干渉）
data/lease_data.db    # SQLite（コミット禁止）
```

- **TS/Next.js**: strict mode厳守・`apiClient`（`src/lib/api.ts`）経由でAPI呼び出し。詳細: @.claude/rules/frontend.md
- **Serena MCP**: `get_symbols_overview` / `find_symbol` / `replace_symbol_body` をReadより優先
- **ブランチ/PR**: `feature/rev-<番号>-<説明>` / `fix/...` / `chore/...`。一括shipは `/git-ship`
- **出典明記**: 提案前に `static_data/` か `notes/` の具体ファイル名を引用。出典がなければ「これは推測です」と明示
- **計画確認**: `scoring_core.py` / `analysis_*.py` / フロントエンド変更前に3文の変更計画を提示し承認を得る
- **矛盾チェック**: 3ヶ月以上前の`ledger.jsonl`決定事項やCLAUDE.mdの方針と矛盾する変更前に必ず確認を取る（最新が正しいとは限らない）

詳細: @.claude/rules/workflow.md | @.claude/rules/security.md | .claude/AGENTS.md
週次サマリーは `scripts/weekly_self_management.py` が毎週月曜 `WEEKLY_LOG.md` に自動追記（本ファイルには追記しない）
