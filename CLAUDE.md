# リース審査AI — Claude Code プロジェクト指示書

## プロジェクト概要
温水式リース審査AIシステム（Streamlit + SQLite + Slack Bot）。
審査スコアリング・レポート・エージェント議論・リースくんウィザードを提供する。

## 主要ファイル構成
- `tune_lease_55.py` — Streamlit エントリポイント・ページルーティング
- `components/` — 各画面コンポーネント（chat_wizard, report, home, sidebar 等）
- `scoring/` — AIスコアリングロジック（industry_hybrid_model.py 等）
- `asset_scorer.py`, `total_scorer.py`, `category_config.py`, `scoring_core.py` — スコアリング中核（ルート直下）
- `quantum_analysis_module.py` — 量子干渉スコアによる財務矛盾検出（score_calculation で呼ばれ quantum_risk≥35 で要注意フラグ）
- `train_quantum.py` — quantum_model.joblib の学習スクリプト
- `slack_bot.py` / `slack_screening.py` — Slack ボット・審査フロー
- `data/` — SQLite DB・セッションファイル（機密情報含む、コミット禁止）
- `.claude/reports/` — エージェント間共有レポート

エージェント協調プロトコルの詳細は `.claude/AGENTS.md` を参照。

---

## Serena MCP 使用方針
コード調査・編集は Serena の `get_symbols_overview` / `find_symbol` / `replace_symbol_body` を優先して使うこと。ファイル全体の `Read` より先にシンボル単位での取得を試みる。

## コーディング規約
- Python 3.10+、型アノテーション推奨
- Streamlit の `st.session_state` 操作は副作用に注意
- `data/` 以下のファイルはコミットしない（`.gitignore` 参照）
- 数値は基本「千円」単位（スコアリングモジュールは「円」単位、変換注意）

## セキュリティ注意事項
- `.streamlit/secrets.toml` は絶対にコミットしない
- `data/*.db`, `data/*.sqlite`, `data/*.jsonl` はコミットしない
- Slack トークンは環境変数か secrets.toml から取得

## ⚠️ フロントエンド lint ルール（必読）

**`eslint --fix` は絶対に実行してはならない。**

過去に `eslint --fix` がUIコンポーネントやAPIエンドポイントを削除する事故が発生している。

### 正しい lint の使い方
```bash
# ✅ チェックのみ（これだけ使う）
cd frontend && npm run lint

# ❌ 絶対禁止（コードが削除される）
cd frontend && npm run lint:fix   # → わざとエラーになるよう設定済み
cd frontend && npx eslint --fix   # → 実行禁止
```

### lint エラーが出たとき
1. エラーメッセージを読んで **手動で修正** する
2. `no-unused-vars` 警告が出ても、そのコードが実際に使われているなら削除しない
3. 意図的に未使用の変数は `_` プレフィックスを付ける（例: `_unused`）
4. どうしても抑制が必要な場合は `// eslint-disable-next-line rule-name` コメントを使う

### next build と lint は分離済み
`next build` は lint を実行しない（`ignoreDuringBuilds: true`）。
lint は CI の `frontend-lint` ジョブが **報告のみ** で実行する（ブロックしない）。
