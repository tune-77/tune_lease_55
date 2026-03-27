---
agent: security-checker
task: セキュリティレビュー（bayesian_engine.py CPT拡張 + Streamlit UI 18ファイル一括置換）
timestamp: 2026-03-28 11:15
status: success
reads_from: [.claude/reports/file-searcher/latest.md, .claude/reports/code-review/latest.md]
---

## サマリー

| 重大度 | 件数 |
|--------|------|
| Critical | 1 |
| High | 2 |
| Medium | 2 |
| Low | 2 |

今回の変更（a4e4820, 3cc1bb5）に直接起因する新規脆弱性は [C-1] のみ（既存コードの発見）。
UI 置換（width=）はセキュリティリスクゼロ。

---

## 詳細

### [C-1] Critical: slack_bot.py:286-287 — Slack経由コマンドインジェクション

- `argument` は Slack メッセージ生テキスト、サニタイズなし
- 送信者の Slack User ID による認可制御が存在しない
- `--dangerously-skip-permissions` が常に付与、ファイルシステム・シークレット読み込みをプロンプト誘導で実行可能

**推奨**: argument から `--` 始まり文字列を除去、送信者ホワイトリスト化、`--dangerously-skip-permissions` 排除

### [H-1] High: components/chat_wizard.py:226 — XSS（AI生成テキスト）

`_user()` / `_humor()` が `unsafe_allow_html=True` で AI テキストを無エスケープ展開。
**推奨**: `html.escape(msg)` を適用

### [H-2] High: components/report.py:297,327,354-355,367 — XSS（LLM生成テキスト）

`strength_tags`, `comparison`, `passion_text` 等が HTML エスケープなしで挿入。
**推奨**: 全変数に `html.escape()` を適用

### [M-1] Medium: data_cases.py:509 — SQLクエリ列名f-string展開

現状は外部入力が流入しないが、将来的な拡張で SQL インジェクション成立構造を持つ。

### [M-2] Medium: components/ai_agent.py:111 — LLM生成式に対する eval()

`{"__builtins__": None}` の制限はオブジェクト探索チェーンで迂回可能。
**推奨**: `ast.literal_eval()` または専用数式評価ライブラリに置き換え

### [L-1] Low: bayesian_engine.py:288 — 入力値バリデーション不在

0/1 以外の値がサイレントにスキップされる（`nr=2` 等）。

---

## 優先修正順

C-1 > H-1 = H-2 > M-2 > M-1 > L-1

---

## 後続エージェントへの申し送り

- C-1 は今回のコミット以前から存在する既存リスク（優先対処推奨）
- H-1, H-2 は今回の UI 変更とは無関係の既存問題
