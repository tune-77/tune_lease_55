# リース審査AI — Claude Code プロジェクト指示書

## プロジェクト概要
温水式リース審査AIシステム（Streamlit + SQLite + Slack Bot）。
審査スコアリング・レポート・エージェント議論・リースくんウィザードを提供する。

## 主要ファイル構成
- `lease_logic_sumaho12.py` — Streamlit エントリポイント・ページルーティング
- `components/` — 各画面コンポーネント（chat_wizard, report, home, sidebar 等）
- `scoring/` — AIスコアリングロジック
- `slack_bot.py` / `slack_screening.py` — Slack ボット・審査フロー
- `data/` — SQLite DB・セッションファイル（機密情報含む、コミット禁止）
- `.claude/reports/` — エージェント間共有レポート（後述）

---

## レポート駆動エージェント協調プロトコル

エージェントはタスク完了後に必ずレポートを `.claude/reports/<agent>/latest.md` へ書く。
後続エージェントは **作業前に上流レポートを Read ツールで読んでから** 開始する。

### レポートの書き方（全エージェント共通）

```markdown
---
agent: <自分の名前>
task: <実施タスク概要>
timestamp: <YYYY-MM-DD HH:MM>
status: success | failure | partial
reads_from: [読んだ上流レポートのパス]
---

## サマリー
（1〜3行）

## 詳細
（箇条書き）

## 課題・リスク
（なければ「なし」）

## 後続エージェントへの申し送り
（次に何をすべきか）
```

### エージェント実行順と依存関係

```
コード変更発生
    │
    ▼
[file-searcher]  → .claude/reports/file-searcher/latest.md
    │
    ├──▶ [change-impact-analyzer]  → .claude/reports/impact-analysis/latest.md
    │         reads: file-searcher
    │
    ├──▶ [code-reviewer]           → .claude/reports/code-review/latest.md
    │         reads: file-searcher
    │         │
    │         └──▶ [security-checker]  → .claude/reports/security/latest.md
    │                   reads: file-searcher, code-review
    │
    ├──▶ [build-runner]            → .claude/reports/build/latest.md
    │
    └──▶ [test-runner]             → .claude/reports/test-results/latest.md
              │
              └──▶ [test-result-analyzer]  → .claude/reports/test-results/latest.md に追記
                        reads: test-results

[log-file-analyzer] → .claude/reports/log-analysis/latest.md
    reads: build, test-results（あれば）

── リース審査ドメイン固有エージェント（独立起動）──────────────────────────

[scoring-auditor]       → .claude/reports/scoring-audit/latest.md
    起動タイミング: asset_scorer.py / category_config.py / total_scorer.py 変更後
    reads: file-searcher（任意）

[data-quality-checker]  → .claude/reports/data-quality/latest.md
    起動タイミング: DBスキーマ変更後、定期監査時（週次推奨）
    reads: なし（DB直接アクセス）

[rule-validator]        → .claude/reports/rule-validation/latest.md
    起動タイミング: rule_manager.py / coeff_definitions.py / category_config.py 変更後
    reads: file-searcher, code-review

[api-health-checker]    → .claude/reports/api-health/latest.md
    起動タイミング: デプロイ後、障害発生時、定期監視時
    reads: なし

[report-stylist]        → .claude/reports/report-stylist/latest.md
    起動タイミング: 審査完了後・スコアリング結果のUI改善要求時
    reads: agent-team/*, scoring-audit/latest.md

[migration-validator]   → .claude/reports/migration/latest.md
    起動タイミング: SQLiteスキーマ変更（CREATE/ALTER/DROP）を含むコード変更後
    reads: file-searcher, code-review
```

### 各エージェントの読み書きルール

| エージェント | 読むレポート | 書くレポート |
|------------|------------|------------|
| file-searcher | なし | `file-searcher/latest.md` |
| change-impact-analyzer | file-searcher | `impact-analysis/latest.md` |
| code-reviewer | file-searcher | `code-review/latest.md` |
| security-checker | file-searcher, code-review | `security/latest.md` |
| build-runner | なし | `build/latest.md` |
| test-runner | なし | `test-results/latest.md` |
| test-result-analyzer | test-results | `test-results/latest.md` に分析セクション追記 |
| log-file-analyzer | build, test-results（任意） | `log-analysis/latest.md` |
| **scoring-auditor** | file-searcher（任意） | `scoring-audit/latest.md` |
| **data-quality-checker** | なし | `data-quality/latest.md` |
| **rule-validator** | file-searcher, code-review | `rule-validation/latest.md` |
| **api-health-checker** | なし | `api-health/latest.md` |
| **report-stylist** | agent-team/*, scoring-audit | `report-stylist/latest.md` |
| **migration-validator** | file-searcher, code-review | `migration/latest.md` |

### カスタムコマンド（スキル）

`.claude/commands/` に以下のスラッシュコマンドが定義されている：

| コマンド | 用途 | 所要時間 |
|---------|-----|---------|
| `/quick-score` | 物件IDと業種からクイックスコアを計算 | 10秒 |
| `/check-health` | 全依存サービス（Gemini/Ollama/Slack/SQLite）の接続確認 | 30〜120秒 |
| `/validate-rules` | ウェイト合計・グレード閾値の整合性チェック | 10〜数分 |
| `/generate-report` | 審査レポートの生成・改善提案 | 数秒〜数分 |

---

## コーディング規約
- Python 3.10+、型アノテーション推奨
- Streamlit の `st.session_state` 操作は副作用に注意
- `data/` 以下のファイルはコミットしない（`.gitignore` 参照）
- 数値は基本「千円」単位（スコアリングモジュールは「円」単位、変換注意）

## セキュリティ注意事項
- `.streamlit/secrets.toml` は絶対にコミットしない
- `data/*.db`, `data/*.sqlite`, `data/*.jsonl` はコミットしない
- Slack トークンは環境変数か secrets.toml から取得
