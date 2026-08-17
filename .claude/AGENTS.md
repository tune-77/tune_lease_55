# レポート駆動エージェント協調プロトコル

理由: サブエージェントは互いの実行結果を直接参照できないため、ファイル経由で受け渡さないと後続が文脈なしで作業して重複調査や矛盾した結論が出る。
適用条件: `.claude/agents/` に定義されたエージェントを起動する時、およびその出力を読む時。
削除条件: エージェント間の文脈受け渡しがハーネス側で保証され、レポートファイルを介さずに上流結果を参照できるようになった時。

エージェントはタスク完了後に必ずレポートを `.claude/reports/<agent>/latest.md` へ書く。
後続エージェントは **作業前に上流レポートを Read ツールで読んでから** 開始する。

## 上流レポートが存在しない・古い場合の対処

| 状況 | 対処 |
|------|------|
| 上流レポートが存在しない | 独自にスコープを判断して作業を進め、`reads_from: []` と記録する |
| 上流レポートの timestamp が 24時間以上前 | ⚠️ 古い情報として扱い、必要に応じて上流エージェントの再実行を申し送りに記載する |
| 上流レポートの status が `failure` | 内容を読んで影響範囲を把握してから作業を進める |

`reads_from` には実際に Read したパスのみを記載する（存在しなかったパスは含めない）。

## レポートの書き方（全エージェント共通）

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

## エージェント実行順と依存関係

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

## 各エージェントの読み書きルール

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

この表は `.claude/agents/` に定義された14エージェントを対象とする。エージェントを追加・削除したら、この表と上の依存関係図を同時に更新すること。

### レポートディレクトリについての注意

- 上表のディレクトリは **初回実行時に作成される**。存在しなくても異常ではない（`reads_from: []` として扱う）。
- **上表に載っていないディレクトリを上流レポートとして信頼しないこと。** 例外は `report-stylist` が読む `agent-team/*` のみ（`.claude/agents/` 未定義の実行主体が書くため、鮮度は保証されない）。
- 過去に `agent-discussion/`・`novelist/`・`general-purpose/` が同様に残存し、数ヶ月更新の止まった内容を誤参照する恐れがあったため削除した（git 履歴から復元可能）。
- レポートの `timestamp` は必ず確認し、古ければ上流エージェントの再実行を申し送る。

### レポートの鮮度維持（再確認キュー）

理由: レポートは書き手が居ないと更新されず、放置すると古い指摘が現在の真実として扱われる。
適用条件: `.claude/reports/` の鮮度に関わる運用・障害調査時。
削除条件: エージェントの再実行が別の仕組みで保証され、STALE が発生しなくなった時。

`scripts/recheck_stale_agent_reports.py` が、30日超（`agent_sidecar_reader.STALE_AFTER_DAYS`）の
レポートを古い順に少数だけ再実行する。日次パイプライン（`scripts/run_daily_improvement_post.sh`）が
Brief 生成の直前に呼ぶため、同じ実行で新しい結果が蒸留される。

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `AGENT_RECHECK_DISABLED` | 未設定 | `1` でキルスイッチ（即停止） |
| `AGENT_RECHECK_DAILY_LIMIT` | `3` | 1日あたりの再実行数。`0` で無効化 |
| `AGENT_RECHECK_MAX_CONSECUTIVE_FAILURES` | `2` | 連続失敗で中断 |
| `AGENT_RECHECK_PERMISSION_MODE` | `acceptEdits` | `claude --print` の権限モード |

`claude --print` は非対話実行のため許可を尋ねる相手が居ない。権限モードを指定しないと
レポートの書き込みが通らないので `acceptEdits` を既定にしている。Bash を多用する
エージェントが権限で止まる場合は `dontAsk` を検討する（`bypassPermissions` は全チェックを
外すため既定にしない）。エージェントの選択はプロンプト頼みにせず `--agent` で明示する。

手動実行は `python3 scripts/recheck_stale_agent_reports.py --dry-run` で対象だけ確認できる。
エージェントを追加したら `AGENT_REPORT_DIRS` の対応表も更新すること（名前とディレクトリ名は
一致しない。例: `scoring-auditor` → `scoring-audit/`）。テストが未マッピングを検出する。

レポートの書式（frontmatter のキーと規則）は `.claude/reports/REPORT_SCHEMA.md` を参照。**配置と依存関係は本ファイルが正**で、REPORT_SCHEMA.md 側に再掲しないこと。

## カスタムコマンド（スキル）

`.claude/commands/` に以下のスラッシュコマンドが定義されている。**各コマンドの正確な引数・挙動は `.claude/commands/<名前>.md` が正**で、下表は索引。

「対応エージェント」欄の `(--full)` 等は、そのフラグを付けた時だけサブエージェントを起動する意味。フラグなしは軽量な直接実行。所要時間が `—` のものはコマンド定義に記載がない。

| コマンド | 用途 | 対応エージェント | 所要時間 |
|---------|-----|----------------|---------|
| `/analyze-logs` | ログファイルのエラー・警告抽出 | `log-file-analyzer` | 10〜30秒 |
| `/analyze-variables` | 変数重要度分析（IV / SHAP） | — | — |
| `/asset-evaluation` | 物件スコア詳細評価 | — | — |
| `/audit-scores` | スコアリング異常・乖離の監査 | `scoring-auditor` (`--full`) | 30秒〜数分 |
| `/batch-export` | バッチ審査エクスポート・CSV形式検証 | — | — |
| `/build-check` | 全モジュールのインポート・依存パッケージ確認 | `build-runner` | 15秒 |
| `/case-similarity` | 類似事例検索 | — | — |
| `/check-data` | DBデータ品質チェック（件数・異常値） | `data-quality-checker` (`--full`) | 10秒〜数分 |
| `/check-health` | 全依存サービス（Gemini/Ollama/Slack/SQLite）の接続確認 | `api-health-checker` | 30〜120秒 |
| `/explain-score` | スコア判定根拠説明 | — | — |
| `/financial-forecast` | 財務予測・3期分析 | — | — |
| `/generate-industry-brief` | 業界動向レポート生成 | — | — |
| `/generate-report` | 審査レポートの生成・改善提案 | `report-stylist` (`--agent`) | 数秒〜数分 |
| `/optimize-coefficients` | 係数自動最適化 | — | — |
| `/quick-score` | 業種・売上・リース額からクイックスコアを計算 | — | 10秒 |
| `/run-tests` | ユニットテスト実行 | `test-runner` | 30〜60秒 |
| `/validate-rules` | ウェイト合計・グレード閾値の整合性チェック | `rule-validator` (`--full`) | 10〜数分 |
