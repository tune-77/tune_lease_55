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

レポートの書式（frontmatterのキーと規則・見出し構成）は `.claude/reports/REPORT_SCHEMA.md` が正。ここには再掲しない。
**新規エージェントを `.claude/agents/` に追加する時も同様に、このテンプレート一式を埋め込まないこと。** 「作業後は `.claude/reports/<agent>/latest.md` へ書く（書式はREPORT_SCHEMA.md参照、`reads_from: [...]`）」の1行＋そのエージェント固有の「詳細」相当セクションの構成・申し送り先だけを記載する（2026-08にテンプレート三重重複が発覚し解消した経緯があるため）。

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
    reads: agent-team/*, scoring-audit/latest.md, data-quality/latest.md

[migration-validator]   → .claude/reports/migration/latest.md
    起動タイミング: SQLiteスキーマ変更（CREATE/ALTER/DROP）を含むコード変更後
    reads: file-searcher, code-review

[ledger-consistency-auditor] → .claude/reports/ledger-consistency/latest.md
    起動タイミング: REV番号採番ロジック変更後、または定期監査時
    reads: なし（台帳直接参照）
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
| **report-stylist** | agent-team/*, scoring-audit, data-quality | `report-stylist/latest.md` |
| **migration-validator** | file-searcher, code-review | `migration/latest.md` |
| **ledger-consistency-auditor** | なし | `ledger-consistency/latest.md` |

この表は `.claude/agents/` に定義された15エージェントを対象とする。エージェントを追加・削除したら、この表と上の依存関係図を同時に更新すること。

### レポートディレクトリについての注意

- 上表のディレクトリは **初回実行時に作成される**。存在しなくても異常ではない（`reads_from: []` として扱う）。
- **上表に載っていないディレクトリを上流レポートとして信頼しないこと。** 例外は `report-stylist` が読む `agent-team/*` のみ（`.claude/agents/` 未定義の実行主体が書くため、鮮度は保証されない）。
- 過去に `agent-discussion/`・`novelist/`・`general-purpose/` が同様に残存し、数ヶ月更新の止まった内容を誤参照する恐れがあったため削除した（git 履歴から復元可能）。
- レポートの `timestamp` は必ず確認し、古ければ上流エージェントの再実行を申し送る。

### レポートの鮮度維持

理由: レポートは書き手が居ないと更新されず、放置すると古い指摘が現在の真実として扱われる。
適用条件: `.claude/reports/` の鮮度に関わる運用・障害調査時。
削除条件: 鮮度維持が別の仕組みで保証され、STALE が発生しなくなった時。

**1. 決定論的な自己監査（`scripts/build_agent_self_reports.py`）**

以下7件は素の Python で監査し、日次パイプラインが毎日 `latest.md` を書き換える。
**LLM もサブエージェントも使わない。** 監査を LLM に書かせると、幻覚で「異常なし」と
書かれるのが最悪の失敗になるため。

| 出力先 | 監査内容 |
|---|---|
| `rule-validation/` | ウェイト合計・資産/債務者配分・グレード閾値の順序 |
| `scoring-audit/` | スコアが 0〜100 に収まるか・`APPROVAL_LINE` 参照可否 |
| `data-quality/` | SQLite のテーブル件数・空テーブル |
| `build/` | コアモジュールの import 可否 |
| `log-analysis/` | `logs/*.log` の ERROR / WARNING 件数 |
| `api-health/` | SQLite・APIキー・Ollama 到達性 |
| `test-results/` | pytest 実行（所要時間が読めないため日次からは除外） |

依存が欠けている等で監査できない場合は `status: failure` を書く。**成功を装わない。**
手動実行は `python3 scripts/build_agent_self_reports.py --only rule-validation` など。

エージェントが書いた既存レポートを上書きする際は、初回に限り全文を同じディレクトリの
`previous-agent-report.md` へ退避し、`latest.md` から参照する。決定論的な監査は
「係数の飽和」のような判断込みの指摘を再現できないため、黙って消さない。
（セクション抽出ではなくファイル退避にしているのは、過去レポートが REPORT_SCHEMA に
厳密でなく、抽出だと静かに取りこぼすため。sidecar は `*/latest.md` しか読まないので
退避先はブリーフを汚さない。）解消したら手で削除してよい。

**2. 変更駆動エージェントは担当ファイル基準で鮮度判定（`AGENT_TRIGGER_PATHS`）**

`code-reviewer` などは変更駆動であり、壁時計の経過日数で判定すると
「その後まったく変更が無いレポート」まで捨ててしまう。担当ファイルが
レポートより後に変わったかで判定する（出典は上の「起動タイミング」欄）。

- 担当ファイル未定義のディレクトリは従来どおり経過日数（`STALE_AFTER_DAYS`）
- git が使えない環境では経過日数へ退避する（判定不能を fresh にしない）
- エージェントを追加したら `AGENT_TRIGGER_PATHS` も更新すること。綴り間違いはテストが検出する

**残る限界**: `code-reviewer` / `security-checker` の判断を伴う中身そのものは、
LLM 無しでは生成できない。2 は「不要な再監査を求めない」だけで、変更があったときに
誰かが実行する必要は残る。

**配置と依存関係は本ファイルが正**（書式は冒頭の通り `.claude/reports/REPORT_SCHEMA.md` が正）。REPORT_SCHEMA.md 側に配置・依存関係を再掲しないこと。

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
| `/explain-diff` | 変更差分の要約とレビュー観点の提示 | — | — |
| `/explain-score` | スコア判定根拠説明 | — | — |
| `/explore-codebase` | リポジトリ構造・依存関係の高速把握 | — | — |
| `/financial-forecast` | 財務予測・3期分析 | — | — |
| `/generate-industry-brief` | 業界動向レポート生成 | — | — |
| `/generate-report` | 審査レポートの生成・改善提案 | `report-stylist` (`--agent`) | 数秒〜数分 |
| `/optimize-coefficients` | 係数自動最適化 | — | — |
| `/plan-feature` | 実装前の作業計画整理（Plan-First Checkpoint） | — | — |
| `/prepare-release` | PR前の変更履歴・確認事項の整理 | — | — |
| `/quick-score` | 業種・売上・リース額からクイックスコアを計算 | — | 10秒 |
| `/run-tests` | ユニットテスト実行 | `test-runner` | 30〜60秒 |
| `/update-docs` | 変更に合わせたドキュメント更新 | — | — |
| `/validate-rules` | ウェイト合計・グレード閾値の整合性チェック | `rule-validator` (`--full`) | 10〜数分 |
| `/write-spec` | 曖昧な依頼を受け入れ条件つき仕様に変換 | — | — |
