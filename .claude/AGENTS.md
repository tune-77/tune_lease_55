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

[judgment-asset-auditor] → .claude/reports/judgment-asset-audit/latest.md
    起動タイミング: 判断資産候補生成・昇格・レビューUI・Cloud Run同期の変更後、または定期監査時
    reads: file-searcher, code-review（任意）
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
| **judgment-asset-auditor** | file-searcher, code-review（任意） | `judgment-asset-audit/latest.md` |

この表は `.claude/agents/` に定義された16エージェントを対象とする。エージェントを追加・削除したら、この表と上の依存関係図を同時に更新すること。

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

## 標準作業プロトコル

理由: AGENTを単発で呼ぶだけだと、探索・実装・検証・PR前確認の間で指摘が落ち、同じ種類の断線を繰り返す。
適用条件: コード変更、設定変更、運用手順変更、PRレビュー対応、git ship 前。
削除条件: ハーネス側で同等の作業ゲートが自動化され、AGENTレポートとテスト実行の対応が機械的に保証された時。

### 1. 受付と領域分類
- 目的を1行にし、変更領域を `判断資産` / `ledger` / `scoring` / `DB` / `chat・RAG` / `UI` / `deploy` / `other` に分類する。
- `data/`、secrets、実DB、外部送信が絡む場合は、ステージ禁止・読み取り専用・承認要否を先に決める。
- 判断資産・ledger・scoring・DB のいずれかを触る場合は、後続の専門AGENTゲートを省略しない。

### 2. 探索ゲート
- 変更前に file-searcher 相当の範囲確認を行い、関連ファイル・既存テスト・既存AGENTレポートを確認する。
- 大きい変更やレビュー対応では `.claude/reports/file-searcher/latest.md` を更新し、後続AGENTが同じ前提を読めるようにする。

### 3. 実装ゲート
- 既存パターンに合わせて小さく直す。
- 状態遷移を触る時は、保存・読込・表示・同期の4点を同時に確認する。
- 判断資産では `promotion_status`、候補保全、dedupe、active count、Cloud Run帰還を必ず確認する。

### 4. 検証ゲート
- まず変更に近い targeted tests を実行する。
- 状態遷移、同期、昇格、候補保全を触った場合は回帰テストを追加する。
- テストが失敗した場合は、修正前に test-result-analyzer 相当で原因・影響・再発防止を分ける。

### 5. 専門AGENTゲート
| 変更領域 | 通すAGENT |
|---|---|
| 判断資産候補・昇格・レビューUI・Cloud Run帰還 | judgment-asset-auditor |
| REV採番・改善台帳・ledger_rules | ledger-consistency-auditor |
| scoring・物件評価・カテゴリウェイト | scoring-auditor, rule-validator |
| DBスキーマ・migration・同期テーブル | migration-validator, data-quality-checker |
| API・認証・外部入力・secrets・ファイル書き込み | security-checker |
| 審査レポートUI・分析表示 | report-stylist |
| 起動・依存・Cloud Run bundle・runtime | build-runner, api-health-checker |
| ログ・常駐プロセス・launchd・pipeline運用 | log-file-analyzer |

### 紫苑からAGENTへの相談キュー
- 紫苑はCodex AGENTを直接実行しない。異常・断線・監査が必要な兆候を見つけた時は `request_agent_consultation` で `data/shion_agent_consultation_queue.jsonl` へ相談票を追記する。
- Codexは作業開始時または `/agent-workflow` 実行時に `/api/shion/agent-consultations` を確認し、必要なら該当AGENTを実行して `.claude/reports/<agent>/latest.md` に結果を書く。
- 相談票は `open` / `in_review` / `done` / `cancelled` の状態だけを持つ。相談票の作成はコード変更・データ正本変更・PR作成・AGENT実行を意味しない。

### 紫苑から外部推論役への相談キュー
- 紫苑はCodex停止時、低確信度、別視点が必要な論点では `request_reasoner_consultation` で `data/shion_reasoner_consultation_queue.jsonl` へ相談票を追記する。
- 相談先は `codex` / `gemini` / `claude` / `auto`。Codexは実装・監査、Geminiは仮説比較、Claudeは長文レビュー・第二意見を基本役割にする。
- 相談票は安全要約のみを保存する。生の顧客名・社名・財務数値・秘密・DB本体は渡さず、`privacy_level=safe_summary_only` を既定にする。
- 相談票の作成は外部AI実行・外部送信・コード変更・データ正本変更を意味しない。Codexは作業開始時に `/api/shion/reasoner-consultations` を確認し、回答要約だけを `answered` / `done` として残す。

### 6. レビューゲート
- code-reviewer 相当で、バグ・回帰・欠けたテスト・状態取りこぼしを優先して見る。
- 指摘を修正したら、関連テストと必要な専門AGENTゲートを再実行する。
- レポートの鮮度は `timestamp` だけでなく、`scripts/agent_sidecar_reader.py` の `AGENT_TRIGGER_PATHS` による担当ファイル変更基準も見る。

### 7. PR前ゲート
- `/pre-merge-agent-check` で必須AGENT、領域別AGENT、禁止混入、テスト不足を確認する。
- `data/`、生成物、secretsが意図せずステージされている場合は ship しない。
- CIで落ちそうなチェックをローカルで先に実行する。

### 8. ship条件
- targeted tests が通っている。
- 必要な専門AGENTゲートが missing/stale のまま残っていない。
- code-reviewer 相当の重大指摘が未解決でない。
- PR前ゲートの判定が `mergeable` である。

## カスタムコマンド（スキル）

`.claude/commands/` に以下のスラッシュコマンドが定義されている。**各コマンドの正確な引数・挙動は `.claude/commands/<名前>.md` が正**で、下表は索引。

「対応エージェント」欄の `(--full)` 等は、そのフラグを付けた時だけサブエージェントを起動する意味。フラグなしは軽量な直接実行。所要時間が `—` のものはコマンド定義に記載がない。

| コマンド | 用途 | 対応エージェント | 所要時間 |
|---------|-----|----------------|---------|
| `/analyze-logs` | ログファイルのエラー・警告抽出 | `log-file-analyzer` | 10〜30秒 |
| `/analyze-variables` | 変数重要度分析（IV / SHAP） | — | — |
| `/agent-workflow` | AGENT標準作業プロトコルの選定 | file-searcher / code-reviewer / test-runner / 該当専門AGENT | 数分 |
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
| `/pre-merge-agent-check` | PRマージ前AGENTチェックリスト | judgment-asset-auditor / ledger-consistency-auditor ほか該当AGENT | 数分 |
| `/quick-score` | 業種・売上・リース額からクイックスコアを計算 | — | 10秒 |
| `/run-tests` | ユニットテスト実行 | `test-runner` | 30〜60秒 |
| `/update-docs` | 変更に合わせたドキュメント更新 | — | — |
| `/validate-rules` | ウェイト合計・グレード閾値の整合性チェック | `rule-validator` (`--full`) | 10〜数分 |
| `/write-spec` | 曖昧な依頼を受け入れ条件つき仕様に変換 | — | — |
