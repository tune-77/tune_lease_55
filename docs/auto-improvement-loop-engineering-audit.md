# Auto-improvement pipeline — Loop Engineering audit

基準は Loop Engineering の `loop-design-checklist.md`、`loop-constraints.md`、`gate.yaml` を採用する。外部ツール一式は導入せず、このリポジトリの既存 ledger・PR・テスト運用へ基準だけを移植する。

## 結論

自動改善は **L2 Assisted** として運用する。低リスク・単一ファイル変更だけを自動実装候補にし、denylist、人間ゲート、最大3試行、maker/checker分離を全実装経路で機械的に強制する。L3 Unattendedには昇格しない。

機械可読な真実源は `.agents/skills/auto-improvement-pipeline/loop_constraints.json` とし、通常Step 3とGemini agent経路の両方が同じ判定を使う。

## 適用した制約

- denylist: secrets、credentials、環境変数、data、Terraform、本番Kubernetes
- human gate: auth、payments、billing、migration、CI/launchd、スコアリング、DB/API、インフラ
- change size: 1回の自動変更は1ファイルまで
- attempts: 1改善につき最大3回。永続attempt ledgerで4回目を実行前に停止
- roles: implementerとverifierを別コンポーネントとして固定
- verification: verifierはdetached git worktreeで候補を検証し、実装側の作業ツリーではテストしない
- completion: verifierの証跡がなければappliedへ昇格しない

## 残るギャップ

トークン使用量による80%停止と、全ループ共通の `pause-all` kill switchは未実装。このためL3 Unattendedではなく、PR人間承認を維持する。

監査は `python scripts/audit_auto_improvement_loop.py` で再実行できる。
