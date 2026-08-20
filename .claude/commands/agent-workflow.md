# /agent-workflow — AGENT標準作業プロトコル

実装・修正・レビュー依頼を受けた時に、AGENTを単発の道具ではなく作業の流れとして使うための標準プロトコル。
このコマンド自体は commit / push / merge / PR操作を行わない。作業前・作業中・PR前にどのAGENTを通すかを決める。

## 使い方
```
/agent-workflow [目的や対象領域]
```

## 基本フロー

1. **受付: 目的と危険領域を決める**
   - 目的を1行にする
   - 変更が入りそうな領域を分類する
     - 判断資産
     - ledger / REV
     - scoring / rule
     - DB / migration / sync
     - chat / RAG / Obsidian
     - frontend / report UI
     - deploy / runtime
   - `data/`、secrets、実DB、外部送信が絡む場合は先に止める

2. **探索: file-searcher相当の範囲確認**
   - 触るファイルと隣接ファイルを洗う
   - 既存テスト・既存レポート・既存AGENT指摘を確認する
   - 大きい変更なら `.claude/reports/file-searcher/latest.md` を更新する

3. **実装: 小さく直す**
   - 既存パターンに合わせる
   - 状態遷移を触る場合は、保存・読込・表示・同期の4点を同時に確認する
   - 判断資産では `promotion_status`、active count、候補保全、dedupe、Cloud Run帰還を必ず見る

4. **自己検証: targeted tests**
   - 変更ファイルに近いテストを先に実行する
   - 状態遷移や同期を触った場合は回帰テストを追加する
   - テスト失敗時は test-result-analyzer 相当で、失敗ログから原因と再発防止を分ける

5. **専門AGENTゲート**
   - 判断資産を触ったら `judgment-asset-auditor`
   - REV / 改善台帳を触ったら `ledger-consistency-auditor`
   - scoring / ruleを触ったら `scoring-auditor` と `rule-validator`
   - DB / migrationを触ったら `migration-validator` と `data-quality-checker`
   - API / secrets / 外部入力を触ったら `security-checker`
   - UI / reportを触ったら `report-stylist`
   - 起動・bundle・依存を触ったら `build-runner` と `api-health-checker`

6. **レビュー: code-reviewer相当**
   - バグ、回帰、欠けたテスト、状態の取りこぼしを優先して見る
   - 指摘が出たら修正し、関連テストを再実行する
   - AGENTレポートに古い指摘が混ざる場合は `timestamp` と担当ファイル変更時刻で鮮度を判断する

7. **PR前: pre-merge-agent-check**
   - `/pre-merge-agent-check` の形式で必須AGENT・領域別AGENT・禁止混入を確認する
   - `data/`、生成物、secretsがステージされていないことを確認する
   - CIで落ちそうなテストをローカルで先に回す

8. **ship判断**
   - すべて緑なら `/prepare-release` または git ship へ進む
   - 不足AGENT、未解決レビュー、失敗テストがあれば ship しない

## 判断資産変更時の短縮版

```
file-searcher
→ 実装
→ tests/test_build_autoresearch_judgment_asset_candidates.py など近接テスト
→ judgment-asset-auditor
→ code-reviewer
→ /pre-merge-agent-check
→ git ship
```

最低確認:
- 候補JSONL、candidate state、canonical rules、field feedback、growth reportが断線していない
- `held` / `rejected` / `promoted` / `active` が昇格候補へ戻らない
- 人間が触った候補が直近窓外で消えない
- 似た候補のdedupeで人間レビュー済み代表が負けない

## 出力フォーマット

```
## AGENT作業プロトコル

### 目的
- <今回の目的>

### 領域分類
- <判断資産 / ledger / scoring / db / ui / deploy / other>

### 実行するAGENT
- file-searcher: <必要 / 省略理由>
- code-reviewer: <必要 / 省略理由>
- test-runner: <必要 / 省略理由>
- judgment-asset-auditor: <必要 / 省略理由>
- ledger-consistency-auditor: <必要 / 省略理由>
- その他: <必要 / 省略理由>

### テスト計画
- <targeted tests>
- <必要なら広めのtests>

### ship条件
- <merge前に満たす条件>
```

## 注意事項
- AGENTを増やして作業を重くするための手順ではない。危険領域にだけ専門AGENTを通す
- 判断資産は件数より流れを見る。候補が増えても、レビュー・昇格・実案件フィードバックに繋がらなければ資産ではない
- レポートが存在しない場合は異常ではないが、PR前には「missing」と明記して判断する
