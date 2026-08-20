# /pre-merge-agent-check — PRマージ前AGENTチェックリスト

PRをマージする前に、変更内容に応じて必要なAGENTレポートを揃えるための読み取り専用チェック。
このコマンド自体は commit / push / merge / PR操作を行わない。

## 使い方
```
/pre-merge-agent-check [baseブランチ（省略時: origin/master）]
```

## 処理手順

1. **差分を確認する**
   ```bash
   git diff --name-only <base>...HEAD
   git diff --stat <base>...HEAD
   git status --short
   ```

2. **共通AGENTゲート**
   - [ ] `file-searcher`: 変更ファイルと関連ファイルの範囲を確認
   - [ ] `code-reviewer`: バグ・回帰・テスト不足を確認
   - [ ] `test-runner`: 対象テストと必要な広めのテストを確認
   - [ ] `test-result-analyzer`: テスト失敗があった場合のみ、原因と再発防止を整理
   - [ ] `security-checker`: API、認証、外部入力、secrets、ファイル書き込みを触った場合のみ実行

3. **変更領域別AGENTゲート**
   - 判断資産候補・昇格・Cloud Run同期・レビューUIを触った場合
     - [ ] `judgment-asset-auditor`
   - REV採番・改善台帳・ledger_rulesを触った場合
     - [ ] `ledger-consistency-auditor`
   - スコアリング、物件評価、カテゴリウェイトを触った場合
     - [ ] `scoring-auditor`
     - [ ] `rule-validator`
   - DBスキーマ・migrate・同期テーブルを触った場合
     - [ ] `migration-validator`
     - [ ] `data-quality-checker`
   - 審査レポートUI・分析表示を触った場合
     - [ ] `report-stylist`
   - 起動・依存・Cloud Run bundleを触った場合
     - [ ] `build-runner`
     - [ ] `api-health-checker`
   - ログ、常駐プロセス、launchd、pipeline運用を触った場合
     - [ ] `log-file-analyzer`

4. **レポート鮮度チェック**
   - 対象AGENTの `.claude/reports/<agent>/latest.md` が存在するか確認
   - `timestamp` が古い場合は、現在差分に対して再実行が必要と判断する
   - `.claude/AGENTS.md` の読み書き表にないレポートディレクトリは、上流根拠として採用しない

5. **禁止・警戒チェック**
   - [ ] `data/`、`.streamlit/secrets.toml`、実DB、生成レポートが意図せずステージされていない
   - [ ] `eslint --fix` 等による無関係な大量整形が混ざっていない
   - [ ] `promotion_status`、REV番号、active count、Cloud Run writeback のような状態遷移を触った場合は回帰テストがある
   - [ ] PRタイトルに必要なREV番号または変更目的が明記されている

6. **出力フォーマット**

```
## PRマージ前AGENTチェック

### 差分概要
- base: <base>
- 変更ファイル: <件数>
- 要注意領域: <判断資産 / ledger / scoring / db / ui / deploy / none>

### 必須AGENT
- [ ] file-searcher: <fresh / stale / missing / not needed>
- [ ] code-reviewer: <fresh / stale / missing / not needed>
- [ ] test-runner: <fresh / stale / missing / not needed>

### 領域別AGENT
- [ ] judgment-asset-auditor: <fresh / stale / missing / not needed>
- [ ] ledger-consistency-auditor: <fresh / stale / missing / not needed>
- [ ] scoring-auditor: <fresh / stale / missing / not needed>
- [ ] rule-validator: <fresh / stale / missing / not needed>
- [ ] migration-validator: <fresh / stale / missing / not needed>
- [ ] data-quality-checker: <fresh / stale / missing / not needed>
- [ ] security-checker: <fresh / stale / missing / not needed>

### マージ判断
- 判定: <mergeable / blocked>
- 理由: <不足しているAGENT・テスト・レビュー>
- 次の一手: <最短で何を実行するか>
```

## 注意事項
- AGENTレポートは助言であり、自動マージ承認ではない
- `judgment-asset-auditor` は判断資産の件数を増やすためではなく、断線・再表示・ノイズ化を止めるために使う
- `BLOCKED` 判定の場合、先に不足AGENTまたはテストを実行し、結果をPR説明かコメントへ要約する
