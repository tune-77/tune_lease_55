# /prepare-release — PR前の変更履歴・確認事項の整理

## 使い方
```
/prepare-release [対象ブランチ（省略時は現在のブランチ）]
```

## 処理手順

1. **変更履歴を集約する**
   ```bash
   git log <base>..<branch> --oneline
   git diff <base>..<branch> --stat
   ```

2. **必須チェック（CLAUDE.md 最優先ルール）**
   - [ ] `cd frontend && npx tsc --noEmit` を実行し、エラーがないことを確認
   - [ ] `cd frontend && npm run lint` を実行（`--fix` は絶対に使わない）
   - [ ] `git status` で `data/` 配下・`.streamlit/secrets.toml` がステージされていないか確認
   - [ ] ブランチ名が `feature/rev-<番号>-...` / `fix/...` / `chore/...` 規約に沿っているか確認

3. **REV番号の確認（改善台帳連携に必須）**
   - 変更が `ledger.jsonl` の改善項目に対応する場合、PRタイトルに REV番号が含まれているか確認
   - 含まれていないと `cleanup_improvement_reviews.py --apply` が台帳を更新できないため、
     未確定なら実装前にユーザーへ確認する

4. **要注意領域チェック**（`/explain-diff` と同等の観点で再確認）
   - `scoring_core.py` / `api/main.py` / `obsidian_bridge.py` / `run_daily_improvement_pipeline.sh` /
     `ledger.jsonl` / `api/shion_*.py` を変更している場合、CLAUDE.md の該当行の注意点を満たしているか確認

5. **変更履歴・確認事項をまとめて出力する**

```
## リリース前チェックリスト

### 変更概要
- <コミット数>コミット、<ファイル数>ファイル変更

### 必須チェック
- [ ] tsc --noEmit: <結果>
- [ ] npm run lint: <結果>
- [ ] data/・secrets.toml 混入なし: <結果>
- [ ] ブランチ命名規約: <結果>
- [ ] REV番号: <該当REV番号 or 該当なし>

### 要注意領域への影響
- <該当ファイルと確認結果、なければ「なし」>

### PRタイトル案
feat: REV-XXX <変更の要約>
```

6. **`/git-ship` への引き継ぎ**
   - 上記チェックがすべて✅の場合のみ `/git-ship` の実行をユーザーに提案する
   - ❌がある場合は先にそれを解消するよう案内し、`/git-ship` は提案しない

## 注意事項
- このコマンド自体は commit/push/PR作成を行わない（確認・整理のみ）
- 実際の add/commit/push/PR作成は `/git-ship` スキルに委ねる
