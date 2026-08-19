# /explain-diff — 変更差分の要約とレビュー観点の提示

`/explain-score`（スコア判定根拠）とは別物。こちらは **git diff** を要約する。

## 使い方
```
/explain-diff [対象ブランチ・コミット（省略時は作業ツリーの差分）]
```

**例:**
```
/explain-diff
/explain-diff master
/explain-diff HEAD~3
```

## 処理手順

1. **差分を取得する**
   ```bash
   git diff <対象> --stat
   git diff <対象>
   ```

2. **変更ファイルを CLAUDE.md の「要注意領域」表と照合する**
   - `scoring_core.py` を含む → UMAPキャッシュ・score_base/scoreキー区別への影響を確認
   - `api/main.py` の `/api/chat` を含む → intent分岐（改善/通常/軍師AI）を壊していないか確認
   - `obsidian_bridge.py` / ChromaDB関連を含む → パス変更でRAG全壊のリスクを確認
   - `run_daily_improvement_pipeline.sh` を含む → 追記のみ・`|| true` が保たれているか確認
   - `ledger.jsonl` を含む → `canonical_key(title)` 形式のキーになっているか確認
   - `api/shion_*.py` を含む → import タイミング方針（先頭 or 遅延）がファイルごとの既存方針と一致するか確認

3. **危険パターンをチェックする**
   - `APPROVAL_LINE` や Q_risk 閾値（35, 60）の値がハードコードされていないか
     （`scoring_core.APPROVAL_LINE` を import せず別定数を定義していないか）
   - `data/` 配下・`.streamlit/secrets.toml` が diff に含まれていないか（コミット禁止）
   - `eslint --fix` の実行痕跡（大量のフォーマット変更のみのdiff）がないか

4. **要約を出力する**

```
## 変更概要
- <ファイル数>ファイル変更 (+<追加行> / -<削除行>)

## 変更内容
- <ファイルパス>: <何をどう変えたか、1行>

## レビュー観点
- ⚠️ <要注意領域に該当する場合、具体的な確認ポイント>
- ✅ 危険パターンなし / ❌ <検出した問題>

## PR前チェック
- [ ] `cd frontend && npx tsc --noEmit`
- [ ] `cd frontend && npm run lint`（`--fix` は使用禁止）
- [ ] PRタイトルに REV番号を含める
```

## 注意事項
- このコマンドは Read-only（diff の取得のみ、コード変更は行わない）
- 危険パターンを検出した場合はマージ前にユーザーへ明示的に警告する
