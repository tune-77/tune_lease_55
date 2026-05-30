# 開発・PR ワークフロー

## ブランチ命名規則

```
feature/rev-<番号>-<説明>           # REV 改善項目の実装
feature/revs-<番号>-<番号>-<説明>   # 複数 REV のまとめ実装
fix/<説明>                          # バグ修正
chore/<説明>                        # 設定・ドキュメント変更
```

## git-ship スキル

`/git-ship` コマンドで変更内容を自動読み取りし、add → commit → push → PR 作成まで実行できる。

手動手順：
```bash
# 1. ステージング（data/ と secrets.toml は除外）
git add -A -- ':!data/' ':!.streamlit/secrets.toml'

# 2. コミット（conventional commits 形式）
git commit -m "feat: <変更の要約>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

# 3. push & PR 作成
git push origin <branch>
gh pr create --title "..." --body "..."
```

## master への pull（競合回避）

`.claude/state/` が他セッションで変更されている場合：
```bash
git stash -- .claude/ && git pull origin master && git stash drop
```

## REV 改善パイプラインとの連携

`~/Library/Logs/tunelease/improvement_YYYYMMDD.log` に自動生成された改善候補（REV-XXX）を参照。
Next.js フロントエンドの実装を優先して進める。

スコア閾値：
- ≥70pt = 承認
- 60-69pt = 条件付き承認  
- <60pt = 否決

Q_risk 閾値：
- ≥35 = 要注意
- ≥60 = 強警戒（`credit_quantum_strong_warning` フラグ）
