---
name: git-ship
description: git add・commit・push・merge・ブランチ削除を一気に実行するスキル。「git全部やって」「add commit push merge」「ブランチ削除まで」「ship」「一気にコミット」などのキーワードが含まれる場合に必ず使用する。コミットメッセージはdiffから自動生成する。フィーチャーブランチ→masterのマージ、masterへの直接pushの両方に対応。
---

# git-ship スキル

ユーザーの指示でgit操作をまとめて実行する。変更内容を読んでコミットメッセージを自動生成し、add→commit→push→(merge→ブランチ削除)まで一気に行う。

## フロー判定

まず現在のブランチを確認する：

```bash
git branch --show-current
```

- **master / main ブランチ** → [Bフロー: 直接push](#b-フロー直接push)
- **フィーチャーブランチ** → [Aフロー: merge＋ブランチ削除](#a-フローフィーチャーブランチ)

---

## 事前確認（共通）

```bash
git status
git diff --stat
git log --oneline -3
```

これらを読んでコミットメッセージを生成する。

### コミットメッセージの自動生成ルール

`git diff`と`git status`の内容から以下のフォーマットで生成：

```
<type>: <変更内容の要約（日本語OK）>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

typeの選び方：
- `feat` — 新機能追加
- `fix` — バグ修正
- `refactor` — リファクタリング
- `docs` — ドキュメント変更
- `chore` — その他（設定変更など）

生成したメッセージをユーザーに見せてから実行する。ユーザーが「OK」「そのままで」「進めて」と言ったら実行。

---

## Aフロー：フィーチャーブランチ

```bash
# 1. ステージング（data/以下・secrets.tomlは除外）
git add -A -- ':!data/' ':!.streamlit/secrets.toml'

# 2. コミット
git commit -m "<自動生成メッセージ>"

# 3. リモートへpush
git push origin <branch>

# 4. masterへマージ
git checkout master
git pull origin master
git merge --no-ff <branch> -m "Merge branch '<branch>'"
git push origin master

# 5. ブランチ削除（ローカル＋リモート）
git branch -d <branch>
git push origin --delete <branch>
git worktree prune 2>/dev/null || true
```

マージ後に確認：
```bash
git log --oneline -3
git branch -a | grep <branch>  # 削除されていればOK
```

---

## Bフロー：直接push（masterブランチ）

```bash
# 1. ステージング
git add -A -- ':!data/' ':!.streamlit/secrets.toml'

# 2. コミット
git commit -m "<自動生成メッセージ>"

# 3. push
git push origin master
```

---

## 注意事項

- `data/` 配下（DB・セッション・jsonl）はコミットしない
- `.streamlit/secrets.toml` はコミットしない
- `--no-verify` は使わない
- force pushは使わない
- worktreeが残っている場合は `git worktree prune` で掃除する

## エラー時

- **コンフリクト発生** → ユーザーに報告してマージを中断。手動解決を依頼。
- **push拒否（non-fast-forward）** → `git pull --rebase` を提案。force pushはしない。
- **ブランチ削除失敗** → マージ済みか確認してから再試行。
