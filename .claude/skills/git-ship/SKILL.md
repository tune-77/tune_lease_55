---
name: git-ship
description: git add・commit・push・PR作成・merge・ブランチ削除を一気に実行するスキル。「git全部やって」「add commit push merge」「ブランチ削除まで」「ship」「一気にコミット」などのキーワードが含まれる場合に必ず使用する。コミットメッセージはdiffから自動生成する。masterへの直接pushはフックで禁止されているため、常にフィーチャーブランチ→PR→マージの経路を取る。
---

# git-ship スキル

ユーザーの指示でgit操作をまとめて実行する。変更内容を読んでコミットメッセージを自動生成し、add→commit→push→PR作成→マージ→ブランチ削除まで一気に行う。

## ⚠️ 大前提: master への直接 push は禁止

**master/main への `git push` はフックでブロックされる。** 必ずフィーチャーブランチ → PR → `gh pr merge` の経路を通すこと。ローカルで master にマージしてから master を push する旧手順も同様にブロックされるので使わない。

## フロー判定

まず現在のブランチを確認する：

```bash
git branch --show-current
```

- **フィーチャーブランチ** → そのまま [shipフロー](#ship-フロー) へ
- **master / main ブランチ** → 先にブランチを切ってから [shipフロー](#ship-フロー) へ：

```bash
git checkout -b feature/<説明>   # バグ修正なら fix/、設定・ドキュメントなら chore/
```

（未コミットの変更は `checkout -b` にそのまま付いてくる）

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

Co-Authored-By: Claude <noreply@anthropic.com>
```

- Co-Authored-By には実行中のモデル名を入れる（例: `Claude Fable 5 <noreply@anthropic.com>`）

typeの選び方：
- `feat` — 新機能追加
- `fix` — バグ修正
- `refactor` — リファクタリング
- `docs` — ドキュメント変更
- `chore` — その他（設定変更など）

実装内容に REV が含まれる場合は PRタイトルに `REV-XXX` を含める（例: `feat: REV-039 パイプライン承認UI`）。これにより `cleanup_improvement_reviews.py --apply` が自動で台帳を applied に更新する。

生成したメッセージをユーザーに見せてから実行する。ユーザーが「OK」「そのままで」「進めて」と言ったら実行。

---

## ship フロー

```bash
# 1. ステージング（data/ 以下は除外）
#    注意: .streamlit/secrets.toml は .gitignore 済み。pathspec で明示除外する
#    （':!.streamlit/secrets.toml' を付ける）と advice エラーで add 全体が失敗するので書かない。
git add -A -- ':!data/'

# 2. コミット
git commit -m "<自動生成メッセージ>"

# 2.5 プリフライト検証ガード（警告のみ・push は止めない）
#     構文崩れ／幻覚import／同一箇所への繰り返し修正を最後の一線で検知する。
#     警告が出ても続行するが、内容はユーザーに提示すること。
python3 scripts/preflight_pr_guard.py || true

# 3. リモートへpush
git push -u origin <branch>

# 4. PR作成 → マージ（master を更新する唯一の経路）
gh pr create --title "<PRタイトル（REV番号があれば含む）>" --body "<変更概要・検証内容>"
gh pr merge <PR番号> --merge --delete-branch

# 5. ローカルを master に戻して掃除
git checkout master
git pull origin master
git branch -d <branch> 2>/dev/null || true
git fetch --prune origin
git worktree prune 2>/dev/null || true
```

マージ後に確認：
```bash
git log --oneline -3
git branch -a | grep <branch>  # 削除されていればOK
```

### マージ後: 作業ログ自動生成

マージ確認が完了したら、以下の情報をもとに作業ログを自動生成・保存する：

1. `git log --oneline -1` でマージコミットのタイトルを取得
2. `git diff --stat master~1 master` で変更ファイル一覧を取得
3. PRタイトルと変更ファイルから「何をしたか」を1〜2行で要約
4. 以下のコマンドで保存：

```bash
python3 scripts/save_work_log.py \
  --title "<PRタイトル（REV番号を含む）>" \
  --what "<変更内容の1〜2行要約>" \
  --pr <PR番号> \
  --tags "自動生成"
```

- `--why-hard` / `--next-time` は自動生成では省略可（後で「改善ポイント」として追記できる）
- Obsidianへの保存が失敗しても処理は続行される（`skipped` が返るだけ）

---

## 注意事項

- `data/` 配下（DB・セッション・jsonl）はコミットしない
- `.streamlit/secrets.toml` は .gitignore 済み（pathspec で明示除外するとエラーになるので書かない）
- master への直接 push・ローカルマージ後の master push はフックでブロックされる
- `--no-verify` は使わない
- force pushは使わない
- worktreeが残っている場合は `git worktree prune` で掃除する

## エラー時

- **コンフリクト発生** → ユーザーに報告してマージを中断。手動解決を依頼。
- **push拒否（non-fast-forward）** → `git pull --rebase` を提案。force pushはしない。
- **`gh pr merge` 失敗** → CI・レビュー必須・ブランチ保護の設定を確認してユーザーに報告。
- **ブランチ削除失敗** → マージ済みか確認してから再試行。
