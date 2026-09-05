---
name: git-ship
description: tune_lease_55で、対象変更を安全に選別してgit add・commit・feature branchへのpush・PR作成・必要ならmergeとブランチ削除まで行う。「git全部やって」「add commit push merge」「ブランチ削除まで」「ship」「一気にコミット」で使用する。単なるgit状態確認やdiff説明には使用しない。
---

# Git Ship

ユーザーが明示した変更だけを、保護ブランチを迂回せず安全に出荷する。

## 適用境界

理由: このリポジトリは日次生成物と無関係なdirty変更が混在しやすく、`master` は保護ブランチである。
適用条件: ユーザーがadd、commit、push、PR、merge、ブランチ削除をまとめて依頼した時。
削除条件: 専用ship CLIが対象分類、検証、PR、merge済み確認、後片付けまで同等に保証する時。

- ユーザーの依頼に含まれるgit操作だけを行う。PR作成までの依頼ならmergeしない。
- `master` / `main` へ直接pushしない。feature/fix/chore branchからPRを使う。
- force push、`--no-verify`、未mergeブランチの強制削除をしない。
- ユーザーの既存dirty変更を、今回の対象だと確認せずステージしない。
- `data/`、`.streamlit/secrets.toml`、実行時ログ、日次生成物は、明示的に依頼対象でない限り含めない。

## 1. 対象を確定する

```bash
python scripts/ops_friction_doctor.py
python scripts/classify_git_ship_candidates.py
git branch --show-current
git status -sb
git worktree list
git diff --stat
```

- 分類器の `include` を基本候補にする。
- `review` は今回の依頼との関係を説明できるものだけ含める。
- `avoid` は含めない。
- 今回の変更と無関係なdirtyファイルはそのまま保護する。

## 2. ブランチを安全に整える

現在地が `master` / `main` なら、未コミット変更を保持したまま依頼内容に合うブランチを作る。

```bash
git checkout -b feature/<短い説明>
```

bugfixは `fix/`、設定・文書中心なら `chore/` を使う。既存の作業ブランチならそのまま使い、別worktreeの `master` を無理にcheckoutしない。

## 3. 検証して対象だけをステージする

- 変更領域に近いtargeted testを実行する。
- frontend変更では `cd frontend && npx tsc --noEmit` を実行する。
- `git add -A` で全dirty treeを拾わず、確認済みパスを列挙してステージする。

```bash
git add <確認済みパス...>
git diff --cached --stat
git diff --cached --check
git status --short
```

ステージ後に、対象漏れと無関係ファイルの混入をもう一度確認する。

## 4. commitとpush

コミットメッセージはstaged diffから生成する。形式は `<type>: <変更要約>`。関連するREV番号がある場合は、コミットとPRタイトルの両方へ含める。

```bash
git commit -m "<type>: <変更要約>"
python3 scripts/preflight_pr_guard.py || true
python scripts/check_pr_change_risk.py --base origin/master...HEAD || true
git push -u origin <branch>
```

プリフライト警告は握りつぶさず、出荷可否へ影響する内容をユーザーへ報告する。

## 5. PR、merge、後片付け

PR作成を依頼されている場合:

```bash
gh pr create --base master --head <branch> --title "<title>" --body "<変更概要と検証結果>"
```

- `gh` 認証が使えない場合は、push後のPR作成URLを返して止める。
- required checksやreviewを迂回しない。
- mergeまで依頼されている場合だけ、checksとmerge可否を確認してPRをmergeする。
- ブランチ削除まで依頼されている場合だけ、merge済みを確認してローカル・リモートブランチを削除する。

```bash
git fetch --prune origin
git log --oneline -3
git ls-remote origin refs/heads/<branch> refs/heads/master
```

## 6. 作業ログを保存する

依頼されたgit操作が正常に完了したら、会話全文や機密情報を含めず、決定・変更・検証結果を日次作業ログへ残す。PRを作成した場合はPR番号も記録する。

```bash
python3 scripts/save_work_log.py \
  --title "<PRタイトルまたは変更要約>" \
  --what "<変更内容と検証結果の1〜2行要約>" \
  --pr <PR番号> \
  --tags "自動生成"
```

- PRを作成していない場合は `--pr` を省略する。
- Obsidianへの保存が `skipped` でも、git操作自体は失敗扱いにしない。
- `scripts/save_work_log.py` が利用できない場合は、`obsidian` skillの `codex-work-log` / `claude-work-log` 手順で保存する。

## エラー時

- conflict: 作業を止め、競合ファイルと選択肢を報告する。
- non-fast-forward: 状態を確認し、通常のrebase/mergeを提案する。force pushしない。
- PR作成失敗: 認証、branch protection、required checksを確認し、PR URLまたは停止理由を返す。
- branch削除失敗: merge済みか再確認し、未mergeなら削除しない。

## 完了報告

- commit SHAとbranch
- push先とPR URL
- merge状態
- 実行した検証
- 意図的に除外したdirtyファイル群
