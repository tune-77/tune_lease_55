---
name: git-ship
description: git add・commit・push・merge・ブランチ削除を一気に実行するスキル。「git全部やって」「add commit push merge」「ブランチ削除まで」「ship」「一気にコミット」などのキーワードが含まれる場合に必ず使用する。コミットメッセージはdiffから自動生成する。フィーチャーブランチ→masterのマージ、masterへの直接pushの両方に対応。
---

# git-ship スキル

ユーザーの指示でgit操作をまとめて実行する。変更内容を読んでコミットメッセージを自動生成し、add→commit→push→PR作成まで一気に行う。master/main が保護ブランチの場合は、直接pushや無人auto-mergeを試行錯誤せず PR URL と状態を返す。

## tune_lease_55 Fast Path（過去ログからの効率化）

理由: このリポジトリでは `master` が保護ブランチで、直接pushは `GH006: Protected branch update failed` になりやすい。また `master` が別worktreeに占有されていることがあり、`git checkout master` で詰まりやすい。さらに日次生成物・`data/`・画像がdirtyに混ざり、手作業のステージング確認が長くなりやすい。
適用条件: `tune_lease_55` で `Gitship` / `ship` / `add commit push merge` を依頼された時。
削除条件: ship 用CLIが dirty分類、PR作成、保護ブランチ判定、CI待ち、マージ可否確認、Obsidianログ保存まで一括で安全に担保する時。

最初にこの順で確認する:

```bash
python scripts/classify_git_ship_candidates.py
git branch --show-current
git status -sb
git worktree list
git log --oneline --left-right --graph --cherry-pick HEAD...origin/master | head -40
```

判断:
- `include` だけを基本ステージング対象にする。`review` は必要性を説明できるものだけ入れる。
- `avoid` は原則入れない。特に `data/`、`reports/*_latest.*`、`static_data/*.json`、`frontend/public/**/*.webp` は日次生成物・実行時成果物として扱う。
- `master/main` へ直接pushしない。保護ブランチ前提で、フィーチャーブランチpush → GitHub PR作成を標準にする。
- `master` が別worktreeで使用中でも、無理にcheckoutしない。現在ブランチに `origin/master` をmergeしてリモートへpushし、PRで統合する。
- `gh auth status` が無効なら `gh pr create` へ固執しない。GitHubコネクタが使える場合はそれでPR作成し、使えない場合はpush後のPR作成URLを返す。
- required checks があるPRで auto-merge を有効化するのは、ユーザーの明示承認がある時だけ。無人マージの迂回はしない。

コミット前の追加ガード:

```bash
python scripts/check_pr_change_risk.py --base origin/master...HEAD || true
```

これは警告用。generated/runtime artifacts が混ざっていないか、PRが過大でないかを早めに見る。

## フロー判定

理由: master/main への直接pushとフィーチャーブランチのmergeでは、必要な確認と後片付けが違う。
適用条件: ユーザーが git add/commit/push/merge をまとめて依頼した時。
削除条件: ship 用CLIが現在ブランチ判定、commit、push、merge、ブランチ削除を安全に自動化した時。

まず現在のブランチを確認する：

```bash
git branch --show-current
```

- **master / main ブランチ** → 原則、新規フィーチャーブランチを切って [Aフロー: PR作成](#a-フローフィーチャーブランチ)
- **フィーチャーブランチ** → [Aフロー: merge＋ブランチ削除](#a-フローフィーチャーブランチ)

---

## 事前確認（共通）

```bash
git status
git diff --stat
git log --oneline -3
python scripts/classify_git_ship_candidates.py
```

これらを読んでコミットメッセージを生成する。

### コミットメッセージの自動生成ルール

`git diff`と`git status`の内容から以下のフォーマットで生成：

```
<type>: <変更内容の要約（日本語OK）>

Co-Authored-By: Codex Sonnet 4.6 <noreply@anthropic.com>
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

# 2.5 プリフライト検証ガード（警告のみ・push は止めない）
#     構文崩れ／幻覚import／同一箇所への繰り返し修正を最後の一線で検知する。
python3 scripts/preflight_pr_guard.py || true
python scripts/check_pr_change_risk.py --base origin/master...HEAD || true

# 3. リモートへpush
git push origin <branch>

# 4. PR作成（master/main保護を前提にする）
# gh が使える場合:
gh pr create --base master --head <branch> --title "<title>" --body "<summary>"
# gh が無効なら GitHubコネクタ、または push 出力のPR URLを使う

# 5. PRマージ後だけブランチ削除（ローカル＋リモート）
git branch -d <branch>
git push origin --delete <branch>
git worktree prune 2>/dev/null || true
```

PR作成後に確認：
```bash
git log --oneline -3
git ls-remote origin refs/heads/<branch> refs/heads/master
```

最後にObsidianへ作業ログを残す。Codex作業なら `codex-work-log`、Claude作業なら `claude-work-log` を使う：

```bash
python3 .agents/skills/obsidian/scripts/obsidian_note.py codex-work-log \
  --vault "$OBSIDIAN_VAULT" \
  --summary "<今回の作業要約>" \
  --chat-summary "<ユーザーとの会話の短い要約。後で経緯を追えるよう、依頼・判断・制約を1〜3行で残す>" \
  --decision "<重要な決定>" \
  --change "<変更ファイル>" \
  --verification "<実行した検証>" \
  --git "commit: <sha>" \
  --git "merge: <sha>"
```

`OBSIDIAN_VAULT` が未設定またはVaultが複数ある場合は、`/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault` を優先する。会話全文、秘密情報、DB生データは保存せず、作業要約と会話要約だけを残す。

---

## Bフロー：直接push（masterブランチ）

```bash
# 1. ステージング
git add -A -- ':!data/' ':!.streamlit/secrets.toml'

# 2. コミット
git commit -m "<自動生成メッセージ>"

# 2.5 プリフライト検証ガード（警告のみ・push は止めない）
python3 scripts/preflight_pr_guard.py || true

# 3. push
git push origin master
```

push後にObsidianへ作業ログを残す。Codex作業なら `codex-work-log`、Claude作業なら `claude-work-log` を使う：

```bash
python3 .agents/skills/obsidian/scripts/obsidian_note.py codex-work-log \
  --vault "$OBSIDIAN_VAULT" \
  --summary "<今回の作業要約>" \
  --chat-summary "<ユーザーとの会話の短い要約。後で経緯を追えるよう、依頼・判断・制約を1〜3行で残す>" \
  --decision "<重要な決定>" \
  --change "<変更ファイル>" \
  --verification "<実行した検証>" \
  --git "commit: <sha>"
```

---

## 注意事項

理由: DB・secrets・force push は復旧困難な事故につながるため、ship時の最後の停止線として残す。
適用条件: git add/commit/push/merge/branch delete を行う時。
削除条件: git hooks と secret scanner と branch protection が同等の禁止を強制できる時。

- `data/` 配下（DB・セッション・jsonl）はコミットしない
- `.streamlit/secrets.toml` はコミットしない
- `--no-verify` は使わない
- force pushは使わない
- worktreeが残っている場合は `git worktree prune` で掃除する
- Codexが利用制限・課金条件で実装途中に止まった場合は、改善IDを `applied` にせず、次で保留記録する：

```bash
python3 scripts/record_codex_auto_status.py REV-xxx \
  --status blocked_by_quota \
  --detail "Codex usage limit reached before implementation"
```

この状態は次回の自動改善キューから除外され、朝レポートの `Blocked by Codex Quota` に表示される。

## エラー時

理由: push拒否やブランチ削除失敗を力技で進めると履歴破壊や未マージ削除につながる。
適用条件: git ship 中に conflict、non-fast-forward、branch delete failure が起きた時。
削除条件: ship 用CLIが安全な中断・rebase提案・merge済み確認を自動で扱える時。

- **コンフリクト発生** → ユーザーに報告してマージを中断。手動解決を依頼。
- **push拒否（non-fast-forward）** → `git pull --rebase` を提案。force pushはしない。
- **ブランチ削除失敗** → マージ済みか確認してから再試行。
