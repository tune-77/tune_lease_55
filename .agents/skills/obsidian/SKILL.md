---
name: obsidian
description: Obsidian Vault とCodex作業を連携するスキル。Vault探索、Markdownノート作成、日次ノート追記、プロジェクト作業ログ・決定事項・TODOのObsidian保存、既存ノート検索、機密情報を避けた要約保存を依頼されたときに使用する。「Obsidianに書いて」「Vaultへ保存」「今日のノートに追記」「codex-obsidian」「作業ログをノート化」などで必ず使用する。
---

# Codex Obsidian Skill

## 目的

Codexの作業結果、意思決定、TODO、検証ログをObsidian VaultへMarkdownとして保存・追記する。

## 原則

- Vault外へ勝手に公開しない。Obsidianはローカル保存先として扱う。
- 個人情報、APIキー、DBの生データ、顧客名、秘密情報は必要最小限に要約する。
- 既存ノートを編集する前に対象パスを確認する。
- 追記は原則として見出し単位で行い、既存本文を破壊しない。
- ユーザーが単に「Obsidianに保存」「Vaultに保存」と言った場合は、iCloud 上の通常の `Obsidian Vault` を既定保存先にする。`lease-wiki-vault` は、ユーザーが明示的に `lease-wiki` / `wiki vault` / `lease_wikiの方` と指定した場合だけ使う。
- Vaultが不明な場合は、`OBSIDIAN_VAULT` 環境変数、ユーザー指定パス、iCloud 上の Obsidian Vault を優先して探す。

## 標準ワークフロー

1. Vaultを決める
   - ユーザーがパスを指定していればそれを使う。
   - 未指定なら、まず `/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault` を使う。
   - `lease-wiki-vault` は通常保存先ではない。ユーザーが明示した場合だけ使う。
   - 未指定なら `OBSIDIAN_VAULT` を確認する。
   - iCloud Vault が無い場合だけ `find` で `.obsidian` ディレクトリを探す。

2. 保存内容を整理する
   - `Summary`: 何をしたか
   - `Decisions`: 決定事項
   - `Changes`: 変更ファイル・DB変更・検証
   - `Open Items`: 残課題
   - `Links`: 関連ファイル、Git commit、GitHub URL

3. ノートを作成または追記する
   - 日次ログなら `YYYY-MM-DD.md` に追記。
   - プロジェクト計画なら `Projects/<project>/...md` に作成。
   - 技術メモなら `Codex/...md` に作成。

4. 最後に保存先を報告する
   - 絶対パス
   - 作成/追記の別
   - 機密情報を省いたこと

## 補助スクリプト

`scripts/obsidian_note.py` を使う。

```bash
python3 .agents/skills/obsidian/scripts/obsidian_note.py find-vaults
python3 .agents/skills/obsidian/scripts/obsidian_note.py append-daily --vault /path/to/vault --text "..."
python3 .agents/skills/obsidian/scripts/obsidian_note.py agent-work-log --agent Codex --vault /path/to/vault --summary "..." --chat-summary "..." --decision "..." --change "..." --verification "..." --git "..."
python3 .agents/skills/obsidian/scripts/obsidian_note.py codex-work-log --vault /path/to/vault --summary "..." --chat-summary "..." --decision "..." --change "..." --verification "..." --git "..."
python3 .agents/skills/obsidian/scripts/obsidian_note.py claude-work-log --vault /path/to/vault --summary "..." --chat-summary "..." --decision "..." --change "..." --verification "..." --git "..."
python3 .agents/skills/obsidian/scripts/obsidian_note.py write-note --vault /path/to/vault --path "Codex/example.md" --title "Example" --text "..."
python3 .agents/skills/obsidian/scripts/obsidian_note.py search --vault /path/to/vault --query "lease"
```

Vaultがワークスペース外の場合、通常の権限ルールに従って承認を取る。

## ノート形式

日次追記:

```markdown
## HH:MM Codex

### Summary
- ...

### Chat Summary
- ...

### Decisions
- ...

### Verification
- ...

### Open Items
- ...
```

Agent作業ログ:

```markdown
## HH:MM Codex Work Log

または

## HH:MM Claude Work Log

### Summary
- ...

### Decisions
- ...

### Changes
- ...

### Verification
- ...

### Git
- ...
```

運用:
- Codex/Claudeとの会話全文は保存しない。
- 保存するのは、作業要約、会話要約、意思決定、変更ファイル、検証、commit/merge SHA、残課題だけにする。
- 会話要約は依頼内容・判断・制約を1〜3行で残し、秘密情報や会話全文は含めない。
- `git-ship` 後は、機密情報を含めずに `codex-work-log` または `claude-work-log` で日次ノートへ追記する。

プロジェクトノート:

```markdown
---
created: YYYY-MM-DD
source: codex
project: tune_lease_55
---

# Title

## Summary
...
```

## このプロジェクトでの推奨配置

リース審査AI関連は、Vault内に以下があれば優先する。

- `Projects/tune_lease_55/`
- `Codex/tune_lease_55/`
- `Daily/`

なければ作成してよい。
