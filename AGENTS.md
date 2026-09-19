# AGENTS.md — tune_lease_55

このファイルには、このリポジトリ固有の事実と制約だけを置く。共通の作業方針、安全境界、文体は各エージェントのグローバル指示に従う。

## プロジェクト固有ルール

このプロジェクト(tune_lease_55)固有の Obsidian/RAG 連携ルール。他プロジェクトのワークスペースには適用しない。

## Operational Stability

- 日次改善パイプラインは、ユーザーが明示的に依頼した場合を除き、拡張・リファクタリングしない。正常稼働している場合は現状を維持する。

## AI Chat / Obsidian Search Rule

AIチャットからObsidian/Vaultを参照する処理を実装・変更する時のルールは `.claude/skills/obsidian-search-rule/SKILL.md` を参照。

## Obsidian Save Destination Rule

「Obsidianに保存」「Vaultに保存」と言われた時の保存先判定は `.claude/skills/obsidian-save/SKILL.md` を参照。

## External Helper Tool Rule

Reason: `context7` と `taste-skill` は有用だが、既存の RAG・記憶・UI ワークフローと役割が重なるため、常時依存にすると不安定化しやすい。
Scope: this project only. Apply when choosing whether to consult external helper tools during implementation or review.
Retirement: remove or rewrite if the repo adopts a dedicated external-doc or design-review pipeline that supersedes these manual rules.

- `context7` は、外部公式ドキュメントの書き方確認が必要な時だけ使う。対象は API 変更、ライブラリ更新、Cloud Run / Next.js / FastAPI の実装差分確認に限定する。
- `context7` の結果は実装方針の確認にだけ使い、プロダクトの回答文、RAG、Obsidian、保存データ、推論フローへ直結させない。
- 既存の Obsidian RAG、社内検索、web検索で十分なら `context7` は使わない。
- `taste-skill` は `frontend` の新規画面、大きな改修、見た目レビューの時だけ使う。
- `taste-skill` はレイアウト、余白、文字組み、色、情報密度、画面の個性の確認に限定し、機能要件やバックエンド設計には踏み込まない。
- 既存デザイン言語がある画面はそれを優先し、`taste-skill` を理由に毎回UIを作り変えない。

## Claude Code / Codex Shared Asset Rule

Reason: 同じSkillを `.claude/skills` と `.agents/skills` に複製すると、片側だけ更新されて判断・安全手順がずれる。
Scope: このプロジェクトでClaude CodeとCodexの両方が同じSkillまたは知識運用規約を使う時。
Retirement: 両ツールが同じ標準Skillディレクトリを直接参照し、symlinkなしで単一実体を保証できるようになった時。

- 共通Skillの正本は `shared-ai/skills/` とし、ツール別Skillディレクトリには相対symlinkだけを置く。
- 共通知識運用は `shared-ai/knowledge/shared-conventions.md` を参照する。
- 1ツール固有の権限、hooks、モデル、プラグイン規約は共通化しない。
