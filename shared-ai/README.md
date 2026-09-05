# Shared AI

Claude Code と Codex が共通利用する、Git管理された実行資産の正本。
Obsidian は知識の正本、`shared-ai/` はスキルと共通規約の正本として分離する。

## 配置ルール

- `skills/`: Claude Code と Codex の両方で同じ手順を使うSkillだけを置く。
- `knowledge/`: 両ツールで共通する短い運用規約を置く。
- Claude固有のSkillは `.claude/skills/`、Codex固有のSkillは `.agents/skills/` に残す。
- 共通Skillはコピーせず、`.claude/skills/` と `.agents/skills/` から相対symlinkで参照する。
- 1ツールでしか使わないSkillを、将来使うかもしれないという理由だけで共通化しない。

## 正本境界

- Obsidian知識: `runtime_paths.resolve_obsidian_vault()` が返す通常Vault。
- Lease Wiki: `runtime_paths.resolve_lease_wiki_vault()` が返す通常Vault内の `lease-wiki-vault/`。
- 共通Skill: このディレクトリの `skills/`。
- 改善・台帳類: `docs/improvement_source_of_truth.md` に従う。

## 棚卸し

- Skill: 30日利用実績と発火競合を確認し、未使用だけで即削除しない。
- MCP: 常駐コンテキストとCLI代替を確認し、実測なしに置換しない。
- 一時物: 削除前に正本・参照元・復元手段を確認する。

2026-09-05から30日間は `scripts/measure_skill_usage.py` で計測する。
スケジュール済みタスク `skill-vault-30` が毎週月曜に累積レポートを作り、
最終日の2026-10-05に旧Vaultアーカイブも読み取り専用で確認する。
