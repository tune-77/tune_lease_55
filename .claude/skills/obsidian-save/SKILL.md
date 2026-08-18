---
name: obsidian-save
description: ユーザーが「Obsidianに保存」「Vaultに保存」と言った時に保存先を判定するスキル。既定Vaultとlease-wiki-vaultの使い分け、保存後の報告方法を扱う。
---

# Obsidian 保存先判定スキル

ユーザーが「Obsidianに保存」「Vaultに保存」と言った場合、既定の保存先は iCloud 上の通常の `Obsidian Vault` とする。

- 既定Vault: `/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault`
- `lease-wiki-vault` には、ユーザーが明示的に「lease-wiki」「wiki vault」「lease_wikiの方」と指定した場合だけ保存する。
- リサーチメモや一般メモは、通常Vault内の `Projects/tune_lease_55/Research/`、`Daily/`、または文脈に合う既存フォルダへ保存する。
- 保存後は、実際の絶対パスを報告し、`lease-wiki-vault` へ保存していないことが重要な文脈では明記する。
