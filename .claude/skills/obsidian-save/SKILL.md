---
name: obsidian-save
description: ユーザーが単に「Obsidianに保存」「Vaultに保存」と依頼し、保存先Vaultを判定する時だけ使用する。既定Vaultとlease-wiki-vaultの選択規則を提供するが、Vault横断検索、RAG実装変更、根拠付きレポート作成、Wikiリンク整理には使用しない。
---

# Obsidian 保存先判定スキル

ユーザーが「Obsidianに保存」「Vaultに保存」と言った場合、既定の保存先は iCloud 上の通常の `Obsidian Vault` とする。

このskillは保存先ポリシーだけを担当する。実際のノート作成・追記は `obsidian` の安全な保存手順を使う。

- 既定Vault: `/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault`
- `lease-wiki-vault` には、ユーザーが明示的に「lease-wiki」「wiki vault」「lease_wikiの方」と指定した場合だけ保存する。
- リサーチメモや一般メモは、通常Vault内の `Projects/tune_lease_55/Research/`、`Daily/`、または文脈に合う既存フォルダへ保存する。
- 保存後は、実際の絶対パスを報告し、`lease-wiki-vault` へ保存していないことが重要な文脈では明記する。
