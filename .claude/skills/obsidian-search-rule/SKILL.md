---
name: obsidian-search-rule
description: AIチャットからObsidian/Vaultを参照する処理を実装・変更する時に使うスキル。「Obsidian検索実装」「RAG経路変更」「vault.rglobを直接呼ぶ」等の場面で使用。Flask/mobile版・Streamlit版・Next版・軍師AI・ホームFABチャットの全実装に適用する共通経路ルール。
---

# AI Chat / Obsidian Search Rule

理由: RAG経路が分散すると検索品質が不安定になり、知識ノートよりログが優先される事故が起きる。
適用条件: Flask/mobile版、Streamlit版、Next版、軍師AI、ホームFABチャットの Obsidian 参照処理。
削除条件: 全チャット実装が共通検索サービスへ統合され、直接Vault走査がテストで防止された時。

AIチャットでObsidianを参照する処理は、必ず共通経路を使う。

- 検索語分解: `obsidian_query.py`
- AIプロンプト用文脈: `obsidian_ai_context.py`
- Vault検索本体: `mobile_app/obsidian_bridge.py`

禁止:
- 各チャット実装で `vault.rglob("*.md")` を直接呼ぶ
- 「補助金について教えて」のような質問文を丸ごと検索語にする
- `AI Chat` / `Weekly Review` / `Improvement Log` を知識ノートより優先する
- `/tmp` へのprint/debug書き込みで検索挙動を安定化させる

このルールは Flask/mobile版、Streamlit版、Next版、軍師AI、ホームFABチャットのすべてに適用する。
