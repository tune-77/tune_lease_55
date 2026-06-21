---
type: agent_policy
title: 紫苑の記憶参照と検索効率
domain: rag
tags: [紫苑, 記憶, RAG, Obsidian, 検索効率]
source: internal_review
confidence: medium
status: active
updated: 2026-06-20
related:
  - obsidian_ai_context.py
  - obsidian_query.py
  - mobile_app/obsidian_bridge.py
---

# 紫苑の記憶参照と検索効率

## 要点

紫苑の記憶参照は、質問文をそのまま検索語にせず、共通経路で検索語分解・文脈生成・Vault検索を行う。

## 判断ルール

- 検索語分解は `obsidian_query.py` を使う。
- AIプロンプト用文脈は `obsidian_ai_context.py` を使う。
- Vault検索本体は `mobile_app/obsidian_bridge.py` を使う。
- `AI Chat` や `Improvement Log` より、知識ノートを優先する。
- OKF風frontmatterの `type`, `domain`, `tags`, `confidence` は検索結果の説明と優先度判断に使う。

## 根拠

経路が分散すると、同じ質問でも実装ごとに検索結果が変わり、記憶精度が安定しない。

## 関連

- [[Obsidian]]
- [[RAG]]
- [[紫苑]]

