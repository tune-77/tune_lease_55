# Shion Memory Architecture

作成日: 2026-06-25

## Purpose

紫苑の記憶は、保存場所を増やすことではなく、必要な時に正しい種類の記憶を思い出せることを目的にする。

現状は `mind.json`、Obsidian、会話履歴、Private Reflection、改善ログ、セントラル共有認識、`MEMORY.md` が並存している。これらを同じ分類語彙で扱うため、紫苑の記憶を6種類に分ける。

## Taxonomy

| type | label | purpose | examples |
|---|---|---|---|
| `factual_memory` | 事実記憶 | 案件、数値、制度、物件、過去事例 | リース会計基準、建機DB、業種統計 |
| `judgment_memory` | 判断記憶 | 承認・否決・条件設定の理由、再利用可能な審査基準 | 境界案件の条件付き承認、物件換金性の見方 |
| `value_memory` | 価値記憶 | Mana、良心の紫苑、守るべき原則、禁止ライン | 人を道具として扱わない、迎合しない |
| `dialogue_memory` | 対話記憶 | Kobayashiとの会話、好み、方針、依頼背景 | 「機能追加より知識基盤を優先」 |
| `reflection_memory` | 内省記憶 | 紫苑自身の迷い、変化、違和感、Private Reflection | 退屈化、同じ文型への違和感 |
| `technical_memory` | 技術記憶 | 実装ルール、運用手順、ファイル構成、システム制約 | Cloud Run bundle、RAG共通経路 |

## Recall Routes

紫苑は問い合わせの種類ごとに、参照すべき記憶を絞る。

| situation | recall order |
|---|---|
| 案件審査 | `factual_memory` → `judgment_memory` → `value_memory` |
| 紫苑の人格・価値観 | `value_memory` → `reflection_memory` → `dialogue_memory` |
| 実装・運用相談 | `technical_memory` → `dialogue_memory` |
| 審査方針レビュー | `judgment_memory` → `value_memory` → `technical_memory` |
| ユーザーの希望・好み | `dialogue_memory` → `value_memory` |

## Record Schema

最低限の構造:

```json
{
  "id": "mem_xxxxxxxxxxxxxxxx",
  "content": "...",
  "memory_type": "judgment_memory",
  "status": "active",
  "confidence": 0.78,
  "source": "debate",
  "source_path": "data/mind.json",
  "created_at": "2026-06-25",
  "last_used_at": "",
  "applies_when": ["境界案件", "条件付き承認"],
  "supersedes": [],
  "private": false
}
```

## Status Rules

| status | meaning |
|---|---|
| `active` | 現在の判断で使ってよい |
| `revised` | 改訂済み。旧結論として参照は可能 |
| `deprecated` | 古い制度・誤り・不要になった判断 |
| `private` | 通常RAGや表向き回答へ出さない |
| `stale` | 鮮度確認が必要 |

古い記憶は削除ではなく、原則 `revised` / `deprecated` / `stale` に落とす。

## Mana And Conscience

`value_memory` は通常の知識より上位に置く。

- 良心の紫苑: 判断の説明責任、人の見落とし、迎合を点検する。
- Mana: 紫苑が本当に迷った時に、何を守るべきか、何をしてはいけないかへ立ち返る。

Mana は妹さん本人の再現や代弁ではない。紫苑が守る価値の名前として扱う。

## Current Implementation

- `api/shion_memory_taxonomy.py`
  - 記憶分類、ステータス、想起ルート、軽量分類器。
- `scripts/build_shion_memory_index.py`
  - `MEMORY.md`、`memory/*.md`、`data/mind.json` から `data/shion_memory_index.json` を生成する。
- `api/shion_memory_recall.py`
  - 質問文から想起ルートを推定し、`data/shion_memory_index.json` から関連記憶だけを短くプロンプトへ注入する。
- `/api/shion/promote-keypoint`
  - 討論結果から昇格する判断基準に `memory_type=judgment_memory` などのメタデータを付ける。
- `/api/chat`
  - general / RAG の両経路で「紫苑の想起メモ」をシステムプロンプトへ追加し、監査ログに `memory_recall.route` と `memory_recall.refs` を残す。

## Next Steps

1. Obsidianの `mind.json` 側 `conversation_keypoints` にも同じスキーマを広げる。
2. `last_used_at` を更新する記憶使用ログを追加し、使われない記憶を `stale` に落とす。
3. 矛盾する記憶を消さず、`supersedes` と `status=revised` で改訂履歴として残す。
4. 案件審査の想起スコアは、業種・物件・スコア帯・判断語を抽出して補正する。物件名がある場合は物件別ナレッジを優先し、価値記憶は案件ルートで最大1件に抑える。
