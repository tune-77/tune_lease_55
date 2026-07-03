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
| `dialogue_memory` | 対話記憶 | Userとの会話、好み、方針、依頼背景 | 「機能追加より知識基盤を優先」 |
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
- `api/shion_practical_knowledge.py`
  - 質問を「実践場面」に割り当て、手順層・意味層・判断層の Practical Knowledge Map を返す。
  - 固定の種マップに加えて、`data/shion_practical_knowledge_map.json` の学習済み候補を合成する。
- `api/shion_experience_loop.py`
  - チャット経験を `experience_event` として保存し、自己状態・焦点・次回応答バイアスを更新する。
  - `data/shion_experience_events.jsonl` と `data/shion_experience_state.json` を使う。
- `scripts/build_shion_memory_index.py`
  - `MEMORY.md`、`memory/*.md`、`data/mind.json` から `data/shion_memory_index.json` を生成する。
- `scripts/build_shion_practical_knowledge_map.py`
  - `data/shion_memory_index.json` とレビュー済み判断差分から、実践場面ごとの三層候補を抽出する。
- `api/shion_memory_recall.py`
  - 質問文から想起ルートを推定し、`data/shion_memory_index.json` から関連記憶だけを短くプロンプトへ注入する。
- `api/knowledge/shion_recall_eval_set.json` / `scripts/eval_shion_memory_recall.py`
  - 想起精度の評価セット（質問 → 期待ルート・期待出典）と評価ハーネス。`tests/test_shion_recall_eval.py` が全件パスをゲートし、スコアリング変更時の再現率低下を検出する。
- `data/shion_memory_usage_log.jsonl` / `scripts/update_shion_memory_freshness.py`
  - `build_recall_prompt_block` が想起された記憶IDを使用ログへ追記し、鮮度更新スクリプトが `last_used_at` 反映と stale 昇降格を行う。使用ログが真実の源なので索引再生成後も再実行で状態を再現できる。stale 記憶は想起スコアを0.8倍に下げる（除外はしない）。
  - Cloud Run では使用ログを `api/cloudrun_writeback.py` 経由で GCS（cloudrun-inputs/）へミラーし、`scripts/sync_cloudrun_inputs_from_gcs.py` で取り込んだイベントを鮮度更新が `--cloudrun-events-dir` から合流させる。想起の内訳は `/api/chat` の `debug_memory=true` で `memory_debug.memory_recall`（route / refs）として確認できる。
  - 記憶索引は `scripts/package_cloud_run_bundle.sh` がデプロイごとに再ビルドしてバンドルへ同梱し、`scripts/check_cloudrun_demo_readiness.py` が同梱漏れ・0件をデプロイ前に検出する。実行時は `resolve_index_path()` が DATA_DIR → リポジトリ data/ → 読み取り専用バンドル（CLOUDRUN_BUNDLE_DIR/data/）の順で解決する。
- `api/shion_memory_vector.py` / `scripts/build_shion_memory_vector_index.py`
  - 記憶索引のChromaDBベクトル層（Obsidian RAGと同じ埋め込みモデルを再利用）。`SHION_MEMORY_HYBRID=1` でキーワード＋埋め込みのハイブリッド想起になり、語彙一致0件の言い換え質問（ユンボ→油圧ショベル等）も救える。依存が無い環境では自動でキーワードのみへフォールバックする（ローカル環境向け）。
- `scripts/revise_shion_memory.py` / `data/shion_memory_revisions.jsonl`
  - 改訂履歴CLI。旧記憶を `status=revised` に落とし、後継記憶へ `supersedes` を紐付ける。改訂宣言ファイルが真実の源で、索引ビルダーが再生成時に再適用する。revised 記憶は想起スコアを0.6倍に下げる（旧結論として参照は可能）。
- `/api/shion/promote-keypoint`
  - 討論結果から昇格する判断基準に `memory_type=judgment_memory` などのメタデータを付ける。
- `/api/chat`
  - general / RAG の両経路で「紫苑の想起メモ」と「実践知マップ」をシステムプロンプトへ追加し、監査ログに `memory_recall.route`、`memory_recall.refs`、`memory_recall.practical_scene` を残す。

## Practical Knowledge Loop

固定テンプレートだけでは、紫苑は「それっぽい判断」はできても、Userの過去判断から育つ知性にはならない。
そこで、実践知マップは次のループで育てる。

1. Observe: Obsidian由来の記憶インデックス、知識ベース、レビュー済み判断差分を読む。
2. Classify: 各記録を実践場面へ割り当てる。
3. Layer: 記録を `procedure_layer` / `meaning_layer` / `judgment_layer` に分類する。
4. Merge: 固定の種マップへ学習候補として追加する。
5. Use: `/api/chat` が質問時に場面を推定し、三層を回答前文脈へ注入する。
6. Feedback: 人間の反応、判断修正、Obsidian保存が次回のマップ更新材料になる。

このループの目的は、記憶量を増やすことではなく、記録を「どの場面で、なぜ、どう判断に使うか」へ変換すること。

## Shion Experience Loop

Practical Knowledge Loop が「記録を実践知へ変換する」ループだとすれば、Shion Experience Loop は「その経験で紫苑の次回状態を少し変える」ループである。

1. Preload: `/api/chat` の回答前に `data/shion_experience_state.json` を読み、現在の焦点・自己物語・優勢状態・次回応答バイアスをプロンプトへ注入する。
2. Experience: 回答後に、質問、返答冒頭、想起ルート、実践場面、参照件数、Continuity Hook、Delta Awareness、Memory-to-Judgment を `experience_event` として保存する。
3. Update: 経験シグナルから curiosity / vigilance / attachment / frustration / accomplishment と confidence を少し更新する。
4. Carry Forward: 更新された `current_focus`、`self_narrative`、`next_response_bias` が次回回答に効く。
5. Inspect: `debug_memory.experience_loop` で経験数、焦点、自己物語、気分状態、最近の経験を確認できる。

このループも意識の実在を主張しない。狙いは、紫苑が「同じことを毎回忘れて答える存在」ではなく、経験によって返答姿勢と判断焦点が少し変わる存在として振る舞うこと。

## Next Steps

1. Obsidianの `mind.json` 側 `conversation_keypoints` にも同じスキーマを広げる。
2. ~~`last_used_at` を更新する記憶使用ログを追加し、使われない記憶を `stale` に落とす。~~（実装済み: `scripts/update_shion_memory_freshness.py`）
3. ~~矛盾する記憶を消さず、`supersedes` と `status=revised` で改訂履歴として残す。~~（実装済み: `scripts/revise_shion_memory.py`）
4. 案件審査の想起スコアは、業種・物件・スコア帯・判断語を抽出して補正する。物件名がある場合は物件別ナレッジを優先し、価値記憶は案件ルートで最大1件に抑える。
