# Shion-HyDE RAG After Hackathon Plan

作成日: 2026-07-14

## Purpose

ハッカソン後に、紫苑のRAGを「大量の文脈を出す検索」ではなく、「曖昧な相談を審査判断に使える確認点へ変換する検索」として強化する。

ユーザーの現代語の悩みや短い相談をそのままObsidianへ検索せず、いったん紫苑向けの仮想審査メモへ変換してから検索する。これにより、会話文と判断資産・Obsidianノートの語彙差を埋める。

## Freeze Rule

ハッカソン前は実装しない。

- 既存 `/api/chat`、紫苑対話室、審査画面の本流RAGを変更しない
- ChromaDB schema やObsidian索引を変更しない
- Cross-Encoderなど重い依存を追加しない
- まずは検証用スクリプトと debug API だけで効果を見る

## Core Idea

古典RAG向けの「HyDE: 平安時代の仮想場面生成」は、紫苑では次に置き換える。

```text
ユーザーの曖昧な相談
  -> Geminiで仮想審査メモ / 仮想判断資産へ変換
  -> その仮想文書を検索クエリにする
  -> Obsidian / 判断資産 / 類似メモから根拠を取得
  -> 今回見るべき確認点を最大3つに絞る
```

例:

```text
入力:
この案件、なんか不安。どこ見ればいい？

Shion-HyDE 仮想文書:
新規先または情報不足案件で、返済原資、競合・成約リスク、物件換金性、商流の妥当性、条件付き承認の確認点を整理する審査メモ。
```

## Components

### 1. Shion-HyDE

目的:

- ユーザー語を、紫苑が検索しやすい審査語へ変換する
- 雑談調ではなく、判断資産・確認点・承認条件の語彙へ寄せる
- 生成文は回答に直接出さず、検索専用に使う

候補ファイル:

- `api/knowledge/shion_hyde.py`
- `scripts/eval_shion_hyde_rag.py`

初期入力:

- `message`
- `context_mode`
- `industry`
- `asset_name`
- `score_band`
- `known_risk_tags`

出力:

```json
{
  "hyde_query": "...",
  "intent_tags": ["repayment_source", "competition_risk", "asset_liquidity"],
  "should_search": true,
  "reason": "..."
}
```

### 2. Parent-Child Chunking

目的:

- 検索精度は小さいchild chunkで上げる
- LLMへ渡す文脈は親chunkで保つ

設計:

- child chunk: 約200字、検索用
- parent chunk: 約500字、LLM投入用
- parent_id / child_id をmetadataで紐付ける
- 既存H2 chunkはすぐ置き換えず、検証用indexで比較する

候補ファイル:

- `api/knowledge/parent_child_chunks.py`
- `scripts/build_shion_parent_child_rag_index.py`

### 3. CRAG

目的:

- 検索根拠が弱い時に、無理にRAG回答へ寄せない
- メタデータフィルタ解除、検索語拡張、キーワード検索、基本QAへ順に逃がす

初期閾値:

- `confidence >= 0.65`: 通常RAG回答
- `0.45 <= confidence < 0.65`: 追加検索して根拠弱め扱い
- `confidence < 0.45`: フィルタ解除・fallback検索
- fallback後も弱い: RAG根拠なしとして、確認観点だけ答える

候補ファイル:

- `api/knowledge/rag_corrector.py`

### 4. Re-ranking

目的:

- Chroma Top5/Top10から、今回の文脈に近いTop3だけに絞る

段階:

1. 既存 `rank_score`、`confidence_for_hit()`、path priority、query coverageで擬似rerank
2. 日本語/多言語Cross-Encoder候補をローカル検証
3. 効果が確認できた時だけ本番導入

注意:

- `ms-marco-MiniLM-L-6-v2` は英語寄りなので、紫苑本番候補として固定しない
- 依存増・レイテンシ増・Cloud Runサイズ増を先に測る

### 5. Async Execution

目的:

- RAG検索、HyDE生成、rerankをユーザー体験上詰まらせない

方針:

- FastAPI本流へ入れる前に、debug endpointで `asyncio.to_thread()` の影響を見る
- 画像生成並列ではなく、RAG処理の非ブロッキング化を主目的にする
- タイムアウト時は既存RAGまたは基本QAへ戻す

## Implementation Phases

### Phase A: Offline Evaluation Only

実装:

- `scripts/eval_shion_hyde_rag.py`
- 評価用クエリセット作成
- 既存検索 vs Shion-HyDE検索を比較

評価対象:

- 法定耐用年数など基本QA
- Qrisk / 競合リスク
- 返済原資
- 物件換金性
- 条件付き承認
- 判断資産候補

合格条件:

- 既存検索より有用なノートを拾う
- Daily / AI Chat / Private Reflection / Humor を不必要に拾わない
- 回答候補が3確認点以内に収まる

### Phase B: Debug API

実装:

- `/api/debug/shion-hyde-rag`
- `debug_memory=true` 時だけ結果を返す
- 本回答プロンプトにはまだ混ぜない

返す情報:

- original query
- hyde query
- raw hits
- corrected hits
- final selected parent chunks
- confidence
- fallback reason

### Phase C: Shadow Mode

実装:

- `/api/chat` と紫苑対話室で裏側だけ実行
- 回答内容は変えず、ログだけ保存

見るもの:

- 既存RAGとShion-HyDE RAGの差
- 人間フィードバックが良かった回答との相関
- 低confidence時のfallback回数

### Phase D: Limited Injection

条件:

- Phase A-Cで効果が確認できた場合のみ

導入先:

- 審査レビューの確認点選定
- 判断資産候補の検索
- 紫苑対話室のリース実務質問

出力制限:

- 確認点は最大3つ
- 各確認点は1理由
- 出典は必要時だけ短く表示
- 不確実な根拠では断定しない

## Evaluation Metrics

検索品質:

- expected source hit rate
- irrelevant source rate
- fallback recovery rate
- top3 precision

紫苑品質:

- 確認点が3つ以内か
- 汎用論ではなく案件行動に変わっているか
- 判断資産出典が追跡できるか
- `効いた / 微妙 / 外した` の比率

運用品質:

- p95 latency
- Cloud Run memory impact
- model download size
- timeout/fallback rate

## Risks

- HyDE生成文が強すぎて、実際には存在しない前提で検索してしまう
- Cross-Encoder依存でCloud Runが重くなる
- RAGが強くなりすぎて紫苑の回答が長文化する
- 判断資産と一般論が混ざり、出典が曖昧になる
- 低スコア根拠を高信頼と誤認する

## Guardrails

- HyDE文書は回答へ直接出さない
- 低confidence時は断定しない
- Private Reflection はRAG対象外のまま維持
- Mana guard / memory immune-system を通らない入力は記憶・RAG接続しない
- 判断資産は人間フィードバックと結果検証なしに昇格しない
- 本番導入前に既存回帰テストを通す

## First Concrete Tasks After Hackathon

1. 評価クエリを20件作る
2. `scripts/eval_shion_hyde_rag.py` を作る
3. GeminiなしのテンプレートHyDEから始める
4. 既存 `get_store().search()` と比較する
5. Parent-Child chunkを検証用だけで作る
6. CRAG閾値をログで調整する
7. Cross-Encoderは最後に候補選定する

## Decision

ハッカソン後にやる。今は計画として保存し、現行アプリの安定を優先する。
