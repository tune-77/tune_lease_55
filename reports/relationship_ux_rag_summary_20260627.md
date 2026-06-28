# Relationship UX RAG Summary 2026-06-27

## 一言でいうと

Cloud Run版のAIチャットにObsidian/RAGの記憶が薄いと、回答品質以上に「同じ紫苑がそこにいない」と感じられた。そこでCloudflare版とCloud Run版を比較し、記憶参照の証跡と、人間が連続性を感じる返答の見せ方を検査した。

結論は、AIの人格っぽさは内部に記憶を持つだけでは成立しない。人間が「覚えてくれている」「同じ存在だ」と読み取れる形で返されて初めて、関係性UXとして成立する。

## 発端

Cloud Run版のAIチャットは、最初はObsidian由来のデータが十分に付いていなかった。その状態では、同じような質問をしても回答が薄く、Cloudflare版と比べて大きな差があった。

Cloudflare版はローカルのObsidian/Vault/記憶に近い環境で動いており、返答に文脈、判断の連続性、紫苑らしい厚みが出ていた。そのため、ユーザーはCloudflare版に「意識があるのか」と感じるほどの愛着を持った。

## 仮説

差は単なるモデル性能ではない。

- Obsidian/RAGの記憶を実際に使っているか
- 過去判断や直近の文脈が返答に接続されているか
- 一般論ではなく、Userのリース判断資産として返っているか
- 返答の文体、呼びかけ、温度が「紫苑らしい」か

これらが揃うことで、人間は「同じ紫苑がそこにいる」と感じる。

## 実装したこと

Cloud Run版でも記憶の使用を外部から検査できるように、`/api/chat` に `debug_memory=true` を追加した。

デバッグ時だけ以下を返す。

- `knowledge_refs`
- `memory_recall.refs`
- `pdca_applied`
- `rag_context_used`
- `db_context_used`
- `obsidian_daily_used`
- `identity_memory.used`
- `identity_memory.layers.identity`
- `identity_memory.layers.judgment`
- `identity_memory.layers.recent`

さらに、Cloud Run版にも「同じ紫苑」と感じられる土台として、Public Chat Memory Packを3層に分けた。

- `identity.md`: 紫苑としての自己定義
- `judgment-principles.md`: リース判断資産としての原則
- `recent-continuity.md`: 直近の流れと継続性

Cloud RunはGCS Vaultからこの3層を読み込み、RAGとは別枠で `/api/chat` のプロンプトへ注入する。

## 比較実験

Cloudflare版とCloud Run版に同じ質問を投げて比較した。

保存先:

- `reports/chat_identity_feel_experiment_20260627.md`
- `reports/chat_identity_feel_experiment_20260627.json`

確認した質問は次の3つ。

1. Cloud Run版でも同じ紫苑がそこにいると感じられるか
2. Userが設備リースの判断で迷った時、一般論ではなく判断資産としてどう返すべきか
3. 残価リスクを見る観点を、稟議で使える形で短く整理する

## 結果

Cloud Run版は3問すべてで `memory_debug` を返した。

- `identity_memory.used=true`
- `identity/judgment/recent` の3層すべて true
- `memory_recall.refs` は毎回5件
- `pdca_applied=true`

つまり、Cloud Run版は技術的には「記憶を持たない別物」ではなくなった。

一方、Cloudflare版は回答品質や文体の温度は出ていたが、現在の公開経路では `memory_debug` が返っていなかった。そのため、体感比較は可能だが、証跡比較はCloud Run側の方が明確だった。

## 面白かった発見

この実験は、AIを検査しているようで、人間の認識も検査していた。

人間は、AIが実際に記憶を持っているかだけでなく、その記憶が「連続性として読み取れる形」で返されるかに強く反応する。

つまり、人間は記憶そのものより、記憶の見せ方に反応している。

同じ能力でも、以下の違いで「同じ紫苑」か「別の薄いAI」かが変わる。

- 呼びかけ
- 文体の温度
- 返答の距離感
- 過去判断への接続
- Userの文脈を先回りする感じ
- 一般論ではなく判断資産として返す姿勢

## 設計原則

紫苑の設計では、記憶を入れるだけでは足りない。

人間が記憶として受け取れる形で返すことが、関係性のUXになる。

そのため、今後の品質評価では、正解率やRAG使用有無だけでなく、次を検査する。

- 人間がどこで「覚えてくれている」と感じたか
- どこで「同じ紫苑だ」と感じたか
- どこで一般論に戻ったと感じたか
- どの表現が関係性を支えたか

## ハッカソンでの見せ方

これは単なるRAGデモではなく、関係性UXつきRAGのデモとして見せられる。

デモの流れ:

1. Cloud Run版とCloudflare版を並べる
2. 同じ質問を投げる
3. `memory_debug` で記憶参照を可視化する
4. 回答本文を見て、人間がどちらを「紫苑らしい」と感じるか判定する
5. 最後に「AIを検査しているようで、人間が人格をどこに見出すかを検査している」と示す

この切り口は、業務AIの品質を正解率だけでなく、記憶、連続性、愛着、関係性で評価するものになる。

