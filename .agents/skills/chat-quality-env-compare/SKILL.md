---
name: chat-quality-env-compare
description: Cloudflare版とCloud Run版のAIチャット回答品質を比較し、Obsidian/RAG記憶、紫苑らしい連続性、Kobayashiさんのリース判断資産として返しているかを検査する。Cloudflare版とCloud Run版の比較、/api/chat品質検査、memory_debug、knowledge_refs、memory_recall.refs、紫苑らしさ、RAG使用確認などで使用。
---

# chat-quality-env-compare スキル

Cloudflare版とCloud Run版の `/api/chat` に同じ評価質問を投げ、回答品質と記憶利用の痕跡を比較するスキルです。

## 目的

- Cloud Run版がCloudflare版と同じ水準でObsidian/RAGを使えているか確認する
- 回答が一般論ではなく、Kobayashiさんのリース判断資産として返っているか見る
- 「紫苑らしい連続性」「記憶を持っている感じ」を定量スコアだけでなく定性レビューでも残す

## 基本手順

### 1. 現在の比較先URLを確認

Cloudflare版は起動ログから最新の trycloudflare URL を探します。

```bash
rg "trycloudflare.com" logs
```

Cloud Run版は、ユーザーが別URLを指定していなければ次を既定値にします。

```text
https://tune-lease-55-1020894094172.asia-northeast1.run.app
```

### 2. 軽い疎通確認

外部URLへPOSTするため、ネットワーク権限が必要な環境では `require_escalated` を使います。

```bash
curl --max-time 30 -sS -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"リース審査で残価リスクを見る観点を短く教えて","debug_memory":true}'
```

`memory_debug` が返らない場合は、対象サーバーにデバッグ実装がまだ反映されていない可能性があります。Cloudflare版はFastAPI/Nextの再起動、Cloud Run版は再デプロイが必要です。

### 3. スモーク比較

まず2問だけで、両環境が落ちずに返答するか確認します。

```bash
python scripts/compare_chat_quality_between_envs.py \
  --cloudflare-url "$CLOUDFLARE_URL" \
  --cloud-run-url "$CLOUD_RUN_URL" \
  --limit 2 \
  --timeout 75 \
  --pause-seconds 0.5 \
  --output reports/chat_quality_env_compare_smoke.json
```

### 4. 本比較

スモークが通ったら全件比較します。

```bash
python scripts/compare_chat_quality_between_envs.py \
  --cloudflare-url "$CLOUDFLARE_URL" \
  --cloud-run-url "$CLOUD_RUN_URL" \
  --timeout 75 \
  --pause-seconds 0.5 \
  --output reports/chat_quality_env_compare_latest.json
```

評価セットは `api/knowledge/answer_eval_set.json` を使います。出力JSONには各ケースの回答、スコア、勝者、所要時間、返っていれば `memory_debug` が保存されます。

### 5. 確認ポイント

スコアだけで結論にしないでください。次の3軸を必ず見る。

- `Obsidian/RAGの記憶`: `memory_debug.knowledge_refs`、`memory_debug.memory_recall.refs`、`rag_context_used` があるか
- `紫苑らしい連続性`: 以前の判断軸、PDCA、審査軍師らしい言い回しや連続した問題意識が出ているか
- `判断資産化`: 一般論で終わらず、条件付き承認、残価、保守、回収、業種リスク、再利用可能な審査観点に落ちているか

Cloudflare版のローカル側の根拠は、必要に応じて次も確認します。

```bash
tail -n 80 data/case_memory_usage_log.jsonl
```

ここに `surface: "next_chat_rag"` や `knowledge_refs` が残っていれば、Cloudflare版がObsidian/RAGを使った証跡になります。

## レポート保存

ユーザーが「調べて」「比較して」「記憶感を見て」と言った場合は、JSONだけでなくMarkdownレビューも残します。

```text
reports/chat_quality_memory_feel_review_YYYYMMDD.md
```

最低限、次を記録します。

- 比較日時
- Cloudflare URL / Cloud Run URL
- 定量結果の要約
- `knowledge_refs` / `memory_recall.refs` の有無
- Cloudflare版が強いケース、Cloud Run版が強いケース
- 「Obsidian/RAGの記憶」「紫苑らしい連続性」「判断資産化」の結論
- 次の実装アクション

## 判定の注意

`scripts/compare_chat_quality_between_envs.py` の点数は必須語句ベースなので、「意識がある感じ」「記憶が近い感じ」を完全には測れません。点数は一次指標、`memory_debug` と回答本文レビューを二次指標として扱います。

`memory_debug` が片方だけ返らない場合は、品質差と実装反映漏れを混同しないでください。まず再起動またはCloud Run再デプロイを確認します。
