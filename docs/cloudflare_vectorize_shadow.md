# Cloudflare Vectorize シャドー検索

既存のChromaDBを変更せず、外部AI投入用に選別済みの160文書だけを
Workers AI `@cf/baai/bge-m3`（1,024次元）とVectorizeで検索する検証経路。
任意の `--rerank` 指定時はVectorize上位10件を
`@cf/baai/bge-reranker-base` で再採点する。日本語評価で劣後する可能性があるため、
既定値はVectorize順位を維持する。

## 安全境界

- 正本・本番検索は引き続きChromaDB
- 生Vault、チャット履歴、案件データは送信しない
- 対象は `data/agent_search/lease_knowledge_export/manifest.json` 掲載文書のみ
- API tokenは環境変数またはmacOSキーチェーンから読み、保存しない
- Vectorize indexは `tune-lease-rag-bge-m3-shadow-v1`。本番トラフィックは接続しない

## 操作

```bash
.venv/bin/python scripts/cloudflare_vectorize_shadow.py status
.venv/bin/python scripts/cloudflare_vectorize_shadow.py sync --apply
.venv/bin/python scripts/cloudflare_vectorize_shadow.py query "再リースの注意点" --confirm-sanitized-query
.venv/bin/python scripts/cloudflare_vectorize_shadow.py query "再リースの注意点" --rerank --confirm-sanitized-query
.venv/bin/python scripts/cloudflare_vectorize_shadow.py evaluate --confirm-sanitized-query
```

`sync` は既定でdry-run。Cloudflare側のindex作成・更新には `--apply` が必要。
適用時はindex内のIDを一覧取得し、現行exportにない古いベクトルを削除してからupsertする。
`query` は個人情報・案件情報・秘密情報を含まない検索文だけを対象とし、
送信確認として `--confirm-sanitized-query` が必要。
候補数は `--candidates`（既定10、最大20）、最終件数は `--top-k`（既定5）で変更できる。

## 自動比較ログ

`config/cloudflare_rag_shadow.json` の `mode=shadow` では、通常チャットのうち
短いリース一般知識質問だけをバックグラウンド比較する。審査・会社名・個人名・
金額・連絡先を含む質問は送信しない。外部クエリは許可済みのリース業務語だけに縮約する。

結果は `data/cloudflare_rag_shadow_log.jsonl` に保存され、本番回答・順位には影響しない。

```bash
.venv/bin/python scripts/report_cloudflare_rag_shadow.py
```
