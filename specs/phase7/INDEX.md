# Phase 7 SPEC INDEX — RAG 精度改善（ドメイン辞書）

Phase 7 の目的: リース業界固有の同義語・専門用語を集約したドメイン辞書を整備し（P7-001）、RAG検索のクエリ拡張に活用して検索の取りこぼしを減らす（P7-002）。`代替案実装計画_精度改善Ver2.md` 代替案Aの詳細設計。スコアリング（RF/LGBM・量子リスク）には一切触れない。

---

## SPEC一覧

| spec_id | ファイル | タイトル | ステータス | 依存 |
|---------|---------|---------|----------|------|
| P7-001 | [P7-001-lease-domain-glossary.md](P7-001-lease-domain-glossary.md) | リース業界ドメイン辞書（データ + ローダー） | implemented | なし |
| P7-002 | [P7-002-rag-query-expansion.md](P7-002-rag-query-expansion.md) | RAG クエリ拡張（シノニム展開と search() 統合） | implemented | P7-001 |

---

## 依存関係図

```text
P7-001 (static_data/lease_domain_glossary.json + api/knowledge/domain_glossary.py)
  ↓
P7-002 (api/knowledge/query_expansion.py + vector_store.search() 統合)
```

---

## 備考

- 計画書（`代替案実装計画_精度改善Ver2.md`）は `mobile_app/` を対象としているが、現行の本番RAG経路は `api/knowledge/vector_store.py` のため、本Phaseは api 側を正とする。配置変更の理由は P7-001 の「計画書からの変更点」を参照。
- 代替案B（埋め込みモデル更新: text-embedding-3-large）は本Phaseに含めない。P7-001/002 の効果測定後に別SPECで扱う。
- REV-179（RAG信頼度スコア表示）でチャット参照ノートに信頼度バッジが付いたため、P7-002 の拡張経由ヒットは減衰スコアにより自然に低信頼度で表示される。
