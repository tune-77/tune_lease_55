# LlamaIndex RAG 比較実験メモ

作成日: 2026-08-24

## 結論

LlamaIndex は採用前提で導入しない。既存の Obsidian RAG / OKF RAG と同じ評価セットで横並び比較し、検索品質・ノイズ量・運用コストが明確に勝つ場合だけ、限定導入を検討する。

## 目的

- Obsidian / 判断資産 / OKF 知識パックの検索精度が既存 Chroma RAG より改善するか確認する。
- 紫苑の回答品質に効く「根拠の拾い方」だけを見る。
- 本番チャット、スコアリング、判断資産昇格、Obsidian 書き込みには接続しない。

## 非目的

- LlamaIndex を本番 RAG の置き換えとして即採用しない。
- `requirements.txt` / `pyproject.toml` の本体依存へ追加しない。
- プロンプト、RAG ランキング、スコアリング、判断資産 active store を変更しない。
- 自動昇格、自動承認、自動否決には使わない。

## 比較対象

| 対象 | 役割 | 既存入口 |
|---|---|---|
| 現行 Obsidian RAG | 本番系の検索基盤 | `api/knowledge/vector_store.py` |
| OKF isolated RAG | 小さな知識パックの回帰評価 | `scripts/evaluate_okf_rag.py` |
| Vertex AI Search pilot | 外部検索基盤との比較材料 | `scripts/compare_obsidian_rag_vertex_search.py` |
| LlamaIndex 試作 | 候補。まだ未採用 | 新規 sidecar script のみ |

## 最小実験

1. LlamaIndex は別 venv または pipeline 専用環境にだけ入れる。
2. `api/knowledge/rag_eval_set.json` と `api/knowledge/okf_rag_eval_set.json` をそのまま使う。
3. Markdown を読み、LlamaIndex で一時 index を作る。
4. 各 query の top-k path を出す。
5. 既存の `evaluate_cases()` と同じ指標で集計する。
6. `reports/llamaindex_rag_comparison_latest.json` と `.md` に保存する。

## 合格条件

- `hit@5` が現行 RAG 以上。
- `forbidden_cases` が現行 RAG 以下。
- `mrr` が現行 RAG より明確に改善、または同等で検索理由が読みやすい。
- 実行時間と依存追加の重さが日次運用に耐える。
- 回答本文ではなく、検索根拠の改善として説明できる。

## 不採用条件

- hit 率が上がっても `AI Chat/`、`Humor/`、古いニュースなどの混入が増える。
- 依存が重く、Cloud Run / ローカル再現性を悪化させる。
- 既存の `google-adk`、紫苑 agentic skill、レビュー箱と役割が重なるだけになる。
- 評価セットでは良く見えるが、判断資産の出典追跡が弱くなる。

## 導入するとしても許す範囲

- 最初は `scripts/compare_llamaindex_rag.py` のような sidecar のみ。
- 出力先は `reports/` の比較レポートだけ。
- 依存は `requirements-pipeline.txt` 相当の任意依存に限定する。
- 本番 API から import しない。

## 次の一手

LlamaIndex を試すなら、最初の実装は「OKF 知識パックだけを対象にした一時 index 比較」にする。OKF は小さく、期待 path が明確で、既存 `scripts/evaluate_okf_rag.py` と比べやすい。ここで勝てなければ Obsidian 全体へ広げない。
