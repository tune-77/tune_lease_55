---
spec_id: P7-002
phase: 7
title: RAG クエリ拡張（ドメイン辞書によるシノニム展開）
status: implemented
author: Claude Fable
reviewer: "tune-77"
version: "1.0"
created: 2026-07-09
updated: 2026-07-09
depends_on: [P7-001]
superseded_by: ""
---

# P7-002 — RAG クエリ拡張（ドメイン辞書によるシノニム展開）

---

## 1. Goal

P7-001 のドメイン辞書を使い、RAG検索クエリをシノニムで自動展開して検索の取りこぼしを減らす。「飲食業の赤字企業」で「飲食店の営業赤字」を書いたノートもヒットさせる。元クエリの結果を常に優先し、拡張ヒットは減衰スコアで補完に徹する。

> 出典: `代替案実装計画_精度改善Ver2.md` 代替案A「検索時の辞書活用（拡張クエリ生成）」「Hybrid Search への統合」。

---

## 2. Scope

### In scope
- クエリ拡張モジュール `api/knowledge/query_expansion.py`（`expand_query()`）
- `api/knowledge/vector_store.py` の `KnowledgeVectorStore.search()` への統合（キーワード検索の語彙拡張 + 埋め込み検索のマルチクエリ）
- `config/rag_ranking.json` への拡張設定キー追加（有効/無効・上限・減衰率）
- 単体テスト・統合テスト

### Out of scope
- 辞書データそのものの追加・変更（P7-001）
- `mobile_app/hybrid_search.py` への統合（計画書の対象だが、現行本番経路は `api/knowledge/` のため後続とする）
- `api/main.py` の `_search_chat_vault_markdown_fallback`（フォールバック検索への適用は任意の後続改善）
- LLM によるクエリリライト（辞書ベースのみ。説明可能性を優先）

---

## 3. Inputs / Outputs

### Inputs
| 項目名 | 型 | 必須/任意 | 説明 |
|-------|-----|----------|------|
| query | str | 必須 | 元の検索クエリ（自然文） |
| max_variants | int | 任意 | 拡張バリエーション上限（既定 4、元クエリ含め最大 5） |

### Outputs
| 項目名 | 型 | 説明 |
|-------|-----|------|
| expand_query() 戻り値 | list[ExpandedQuery] | 先頭は必ず元クエリ（weight=1.0）。以降はシノニム置換版（weight=グループweight×減衰率） |

---

## 4. Data Model

```python
class ExpandedQuery(TypedDict):
    query: str       # 展開後クエリ文字列
    weight: float    # スコア係数。元クエリ=1.0、拡張=expansion_decay × group_weight
    replaced: str    # 置換した語（デバッグ・検索ログ用。元クエリは ""）
```

### config/rag_ranking.json 追加キー（既存キーは変更しない）

```json
{
  "query_expansion_enabled": true,
  "query_expansion_max_variants": 4,
  "query_expansion_decay": 0.4
}
```

- `query_expansion_decay: 0.4` は計画書「スコアを 0.3-0.5 に減衰」の中央値。

---

## 5. API / Interface

```python
# api/knowledge/query_expansion.py（新規）
def expand_query(query: str, max_variants: int = 4) -> list[ExpandedQuery]:
    """P7-001 の synonyms_for() を用いてクエリを展開する。

    - クエリに canonical / synonym が含まれるグループを検出し、
      他の語に置換したバリエーションを weight 降順に生成する
    - 元クエリを必ず先頭（weight=1.0）に置く
    - 該当語がなければ元クエリのみを返す（無害なno-op）
    """
```

### KnowledgeVectorStore.search() への統合方針

```
search(query)
├─ 1. 元クエリで従来どおり検索（ベクトル + キーワードマージ）        … 変更なし
├─ 2. query_expansion_enabled のとき:
│     expand_query() の拡張クエリ（元除く）それぞれで
│     _keyword_search(expanded, top_k=3) を実行し、
│     hit["score"] *= expanded.weight で減衰
│     （埋め込み側は拡張1本目のみ任意適用。レイテンシ優先）
├─ 3. 既存のマージ辞書（(display_path, section) キー）に統合
│     既出ヒットは score の最大値維持（計画書 _merge_results と同等）
└─ 4. 既存の _rerank_hits() で最終ランク           … 変更なし
```

- 検索ログ（`_write_search_log`）に `expanded_from` を追記し、拡張経由ヒットを追跡可能にする。

---

## 6. Business Rules

**BR-711**: 元クエリ優先の原則
- 条件：常時
- 処理：拡張クエリのヒットは必ず weight（≤0.5）で減衰し、元クエリの同一ヒットのスコアを上書きしない（max 維持）
- 根拠：計画書「オリジナルクエリの重み 1.0 / 拡張 0.3-0.5」。拡張が本来の検索意図を押しのけないため

**BR-712**: 設定で完全に無効化できる
- 条件：`query_expansion_enabled: false` または辞書ロード失敗時
- 処理：search() は従来と完全に同一の動作・結果になる
- 根拠：P7-001 BR-703 と同じフェイルセーフ方針・後方互換

**BR-713**: レイテンシ上限
- 条件：拡張検索の追加処理
- 処理：拡張は `_keyword_search`（Chroma get + 文字列照合）中心とし、追加の埋め込み計算は最大1回まで
- 根拠：`scoring_core.py` の教訓（毎リクエスト重処理でスレッドプール枯渇）。チャット応答レイテンシを悪化させない

---

## 7. UI / UX

フロントエンド変更なし。ただし拡張経由ヒットも REV-179 の信頼度バッジ（`confidence_for_hit`）を通るため、減衰スコアにより自然に低めの信頼度で表示される（追加実装不要であることをテストで確認する）。

---

## 8. Error Handling

| エラー条件 | 処理 | ユーザー向けメッセージ |
|-----------|------|---------------------|
| 辞書ロード失敗 | 拡張なしで通常検索を継続 | （表示なし） |
| 拡張クエリの検索で例外 | 当該バリエーションをスキップしログ記録 | （表示なし） |
| max_variants ≤ 0 | 元クエリのみ返す | （表示なし） |

---

## 9. Acceptance Criteria

**AC-711**: 基本拡張
- Given: 辞書に「赤字企業」グループ（synonyms: 営業赤字, 当期赤字, …）と「飲食業」グループ（synonyms: 飲食店, …）
- When: `expand_query("飲食業の赤字企業")` を呼ぶ
- Then: 先頭が元クエリ（weight=1.0）で、「飲食店の赤字企業」「飲食業の営業赤字」等が weight<1.0 で含まれ、総数は max_variants+1 以下

**AC-712**: 該当なし no-op
- Given: 辞書のどの語も含まないクエリ「今日の天気」
- When: `expand_query("今日の天気")` を呼ぶ
- Then: 元クエリ1件のみが返る

**AC-713**: 検索統合（拡張ヒットの補完）
- Given: 「営業赤字」を含むノートだけがインデックスにあり、クエリは「赤字企業について」
- When: `search()` を拡張有効で呼ぶ
- Then: 当該ノートがヒットし、その score は元クエリ直接ヒット相当より低い

**AC-714**: 無効化時の完全一致
- Given: 同一インデックス・同一クエリ
- When: `query_expansion_enabled: false` で `search()` を呼ぶ
- Then: 拡張実装前と同一の結果（件数・順序・スコア）が返る

**AC-715**: 元クエリヒットのスコア不変
- Given: 元クエリでも拡張クエリでもヒットする同一ノート
- When: `search()` を拡張有効で呼ぶ
- Then: 当該ノートのスコアは元クエリ単独時と同じ（max 維持、加算しない）

---

## 10. Non-Functional Requirements

- **パフォーマンス**: 拡張有効時の search() レイテンシ増加は +30% 以内（キーワード照合のみのため）
- **後方互換性**: 無効時は完全に従来動作。`config/rag_ranking.json` の既存キーは変更しない
- **説明可能性**: 検索ログから「どの語の展開でヒットしたか」を追跡できる
- **テストカバレッジ**: AC-711〜715 全件

---

## 11. Implementation Notes（実装者向け）

- **触れてはいけないファイル**: `scoring_core.py` ほかスコアリング一式、`api/main.py`（search() の内部改善で完結させ、エンドポイント側は無変更）
- **統合位置**: `vector_store.py` の `search()` 内、キーワードマージ（`keyword_hits` 取得）直後に拡張ヒットを追加するのが最小差分。マージキーは既存の `(self._display_path(hit), section)` を再利用する
- **設定読込**: 既存の `_maybe_reload_ranking_config()` に追加キーを読ませる（新規の読込機構を作らない）
- **REV-179 との関係**: 拡張ヒットにも `rank_score` が付くため `confidence_for_hit()` はそのまま機能する。専用対応不要
- **テストファイル**: `tests/test_query_expansion.py`（AC-711/712）、`tests/test_vector_store_expansion.py`（AC-713〜715、Chroma はインメモリまたはモック）

---

## 12. Test Plan

### 単体テスト
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_711 | AC-711 | シノニム展開・weight・上限 |
| test_712 | AC-712 | 該当語なしの no-op |
| test_713 | AC-713 | 拡張経由ヒットの補完と減衰 |
| test_714 | AC-714 | 無効時の完全後方互換 |
| test_715 | AC-715 | 重複ヒットの max 維持 |

### 回帰テスト
- `tests/test_rag_confidence.py` を含む既存 RAG テストが全件パス
- 拡張無効設定での検索結果スナップショット比較

### 手動確認（実装後）
- [ ] 紫苑チャットで「建機のリース」と質問し、「建設機械」ノートが参照される
- [ ] 検索ログ（`data/rag_search_log.jsonl`）に expanded_from が記録される
- [ ] 応答レイテンシが体感悪化しない
