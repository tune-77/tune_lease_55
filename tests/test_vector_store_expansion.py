"""P7-002 クエリ拡張の search() 統合テスト（AC-713〜715）。

Chroma は使わず、フェイク collection と _keyword_search の差し替えで検証する。
シード辞書の「赤字企業」グループ（synonyms に「営業赤字」を含む）を前提とする。
"""
import pytest

from api.knowledge.vector_store import KnowledgeVectorStore


def _config(enabled: bool) -> dict:
    return {
        "preferred_path_boosts": {},
        "low_priority_path_penalties": {},
        "sync_copy_penalty": 0.35,
        "keyword_pool_multiplier": 4,
        "keyword_pool_min": 12,
        "query_expansion_enabled": enabled,
        "query_expansion_max_variants": 4,
        "query_expansion_decay": 0.4,
    }


class _FakeCollection:
    """ベクトル検索が常に空を返す最小コレクション。"""

    def count(self):
        return 1

    def query(self, **_kwargs):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _note_hit(score: int) -> dict:
    return {
        "doc_id": "doc-1",
        "text": "当社の営業赤字の取引先対応メモ",
        "ref": "リース知識/営業赤字の対応.md#方針",
        "file_name": "営業赤字の対応.md",
        "file_path": "リース知識/営業赤字の対応.md",
        "section": "方針",
        "wikilinks": "",
        "metadata": {},
        "score": score,
        "source": "keyword",
    }


def _make_store(enabled: bool, keyword_results: dict[str, list[dict]]) -> KnowledgeVectorStore:
    """query 文字列に応じた _keyword_search 結果を返すストアを組み立てる。"""
    store = KnowledgeVectorStore(
        chroma_dir="/tmp/unused-rag-expansion-test",
        ranking_config=_config(enabled),
    )
    store._collection = _FakeCollection()
    store._encoder = object()

    def _fake_keyword_search(query: str, top_k: int) -> list[dict]:
        for needle, hits in keyword_results.items():
            if needle in query:
                return [dict(h) for h in hits]
        return []

    store._keyword_search = _fake_keyword_search
    store._embed = lambda texts: [[0.0]]
    return store


def test_ac713_expanded_hit_supplements_results(monkeypatch, tmp_path):
    # 「営業赤字」ノートのみ存在し、クエリは「赤字企業」→ 拡張経由でヒットする
    store = _make_store(True, {"営業赤字": [_note_hit(score=10)]})
    monkeypatch.setattr(
        "api.knowledge.vector_store._SEARCH_LOG_PATH", str(tmp_path / "log.jsonl")
    )

    results = store.search("赤字企業の対応について")

    assert results, "拡張クエリ経由でヒットするはず"
    hit = results[0]
    assert hit["expanded_from"] == "赤字企業"
    assert hit["source"] == "keyword_expanded"
    # 減衰確認: score 10 → 10 * (0.4 * 1.0) = 4.0 → base = 4/20 = 0.2
    assert hit["score"] == pytest.approx(4.0)


def test_ac714_disabled_expansion_matches_legacy_behavior(monkeypatch, tmp_path):
    store = _make_store(False, {"営業赤字": [_note_hit(score=10)]})
    monkeypatch.setattr(
        "api.knowledge.vector_store._SEARCH_LOG_PATH", str(tmp_path / "log.jsonl")
    )

    results = store.search("赤字企業の対応について")

    assert results == []  # 従来どおり: 元クエリではヒットなし


def test_ac715_original_hit_score_is_kept(monkeypatch, tmp_path):
    # 元クエリでも拡張クエリでも同じノートがヒットするケース
    keyword_results = {
        "赤字企業": [_note_hit(score=10)],
        "営業赤字": [_note_hit(score=8)],
    }
    monkeypatch.setattr(
        "api.knowledge.vector_store._SEARCH_LOG_PATH", str(tmp_path / "log.jsonl")
    )

    enabled_results = _make_store(True, keyword_results).search("赤字企業の対応について")
    disabled_results = _make_store(False, keyword_results).search("赤字企業の対応について")

    assert len(enabled_results) == 1  # 重複しない
    assert len(disabled_results) == 1
    # 元クエリヒットのスコア・ランクは拡張の有無で変わらない（max維持・加算なし）
    assert enabled_results[0]["score"] == disabled_results[0]["score"]
    assert enabled_results[0]["rank_score"] == disabled_results[0]["rank_score"]
    assert "expanded_from" not in enabled_results[0]
