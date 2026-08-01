"""hybrid_search.py へのクエリ拡張統合（P7-002 フォローアップ）のテスト。

mobile_app.hybrid_search.HybridSearchEngine._expand_scores を、実際の
SemanticRAGRetriever/BM25 初期化を経由せず単体でテストする
（重いモデルロードに依存しないため、フェイクの retriever/engine を注入する）。
"""
from __future__ import annotations

from mobile_app.hybrid_search import HybridSearchEngine


class _FakeSemanticRetriever:
    def __init__(self, docs_by_query):
        self._docs_by_query = docs_by_query
        self.calls: list[str] = []

    def retrieve(self, query, top_k=5):
        self.calls.append(query)
        return self._docs_by_query.get(query, [])


class _FakeBM25Engine:
    def __init__(self, hits_by_query):
        self._hits_by_query = hits_by_query
        self.calls: list[str] = []

    def search(self, query, top_k=5):
        self.calls.append(query)
        return self._hits_by_query.get(query, [])


def _bare_engine(**overrides) -> HybridSearchEngine:
    """__init__ をバイパスして _expand_scores 単体テスト用のインスタンスを作る。"""
    engine = object.__new__(HybridSearchEngine)
    engine.enable_query_expansion = True
    engine.query_expansion_max_variants = 4
    engine.query_expansion_decay = 0.4
    engine.semantic_available = False
    engine.bm25_available = False
    engine.semantic_retriever = None
    engine.bm25_engine = None
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


def test_expand_scores_surfaces_synonym_only_semantic_hit():
    # クエリは「飲食店」（辞書のsynonym）だが、ドキュメントは「飲食業」（canonical）でのみヒットする状況を模す。
    fake_retriever = _FakeSemanticRetriever({
        "飲食業のリース審査ポイント": [{"path": "lease/restaurant", "similarity_score": 0.9}],
    })
    engine = _bare_engine(semantic_available=True, semantic_retriever=fake_retriever)

    semantic_scores: dict = {}
    bm25_scores: dict = {}
    engine._expand_scores("飲食店のリース審査ポイント", 5, semantic_scores, bm25_scores)

    assert "lease/restaurant" in semantic_scores
    # BR-711 相当: 拡張ヒットは decay(0.4) × group_weight(0.95) 以下に減衰している
    assert 0.0 < semantic_scores["lease/restaurant"] < 0.9


def test_expand_scores_does_not_overwrite_original_query_hit():
    # 元クエリで既にヒットしているdoc_idは、拡張ヒットで上書きされない（BR-711）。
    fake_retriever = _FakeSemanticRetriever({
        "飲食業のリース審査ポイント": [{"path": "lease/restaurant", "similarity_score": 0.5}],
    })
    engine = _bare_engine(semantic_available=True, semantic_retriever=fake_retriever)

    semantic_scores = {"lease/restaurant": 0.95}
    engine._expand_scores("飲食店のリース審査ポイント", 5, semantic_scores, {})

    assert semantic_scores["lease/restaurant"] == 0.95


def test_expand_scores_bm25_side_uses_decayed_weight():
    fake_bm25 = _FakeBM25Engine({
        "飲食業のリース審査ポイント": [({"path": "lease/restaurant"}, 10.0)],
    })
    engine = _bare_engine(bm25_available=True, bm25_engine=fake_bm25)

    bm25_scores: dict = {}
    engine._expand_scores("飲食店のリース審査ポイント", 5, {}, bm25_scores)

    assert 0.0 < bm25_scores["lease/restaurant"] < 10.0


def test_expand_scores_noop_when_no_glossary_match():
    fake_retriever = _FakeSemanticRetriever({})
    engine = _bare_engine(semantic_available=True, semantic_retriever=fake_retriever)

    semantic_scores: dict = {}
    engine._expand_scores("存在しない企業名 XYZ123", 5, semantic_scores, {})

    assert semantic_scores == {}
    assert fake_retriever.calls == []


def test_expand_scores_swallows_retriever_exception():
    class _BoomRetriever:
        def retrieve(self, query, top_k=5):
            raise RuntimeError("boom")

    engine = _bare_engine(semantic_available=True, semantic_retriever=_BoomRetriever())

    semantic_scores: dict = {}
    # 例外を握りつぶし、呼び出し元へは伝播しない（BR-712 相当）
    engine._expand_scores("飲食店のリース審査ポイント", 5, semantic_scores, {})
    assert semantic_scores == {}


def test_get_stats_reports_query_expansion_flag():
    engine = _bare_engine(
        documents=[],
        enable_query_expansion=False,
        semantic_weight=0.6,
        bm25_weight=0.4,
    )
    stats = HybridSearchEngine.get_stats(engine)
    assert stats["query_expansion_enabled"] is False
