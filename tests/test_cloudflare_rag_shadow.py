import json
import sys
import types

from api import cloudflare_rag_shadow as shadow


def test_eligibility_allows_generic_lease_knowledge(monkeypatch):
    monkeypatch.setattr(shadow, "shadow_enabled", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "obsidian_query",
        types.SimpleNamespace(split_query_terms=lambda _text: ["再リース", "満了", "選択肢"]),
    )
    allowed, reason, query = shadow.eligible_shadow_query(
        "再リースするときのリスクと満了後の選択肢を知りたい",
        question_category="lease_knowledge",
        is_general_response_mode=False,
    )
    assert allowed and reason == "eligible"
    assert query == "再リース 満了 選択肢"


def test_eligibility_rejects_screening_and_pii(monkeypatch):
    monkeypatch.setattr(shadow, "shadow_enabled", lambda: True)
    for message in (
        "株式会社山田の案件を審査して",
        "売上1億円の申込先を承認できますか",
        "担当者: 山田太郎の再リース相談",
    ):
        allowed, reason, query = shadow.eligible_shadow_query(
            message,
            question_category="lease_knowledge",
            is_general_response_mode=False,
        )
        assert not allowed
        assert reason in {"sensitive", "redaction_required"}
        assert query == ""


def test_eligibility_rejects_non_knowledge_categories(monkeypatch):
    monkeypatch.setattr(shadow, "shadow_enabled", lambda: True)
    allowed, reason, _query = shadow.eligible_shadow_query(
        "今日はどう？",
        question_category="general",
        is_general_response_mode=False,
    )
    assert not allowed and reason == "category_excluded"


def test_external_query_keeps_only_allowlisted_domain_terms(monkeypatch):
    monkeypatch.setattr(shadow, "shadow_enabled", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "obsidian_query",
        types.SimpleNamespace(split_query_terms=lambda _text: ["山田", "再リース", "満了"]),
    )
    allowed, _reason, query = shadow.eligible_shadow_query(
        "再リースと満了について知りたい",
        question_category="lease_knowledge",
        is_general_response_mode=False,
    )
    assert allowed
    assert query == "再リース 満了"
    assert "山田" not in query


def test_shadow_is_disabled_implicitly_during_pytest():
    assert not shadow.shadow_enabled({"PYTEST_CURRENT_TEST": "case"})


def test_compare_logs_only_safe_refs_and_rank_differences(monkeypatch, tmp_path):
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("CLOUDFLARE_RAG_SHADOW_LOG_PATH", str(log_path))

    class Client:
        def query(self, _vector, top_k):
            assert top_k == 10
            return [
                {"id": "v1", "metadata": {"path": "safe/a.md", "text": "A"}},
                {"id": "v2", "metadata": {"path": "safe/b.md", "text": "B"}},
            ]

    class Embedder:
        def embed(self, texts, kind):
            assert texts == ["再リース 満了"] and kind == "query"
            return [[0.1, 0.2]]

    class Reranker:
        def rerank(self, _query, matches, top_k):
            assert top_k == 5
            return [matches[1], matches[0]]

    factory = lambda: (Client(), {"embedder": Embedder(), "reranker": Reranker()})
    entry = shadow.compare_cloudflare_shadow(
        "再リース 満了",
        [{"ref": "[[safe/a#section]]", "file_path": "/private/path/must-not-log.md", "text": "secret"}],
        client_factory=factory,
    )

    assert entry["local_refs"] == ["[[safe/a#section]]"]
    assert entry["vectorize_refs"] == ["safe/a.md", "safe/b.md"]
    assert entry["reranker_refs"] == ["safe/b.md", "safe/a.md"]
    assert entry["local_vectorize_overlap_at_5"] == 1
    body = log_path.read_text(encoding="utf-8")
    assert "/private/path" not in body and "secret" not in body
    assert json.loads(body)["status"] == "ok"
