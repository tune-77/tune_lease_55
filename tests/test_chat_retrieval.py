import sys
import types
from pathlib import Path

from api.chat_retrieval import (
    build_chat_retrieval_context,
    chat_memory_roots,
    display_vault_ref,
    extract_vertex_search_hint,
    search_chat_vault_markdown_fallback,
)


def test_extract_vertex_search_hint_uses_compact_marker_block():
    message = "\n".join(
        [
            "【審査分析画面からの紫苑レビュー依頼】",
            "長いレビュー本文",
            "",
            "【Vertex補助検索ヒント】",
            "工作機械 補助金 リース料軽減",
            "公募要領 対象経費 Q_risk",
            "",
            "前提: スコア55点",
        ]
    )

    assert extract_vertex_search_hint(message) == "工作機械 補助金 リース料軽減 公募要領 対象経費 Q_risk"


def test_build_chat_retrieval_context_zero_top_k_returns_stable_shape():
    result = build_chat_retrieval_context(
        "補助金前提の工作機械リース",
        rag_top_k=0,
        question_category="lease_knowledge",
        is_general_response_mode=False,
        fallback_search=lambda _message, _top_k: [{"text": "unused"}],
    )

    assert result.rag_context == ""
    assert result.rag_refs == []
    assert result.rag_knowledge_refs == []
    assert result.vertex_agent_search["status"] == "not_attempted"
    assert result.vertex_answer_api["status"] == "not_attempted"


def test_chat_memory_roots_deduplicates_configured_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("GCS_VAULT_LOCAL_DIR", str(tmp_path / "gcs"))
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("OBSIDIAN_VAULT", str(tmp_path / "vault"))

    roots = chat_memory_roots(str(tmp_path / "vault"))

    assert roots[:2] == [tmp_path / "gcs", tmp_path / "vault"]
    assert roots.count(tmp_path / "vault") == 1


def test_display_vault_ref_uses_obsidian_wikilink():
    root = Path("/vault")

    assert display_vault_ref(root, "/vault/Projects/a.md", "結論") == "[[Projects/a#結論]]"
    assert display_vault_ref(root, "/other/a.md") == "[[a]]"


def test_markdown_fallback_search_keeps_legacy_hit_shape(monkeypatch, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    chunk = types.SimpleNamespace(
        doc_id="doc-1",
        text="工作機械 補助金 リース 注意",
        file_path=str(vault / "Projects/tune_lease_55/Research/note.md"),
        file_name="note.md",
        section="結論",
        mtime=123.0,
    )
    monkeypatch.setitem(
        sys.modules,
        "api.knowledge.obsidian_loader",
        types.SimpleNamespace(scan_vault=lambda _root: [chunk]),
    )
    monkeypatch.setitem(
        sys.modules,
        "obsidian_query",
        types.SimpleNamespace(split_query_terms=lambda _query: ["工作機械", "補助金"]),
    )

    hits = search_chat_vault_markdown_fallback(
        "工作機械 補助金",
        top_k=1,
        obsidian_vault_path=str(vault),
    )

    assert hits == [
        {
            "doc_id": "doc-1",
            "text": "工作機械 補助金 リース 注意",
            "ref": "[[Projects/tune_lease_55/Research/note#結論]]",
            "file_name": "note.md",
            "file_path": str(vault / "Projects/tune_lease_55/Research/note.md"),
            "section": "結論",
            "mtime": 123.0,
            "score": 22.29,
            "source": "vault_markdown_fallback",
        }
    ]


def test_main_chat_rag_uses_typesafe_candidate_gate(monkeypatch):
    seen = {}
    raw_hits = [
        {"doc_id": "1", "text": "weak", "ref": "[[weak]]", "file_name": "weak.md"},
        {"doc_id": "2", "text": "strong", "ref": "[[strong]]", "file_name": "strong.md"},
        {"doc_id": "3", "text": "attack", "ref": "[[attack]]", "file_name": "attack.md"},
        {"doc_id": "4", "text": "extra", "ref": "[[extra]]", "file_name": "extra.md"},
    ]

    class FakeStore:
        def search(self, _message, top_k):
            seen["top_k"] = top_k
            return raw_hits[:top_k]

    monkeypatch.setitem(
        sys.modules,
        "api.knowledge.vector_store",
        types.SimpleNamespace(
            get_store=lambda: FakeStore(),
            confidence_for_hit=lambda _hit: (0.9, "high"),
        ),
    )
    monkeypatch.setattr(
        "api.chat_retrieval._typesafe_rag_filter",
        lambda _message: lambda _query, hits: (
            [{**hits[1], "typesafe_route": "include"}],
            {
                "status": "applied",
                "model": "jev-test",
                "candidate_count": len(hits),
                "accepted_count": 1,
                "excluded_count": len(hits) - 1,
            },
        ),
    )

    result = build_chat_retrieval_context(
        "query",
        rag_top_k=2,
        question_category="general",
        is_general_response_mode=False,
    )

    assert seen["top_k"] == 4
    assert "strong" in result.rag_context
    assert "weak" not in result.rag_context
    assert result.rag_refs == ["[[strong]]"]
    assert result.typesafe_rag["status"] == "applied"


def test_screening_rag_does_not_leave_process_by_default(monkeypatch):
    monkeypatch.delenv("TYPESAFE_ALLOW_SCREENING", raising=False)
    monkeypatch.setattr(
        "api.chat_retrieval._typesafe_rag_filter",
        lambda _message: (_ for _ in ()).throw(AssertionError("must not enable external gate")),
    )
    monkeypatch.setitem(
        sys.modules,
        "api.knowledge.vector_store",
        types.SimpleNamespace(
            get_store=lambda: types.SimpleNamespace(
                search=lambda _message, top_k: [
                    {"doc_id": "1", "text": "local", "ref": "[[local]]", "file_name": "local.md"}
                ][:top_k]
            ),
            confidence_for_hit=lambda _hit: (0.9, "high"),
        ),
    )

    result = build_chat_retrieval_context(
        "A社の案件を審査して",
        rag_top_k=2,
        question_category="lease_screening",
        is_general_response_mode=True,
    )

    assert "local" in result.rag_context
    assert result.typesafe_rag["status"] == "disabled"


def test_sensitive_query_skips_typesafe_even_when_baseline_category_is_wrong(monkeypatch):
    monkeypatch.setenv("TYPESAFE_RAG_ENABLED", "1")
    monkeypatch.delenv("TYPESAFE_ALLOW_SCREENING", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "api.knowledge.vector_store",
        types.SimpleNamespace(
            get_store=lambda: types.SimpleNamespace(
                search=lambda _message, top_k: [
                    {"doc_id": "1", "text": "local", "ref": "[[local]]", "file_name": "local.md"}
                ][:top_k]
            ),
            confidence_for_hit=lambda _hit: (0.9, "high"),
        ),
    )

    result = build_chat_retrieval_context(
        "A社案件の売上と財務を確認して",
        rag_top_k=2,
        question_category="lease_knowledge",
        is_general_response_mode=True,
    )

    assert "local" in result.rag_context
    assert result.typesafe_rag["status"] == "disabled"
