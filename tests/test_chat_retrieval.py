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
