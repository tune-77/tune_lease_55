import sys
import types

from api.lease_intelligence_connection import (
    build_lease_intelligence_knowledge_connection,
    count_markdown_notes,
    lease_intelligence_indexed_notes_for_compat,
)


class _FakeStore:
    def __init__(self, count: int):
        self._count = count

    def count(self) -> int:
        return self._count


def test_count_markdown_notes_counts_markdown_only(tmp_path):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("c", encoding="utf-8")

    assert count_markdown_notes(tmp_path) == 2


def test_knowledge_connection_prefers_vector_store(monkeypatch, tmp_path):
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "api.knowledge.vector_store",
        types.SimpleNamespace(get_store=lambda: _FakeStore(7)),
    )
    monkeypatch.setenv("K_SERVICE", "lease-api")

    connection = build_lease_intelligence_knowledge_connection(None, repo_root=tmp_path)

    assert connection["source"] == "knowledge_vector_db"
    assert connection["label"] == "知識DB検索可能: 7チャンク"
    assert connection["is_cloud_run"] is True
    assert connection["vector_chunks"] == 7


def test_knowledge_connection_uses_markdown_when_vector_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "api.knowledge.vector_store",
        types.SimpleNamespace(get_store=lambda: _FakeStore(0)),
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("note", encoding="utf-8")

    connection = build_lease_intelligence_knowledge_connection(vault, repo_root=tmp_path)

    assert connection["source"] == "markdown_vault_copy"
    assert connection["markdown_notes"] == 1
    assert connection["markdown_root"] == str(vault)


def test_indexed_notes_compat_uses_first_available_count():
    assert lease_intelligence_indexed_notes_for_compat({"markdown_notes": 2, "vector_chunks": 7}) == 2
    assert lease_intelligence_indexed_notes_for_compat({"markdown_notes": 0, "vector_chunks": 7}) == 7
    assert lease_intelligence_indexed_notes_for_compat({"case_count": 3}) == 3
    assert lease_intelligence_indexed_notes_for_compat({}) == 0
