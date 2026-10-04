from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import types

import pytest
from filelock import FileLock

from api.knowledge.chroma_write_lock import (
    ChromaWriteLockTimeout,
    _timeout_seconds,
    chroma_write_lock,
)


def test_chroma_write_lock_times_out_when_another_writer_holds_it(tmp_path: Path) -> None:
    lock_path = tmp_path / "chroma-writer.lock"
    owner = FileLock(str(lock_path))

    with owner.acquire(timeout=0):
        with pytest.raises(ChromaWriteLockTimeout, match="memory_sync"):
            with chroma_write_lock("memory_sync", timeout=0, lock_path=lock_path):
                pytest.fail("contending writer must not enter the critical section")


def test_chroma_write_lock_releases_after_failure(tmp_path: Path) -> None:
    lock_path = tmp_path / "chroma-writer.lock"

    with pytest.raises(RuntimeError, match="boom"):
        with chroma_write_lock("first", timeout=0, lock_path=lock_path):
            raise RuntimeError("boom")

    with chroma_write_lock("second", timeout=0, lock_path=lock_path):
        acquired_again = True

    assert acquired_again is True


def test_chroma_write_lock_is_reentrant_for_nested_writer_calls(tmp_path: Path) -> None:
    lock_path = tmp_path / "chroma-writer.lock"

    with chroma_write_lock("full_rebuild", timeout=0, lock_path=lock_path):
        with chroma_write_lock("batch_upsert", timeout=0, lock_path=lock_path):
            nested_writer_entered = True

    assert nested_writer_entered is True


def test_chroma_write_lock_reads_configured_timeout(monkeypatch) -> None:
    monkeypatch.setenv("TUNELEASE_CHROMA_WRITE_LOCK_TIMEOUT_SECONDS", "2.5")

    assert _timeout_seconds() == 2.5


def test_writer_entry_points_share_the_common_lock() -> None:
    maintenance = Path("mobile_app/rag_daily_maintenance.py").read_text(encoding="utf-8")
    memory_vector = Path("api/shion_memory_vector.py").read_text(encoding="utf-8")
    vector_store = Path("api/knowledge/vector_store.py").read_text(encoding="utf-8")
    feedback_store = Path("api/knowledge/feedback_watcher.py").read_text(encoding="utf-8")
    direct_reindex = Path("scripts/reindex_obsidian.py").read_text(encoding="utf-8")
    reindex_runner = Path("scripts/run_obsidian_reindex.sh").read_text(encoding="utf-8")

    assert 'chroma_write_lock("obsidian_reindex")' in maintenance
    assert 'chroma_write_lock("shion_memory_sync")' in memory_vector
    assert 'chroma_write_lock("obsidian_knowledge_upsert")' in vector_store
    assert 'chroma_write_lock("obsidian_knowledge_delete")' in vector_store
    assert 'chroma_write_lock("obsidian_knowledge_initialize", timeout=timeout)' in vector_store
    assert 'chroma_write_lock("lease_feedback_upsert")' in feedback_store
    assert 'chroma_write_lock("lease_feedback_initialize")' in feedback_store
    assert 'chroma_write_lock("obsidian_full_reindex")' in direct_reindex
    assert 'report.get("status") == "deferred"' in maintenance
    assert "REINDEX_EXIT -ne 75" in reindex_runner
    assert "GCS sync もスキップ" in reindex_runner
    assert 'REINDEX_MAX_ATTEMPTS="${REINDEX_MAX_ATTEMPTS:-2}"' in reindex_runner
    assert 'sleep "$REINDEX_RETRY_DELAY_SECONDS"' in reindex_runner


def test_obsidian_reindex_reports_retryable_defer_on_lock_timeout(monkeypatch) -> None:
    from mobile_app import rag_daily_maintenance as maintenance

    @contextmanager
    def busy_writer(_operation: str):
        raise ChromaWriteLockTimeout("busy")
        yield

    monkeypatch.setattr(maintenance, "chroma_write_lock", busy_writer)

    result = maintenance.run_chroma_reindex("/unused")

    assert result["status"] == "deferred"
    assert result["reason"] == "chroma_writer_busy"
    assert result["retryable"] is True


def test_first_obsidian_search_does_not_wait_for_active_writer(tmp_path, monkeypatch) -> None:
    from api.knowledge import chroma_write_lock as lock_module
    from api.knowledge.vector_store import KnowledgeVectorStore

    lock_calls: list[float | None] = []

    @contextmanager
    def busy_writer(_operation: str, *, timeout=None):
        lock_calls.append(timeout)
        raise ChromaWriteLockTimeout("busy")
        yield

    monkeypatch.setattr(lock_module, "chroma_write_lock", busy_writer)
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(PersistentClient=lambda **_kwargs: pytest.fail("must not initialize")),
    )
    store = KnowledgeVectorStore(chroma_dir=str(tmp_path))

    assert store.search("再リース") == []
    assert lock_calls == [0]

    writer_store = KnowledgeVectorStore(chroma_dir=str(tmp_path / "writer"))
    with pytest.raises(ChromaWriteLockTimeout, match="busy"):
        writer_store._ensure_collection()
    assert lock_calls == [0, None]


def test_direct_reindex_cli_returns_tempfail_on_writer_contention(monkeypatch) -> None:
    from scripts import reindex_obsidian

    monkeypatch.setattr(sys, "argv", ["reindex_obsidian.py", "--full", "--vault", "/unused"])
    monkeypatch.setattr(
        reindex_obsidian,
        "full_reindex",
        lambda _vault: (_ for _ in ()).throw(ChromaWriteLockTimeout("busy")),
    )

    assert reindex_obsidian.main() == 75
