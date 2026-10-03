from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from filelock import FileLock

from api.knowledge.chroma_write_lock import (
    ChromaWriteLockTimeout,
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


def test_writer_entry_points_share_the_common_lock() -> None:
    maintenance = Path("mobile_app/rag_daily_maintenance.py").read_text(encoding="utf-8")
    memory_vector = Path("api/shion_memory_vector.py").read_text(encoding="utf-8")
    reindex_runner = Path("scripts/run_obsidian_reindex.sh").read_text(encoding="utf-8")

    assert 'chroma_write_lock("obsidian_reindex")' in maintenance
    assert 'chroma_write_lock("shion_memory_sync")' in memory_vector
    assert 'report.get("status") == "deferred"' in maintenance
    assert "REINDEX_EXIT -ne 75" in reindex_runner
    assert "GCS sync もスキップ" in reindex_runner


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
