"""Cross-process writer lock for the shared local ChromaDB store.

The ``obsidian_knowledge`` and ``shion_memory`` collections are different, but
both live under ``api/chroma_db`` and therefore share Chroma's SQLite metadata.
This lock serializes maintenance jobs that mutate either collection.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout


DEFAULT_LOCK_PATH = Path("/tmp/tunelease-chromadb-writer.lock")
DEFAULT_TIMEOUT_SECONDS = 15 * 60
_LOCKS: dict[str, FileLock] = {}
_LOCKS_GUARD = threading.Lock()


class ChromaWriteLockTimeout(RuntimeError):
    """Raised when another ChromaDB writer did not finish within the timeout."""


def _lock_path() -> Path:
    configured = os.environ.get("TUNELEASE_CHROMA_WRITE_LOCK_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_LOCK_PATH


def _timeout_seconds() -> float:
    raw = os.environ.get("TUNELEASE_CHROMA_WRITE_LOCK_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_TIMEOUT_SECONDS)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_TIMEOUT_SECONDS)


def _shared_file_lock(path: Path) -> FileLock:
    """Return one FileLock instance per path so nested writer calls are reentrant."""

    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = FileLock(key)
            _LOCKS[key] = lock
        return lock


@contextmanager
def chroma_write_lock(
    operation: str,
    *,
    timeout: float | None = None,
    lock_path: Path | None = None,
) -> Iterator[None]:
    """Acquire the host-wide ChromaDB writer lock or raise a retryable error."""

    resolved_path = lock_path or _lock_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    wait_seconds = _timeout_seconds() if timeout is None else max(0.0, timeout)
    lock = _shared_file_lock(resolved_path)
    try:
        with lock.acquire(timeout=wait_seconds):
            yield
    except Timeout as exc:
        raise ChromaWriteLockTimeout(
            f"ChromaDB writer lock timeout: operation={operation} "
            f"timeout={wait_seconds:g}s path={resolved_path}"
        ) from exc
