"""
Obsidian Vault のインデックス化スクリプト。
起動時にバックグラウンドスレッドで実行（メインスレッドをブロックしない）。
初回は全件、2回目以降はファイル更新日時で差分のみ更新。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from api.knowledge.obsidian_loader import scan_vault, _VAULT_PATH
from api.knowledge.vector_store import get_store

logger = logging.getLogger(__name__)

# バッチサイズ（大きな Vault でも OOM しないよう分割 upsert）
_BATCH_SIZE = 50


def _get_indexed_mtimes() -> dict[str, float]:
    """ChromaDB に保存済みの doc_id → mtime マップを取得する。"""
    store = get_store()
    try:
        store._ensure_collection()
        result = store._collection.get(include=["metadatas"])
        return {
            doc_id: float(meta.get("mtime", 0))
            for doc_id, meta in zip(result["ids"], result["metadatas"])
        }
    except Exception:
        return {}


def run_indexing(
    vault_path: str = _VAULT_PATH,
    on_done: Callable[[int, int], None] | None = None,
    prune_missing: bool = False,
) -> tuple[int, int]:
    """
    Vault をスキャンして差分チャンクを upsert する。

    Args:
        vault_path: Obsidian Vault のパス
        on_done:    完了コールバック (added, skipped) -> None

    Returns:
        (added_count, skipped_count)
    """
    logger.info(f"[Indexer] start scanning: {vault_path}")
    indexed_mtimes = _get_indexed_mtimes()

    pending: list = []
    seen_ids: set[str] = set()
    added = 0
    skipped = 0

    for chunk in scan_vault(vault_path):
        if chunk.doc_id in seen_ids:
            logger.debug(f"[Indexer] duplicate doc_id skipped: {chunk.doc_id} ({chunk.file_name}#{chunk.section})")
            skipped += 1
            continue
        seen_ids.add(chunk.doc_id)

        cached_mtime = indexed_mtimes.get(chunk.doc_id, 0.0)
        if chunk.mtime <= cached_mtime:
            skipped += 1
            continue

        pending.append(chunk)
        if len(pending) >= _BATCH_SIZE:
            added += get_store().upsert_chunks(pending)
            pending.clear()

    if pending:
        added += get_store().upsert_chunks(pending)

    pruned = 0
    if prune_missing:
        pruned = _prune_missing(vault_path, seen_ids)

    total = added + skipped
    logger.info(f"[Indexer] done: added={added}, skipped={skipped}, pruned={pruned}, total={total}")
    if on_done:
        on_done(added, skipped)
    return added, skipped


def _prune_missing(vault_path: str, current_ids: set[str]) -> int:
    """Delete Chroma chunks whose source file under vault_path no longer exists."""
    store = get_store()
    try:
        store._ensure_collection()
        result = store._collection.get(include=["metadatas"])
    except Exception as exc:
        logger.warning("[Indexer] prune skipped: %s", exc)
        return 0

    vault_root = os.path.abspath(vault_path)
    delete_ids: list[str] = []
    for doc_id, meta in zip(result.get("ids") or [], result.get("metadatas") or []):
        file_path = os.path.abspath(str((meta or {}).get("file_path") or ""))
        if not file_path.startswith(vault_root + os.sep):
            continue
        if doc_id not in current_ids:
            delete_ids.append(doc_id)

    if not delete_ids:
        return 0
    store._collection.delete(ids=delete_ids)
    logger.info("[Indexer] pruned missing chunks: %d", len(delete_ids))
    return len(delete_ids)


def start_background_indexing(vault_path: str = _VAULT_PATH) -> threading.Thread:
    """
    バックグラウンドスレッドでインデックス化を開始する。
    FastAPI startup イベントから呼ぶ。
    """
    def _task():
        try:
            run_indexing(vault_path)
        except Exception as e:
            logger.warning(f"[Indexer] background indexing failed: {e}")

    t = threading.Thread(target=_task, name="obsidian-indexer", daemon=True)
    t.start()
    return t
