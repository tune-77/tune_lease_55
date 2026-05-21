"""
Obsidian Vault のインデックス化スクリプト。
起動時にバックグラウンドスレッドで実行（メインスレッドをブロックしない）。
初回は全件、2回目以降はファイル更新日時で差分のみ更新。
"""
from __future__ import annotations

import logging
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
        store._ensure_initialized()
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
        cached_mtime = indexed_mtimes.get(chunk.doc_id, 0.0)
        if chunk.mtime <= cached_mtime:
            skipped += 1
            continue
        if chunk.doc_id in seen_ids:
            logger.debug(f"[Indexer] duplicate doc_id skipped: {chunk.doc_id} ({chunk.file_name}#{chunk.section})")
            skipped += 1
            continue
        seen_ids.add(chunk.doc_id)

        pending.append(chunk)
        if len(pending) >= _BATCH_SIZE:
            added += get_store().upsert_chunks(pending)
            pending.clear()

    if pending:
        added += get_store().upsert_chunks(pending)

    total = added + skipped
    logger.info(f"[Indexer] done: added={added}, skipped={skipped}, total={total}")
    if on_done:
        on_done(added, skipped)
    return added, skipped


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
