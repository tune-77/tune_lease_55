#!/usr/bin/env python3
"""
Obsidian Vault を ChromaDB (obsidian_knowledge) に再インデックスするスクリプト。

使い方:
    # 差分更新（デフォルト）: mtime が変化したファイルのみ upsert
    .venv/bin/python scripts/reindex_obsidian.py

    # 完全再構築: コレクションを削除して全件インデックス
    .venv/bin/python scripts/reindex_obsidian.py --full

    # Vault パスを上書き
    OBSIDIAN_VAULT_PATH=/path/to/vault .venv/bin/python scripts/reindex_obsidian.py

毎晩 3:00 に launchd から自動実行される（--full モード推奨）。
ログ: ~/Library/Logs/tune_lease_55_obsidian_reindex.out.log
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# リポジトリルートを PYTHONPATH に追加（このスクリプトは scripts/ に置かれる）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("reindex_obsidian")

# デフォルト Vault: env で上書き可
_DEFAULT_VAULT = "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", _DEFAULT_VAULT)


def full_reindex(vault_path: str) -> tuple[int, int]:
    """コレクションを削除して全件インデックスし直す。"""
    import chromadb
    from api.knowledge.vector_store import KnowledgeVectorStore, _CHROMA_DIR, _COLLECTION_NAME

    logger.info("[reindex] --full モード: コレクション %s を再構築します", _COLLECTION_NAME)

    # 既存コレクション削除
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    try:
        client.delete_collection(_COLLECTION_NAME)
        logger.info("[reindex] 既存コレクション削除完了")
    except Exception:
        logger.info("[reindex] 削除対象コレクションなし（初回実行）")

    # 新しいストアで全件インデックス
    store = KnowledgeVectorStore()
    from api.knowledge.obsidian_loader import scan_vault

    logger.info("[reindex] Vault スキャン開始: %s", vault_path)
    pending, added, skipped = [], 0, 0
    seen_ids: set[str] = set()

    for chunk in scan_vault(vault_path):
        if chunk.doc_id in seen_ids:
            skipped += 1
            continue
        seen_ids.add(chunk.doc_id)
        pending.append(chunk)

        if len(pending) >= 50:
            added += store.upsert_chunks(pending)
            logger.info("[reindex]  ... %d 件登録済み", added)
            pending.clear()

    if pending:
        added += store.upsert_chunks(pending)

    return added, skipped


def diff_reindex(vault_path: str, prune_missing: bool = False) -> tuple[int, int]:
    """差分更新: mtime が変化したチャンクのみ upsert。"""
    logger.info("[reindex] 差分更新モード")
    from api.knowledge.indexer import run_indexing
    return run_indexing(vault_path, prune_missing=prune_missing)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obsidian Vault を ChromaDB に再インデックス"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="コレクションを削除して全件再構築（デフォルト: 差分更新）",
    )
    parser.add_argument(
        "--vault",
        default=VAULT_PATH,
        help=f"Obsidian Vault のパス (default: {VAULT_PATH})",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="指定Vault配下で消えたファイルの古いチャンクをChromaDBから削除",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("[reindex] 開始  vault=%s  full=%s", args.vault, args.full)
    start = time.time()

    if args.full:
        added, skipped = full_reindex(args.vault)
    else:
        added, skipped = diff_reindex(args.vault, prune_missing=args.prune_missing)

    elapsed = time.time() - start

    # 最終件数確認
    try:
        from api.knowledge.vector_store import get_store
        total = get_store().count()
        logger.info(
            "[reindex] 完了  added=%d  skipped=%d  total_in_db=%d  elapsed=%.1fs",
            added, skipped, total, elapsed,
        )
    except Exception:
        logger.info(
            "[reindex] 完了  added=%d  skipped=%d  elapsed=%.1fs",
            added, skipped, elapsed,
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
