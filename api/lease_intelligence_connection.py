"""Knowledge-connection state helpers for the lease-intelligence dialogue."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from api.db_connection import current_backend, get_connection
from obsidian_query import count_vault_notes


def count_markdown_notes(root: Path, *, max_scan: int = 2000) -> int:
    """root配下の.md件数を返す（obsidian_query.count_vault_notesへ委譲）。"""
    return count_vault_notes(root, max_scan=max_scan)


def _table_exists(cur: Any, table_name: str) -> bool:
    if current_backend() == "postgresql":
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        row = cur.fetchone()
        return bool(row and row[0])
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return bool(cur.fetchone())


def build_lease_intelligence_knowledge_connection(
    vault: Path | None,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    vector_chunks = 0
    try:
        from api.knowledge.vector_store import get_store

        vector_chunks = int(get_store().count() or 0)
    except Exception:
        vector_chunks = 0

    case_count = 0
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            if _table_exists(cur, "past_cases"):
                cur.execute("SELECT COUNT(*) FROM past_cases")
                row = cur.fetchone()
                case_count = int(row[0] if row else 0)
    except Exception:
        case_count = 0

    markdown_roots: list[Path] = []
    for raw in (
        os.environ.get("OBSIDIAN_VAULT", ""),
        os.environ.get("OBSIDIAN_VAULT_PATH", ""),
        str(vault or ""),
        str(Path(repo_root) / ".cloudrun_bundle" / "obsidian_vault"),
    ):
        if raw:
            path = Path(raw)
            if path not in markdown_roots:
                markdown_roots.append(path)
    markdown_count = 0
    markdown_root = ""
    for root in markdown_roots:
        count = count_markdown_notes(root)
        if count > markdown_count:
            markdown_count = count
            markdown_root = str(root)

    is_cloud_run = bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUDRUN_DATA_MODE"))
    if vector_chunks > 0:
        label = f"知識DB検索可能: {vector_chunks}チャンク"
        source = "knowledge_vector_db"
    elif markdown_count > 0:
        label = f"知識コピー検索可能: {markdown_count}ノート"
        source = "markdown_vault_copy"
    elif case_count > 0:
        label = f"案件DB接続: {case_count}件"
        source = "case_database"
    else:
        label = "知識接続: 未確認"
        source = "unavailable"

    return {
        "label": label,
        "source": source,
        "is_cloud_run": is_cloud_run,
        "vector_chunks": vector_chunks,
        "markdown_notes": markdown_count,
        "case_count": case_count,
        "markdown_root": markdown_root,
    }


def lease_intelligence_indexed_notes_for_compat(connection: dict[str, Any]) -> int:
    """旧UI向けの indexed_notes にも接続済み件数を入れる。"""
    for key in ("markdown_notes", "vector_chunks", "case_count"):
        try:
            count = int(connection.get(key) or 0)
        except Exception:
            count = 0
        if count > 0:
            return count
    return 0
