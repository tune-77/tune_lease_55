"""Lifecycle guards shared by case and screening-record persistence paths."""
from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from typing import Any


CANONICAL_CASE_ID_RE = re.compile(r"^\d{20}_[0-9a-fA-F]{8}$")
LIFECYCLE_COLUMNS = {
    "record_state": "TEXT NOT NULL DEFAULT 'active'",
    "parent_deleted_at": "TEXT",
    "deletion_event_id": "TEXT",
}


def is_canonical_case_id(case_id: str) -> bool:
    return bool(CANONICAL_CASE_ID_RE.fullmatch(str(case_id or "").strip()))


def active_screening_predicate(conn: Any, alias: str = "") -> str:
    """Return an active-row SQL predicate, with legacy-schema compatibility."""
    prefix = f"{alias}." if alias else ""
    table_alias = f" {alias}" if alias else ""
    try:
        conn.cursor().execute(
            f"SELECT {prefix}record_state FROM screening_records{table_alias} WHERE 1=0"
        )
    except Exception:
        return "1=1"
    return f"COALESCE({prefix}record_state, 'active')='active'"


def ensure_screening_lifecycle_columns(conn: Any, *, is_postgres: bool = False) -> None:
    """Add lifecycle columns to an existing screening_records table idempotently."""
    cursor = conn.cursor()
    if is_postgres:
        for name, definition in LIFECYCLE_COLUMNS.items():
            cursor.execute(
                f'ALTER TABLE screening_records ADD COLUMN IF NOT EXISTS "{name}" {definition}'
            )
        return

    existing = {
        str(row[1])
        for row in cursor.execute("PRAGMA table_info(screening_records)").fetchall()
    }
    for name, definition in LIFECYCLE_COLUMNS.items():
        if name not in existing:
            cursor.execute(f'ALTER TABLE screening_records ADD COLUMN "{name}" {definition}')


def screening_parent_exists(conn: Any, case_id: str, *, placeholder: str = "?") -> bool:
    """Return whether a case exists in either supported parent table."""
    cursor = conn.cursor()
    query = f"""
        SELECT 1 FROM past_cases WHERE id={placeholder}
        UNION ALL
        SELECT 1 FROM excluded_grade_cases WHERE id={placeholder}
        LIMIT 1
    """
    try:
        return cursor.execute(query, (case_id, case_id)).fetchone() is not None
    except Exception:
        return False


def mark_screening_parent_deleted(
    conn: Any,
    case_ids: Iterable[str],
    *,
    placeholder: str = "?",
    is_postgres: bool = False,
    deletion_event_id: str | None = None,
) -> tuple[int, str]:
    """Mark child screening records before their parent case is removed."""
    normalized = sorted({str(case_id).strip() for case_id in case_ids if str(case_id).strip()})
    event_id = deletion_event_id or f"del_{uuid.uuid4().hex}"
    if not normalized:
        return 0, event_id

    ensure_screening_lifecycle_columns(conn, is_postgres=is_postgres)
    cursor = conn.cursor()
    changed = 0
    for case_id in normalized:
        cursor.execute(
            f"""
            UPDATE screening_records
               SET record_state='parent_deleted',
                   parent_deleted_at=COALESCE(parent_deleted_at, CURRENT_TIMESTAMP),
                   deletion_event_id=COALESCE(deletion_event_id, {placeholder}),
                   updated_at=CURRENT_TIMESTAMP
             WHERE case_id={placeholder}
               AND COALESCE(record_state, 'active') <> 'parent_deleted'
            """,
            (event_id, case_id),
        )
        if cursor.rowcount and cursor.rowcount > 0:
            changed += int(cursor.rowcount)
    return changed, event_id
