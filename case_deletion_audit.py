"""Transactional audit trail for destructive ``past_cases`` operations."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Mapping

from screening_record_lifecycle import mark_screening_parent_deleted


@dataclass(frozen=True)
class DeletionAuditEvent:
    event_id: str
    requested_case_ids: tuple[str, ...]
    matched_case_ids: tuple[str, ...]
    affected_screening_count: int


def _normalize_case_ids(case_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(case_id).strip() for case_id in case_ids if str(case_id).strip()}))


def ensure_case_deletion_audit_tables(conn: Any, *, is_postgres: bool = False) -> None:
    """Create the append-only deletion audit tables idempotently."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS case_deletion_events (
            event_id TEXT PRIMARY KEY,
            occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            route TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            requested_count INTEGER NOT NULL,
            matched_count INTEGER NOT NULL,
            deleted_count INTEGER NOT NULL DEFAULT 0,
            affected_screening_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'started',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS case_deletion_event_items (
            event_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            parent_table TEXT NOT NULL DEFAULT 'past_cases',
            item_status TEXT NOT NULL DEFAULT 'matched',
            PRIMARY KEY (event_id, case_id)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_case_deletion_events_occurred "
        "ON case_deletion_events(occurred_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_case_deletion_items_case "
        "ON case_deletion_event_items(case_id)"
    )


def begin_case_deletion_event(
    conn: Any,
    case_ids: Iterable[str],
    *,
    route: str,
    reason: str,
    placeholder: str = "?",
    is_postgres: bool = False,
    event_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DeletionAuditEvent:
    """Record targets and mark child records within the caller's transaction."""
    requested = _normalize_case_ids(case_ids)
    audit_event_id = event_id or f"del_{uuid.uuid4().hex}"
    ensure_case_deletion_audit_tables(conn, is_postgres=is_postgres)

    cursor = conn.cursor()
    matched: list[str] = []
    for case_id in requested:
        cursor.execute(f"SELECT 1 FROM past_cases WHERE id={placeholder}", (case_id,))
        if cursor.fetchone() is not None:
            matched.append(case_id)

    cursor.execute(
        f"""
        INSERT INTO case_deletion_events (
            event_id, route, reason, requested_count, matched_count,
            deleted_count, affected_screening_count, status, metadata_json
        ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, 0, 0, 'started', {placeholder})
        """,
        (
            audit_event_id,
            str(route),
            str(reason),
            len(requested),
            len(matched),
            json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True),
        ),
    )
    matched_set = set(matched)
    for case_id in requested:
        cursor.execute(
            f"""
            INSERT INTO case_deletion_event_items (event_id, case_id, parent_table, item_status)
            VALUES ({placeholder}, {placeholder}, 'past_cases', {placeholder})
            """,
            (audit_event_id, case_id, "matched" if case_id in matched_set else "not_found"),
        )

    affected_count, _ = mark_screening_parent_deleted(
        conn,
        matched,
        placeholder=placeholder,
        is_postgres=is_postgres,
        deletion_event_id=audit_event_id,
    )
    cursor.execute(
        f"UPDATE case_deletion_events SET affected_screening_count={placeholder} WHERE event_id={placeholder}",
        (affected_count, audit_event_id),
    )
    return DeletionAuditEvent(
        event_id=audit_event_id,
        requested_case_ids=requested,
        matched_case_ids=tuple(matched),
        affected_screening_count=affected_count,
    )


def complete_case_deletion_event(
    conn: Any,
    event: DeletionAuditEvent,
    deleted_case_ids: Iterable[str],
    *,
    placeholder: str = "?",
) -> str:
    """Finalize an audit event after the parent DELETE has succeeded."""
    deleted = _normalize_case_ids(deleted_case_ids)
    matched_set = set(event.matched_case_ids)
    deleted = tuple(case_id for case_id in deleted if case_id in matched_set)
    if not event.matched_case_ids:
        status = "no_match"
    elif set(deleted) == matched_set:
        status = "completed"
    else:
        status = "partial"

    cursor = conn.cursor()
    for case_id in deleted:
        cursor.execute(
            f"""
            UPDATE case_deletion_event_items
               SET item_status='deleted'
             WHERE event_id={placeholder} AND case_id={placeholder}
            """,
            (event.event_id, case_id),
        )
    cursor.execute(
        f"""
        UPDATE case_deletion_events
           SET deleted_count={placeholder}, status={placeholder}
         WHERE event_id={placeholder}
        """,
        (len(deleted), status, event.event_id),
    )
    return status


def list_case_deletion_events(
    conn: Any,
    *,
    limit: int = 50,
    offset: int = 0,
    placeholder: str = "?",
    status: str = "",
    date_from: str = "",
    date_to: str = "",
) -> dict[str, Any]:
    """Return a read-only, newest-first view of deletion audit events."""
    safe_limit = max(1, min(int(limit), 200))
    safe_offset = max(0, int(offset))
    allowed_statuses = {"started", "completed", "partial", "no_match"}
    safe_status = str(status or "").strip()
    if safe_status and safe_status not in allowed_statuses:
        raise ValueError("invalid deletion audit status")

    def _validated_date(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        return date.fromisoformat(raw).isoformat()

    safe_date_from = _validated_date(date_from)
    safe_date_to = _validated_date(date_to)
    if safe_date_from and safe_date_to and safe_date_from > safe_date_to:
        raise ValueError("date_from must not be after date_to")

    predicates: list[str] = []
    params: list[Any] = []
    if safe_status:
        predicates.append(f"status={placeholder}")
        params.append(safe_status)
    if safe_date_from:
        predicates.append(f"occurred_at >= {placeholder}")
        params.append(f"{safe_date_from} 00:00:00")
    if safe_date_to:
        exclusive_end = date.fromisoformat(safe_date_to) + timedelta(days=1)
        predicates.append(f"occurred_at < {placeholder}")
        params.append(f"{exclusive_end.isoformat()} 00:00:00")
    where_sql = f" WHERE {' AND '.join(predicates)}" if predicates else ""

    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM case_deletion_events{where_sql}", tuple(params))
    total = int(cursor.fetchone()[0])
    cursor.execute(
        f"""
        SELECT event_id, occurred_at, route, reason, requested_count, matched_count,
               deleted_count, affected_screening_count, status, metadata_json
          FROM case_deletion_events{where_sql}
         ORDER BY occurred_at DESC, event_id DESC
         LIMIT {placeholder} OFFSET {placeholder}
        """,
        tuple([*params, safe_limit, safe_offset]),
    )
    events = []
    for row in cursor.fetchall():
        event_id = str(row[0])
        item_cursor = conn.cursor()
        item_cursor.execute(
            f"""
            SELECT case_id, parent_table, item_status
              FROM case_deletion_event_items
             WHERE event_id={placeholder}
             ORDER BY case_id
            """,
            (event_id,),
        )
        items = [
            {"case_id": str(item[0]), "parent_table": str(item[1]), "status": str(item[2])}
            for item in item_cursor.fetchall()
        ]
        try:
            metadata = json.loads(row[9] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        occurred_at = row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1])
        events.append(
            {
                "event_id": event_id,
                "occurred_at": occurred_at,
                "route": str(row[2]),
                "reason": str(row[3]),
                "requested_count": int(row[4]),
                "matched_count": int(row[5]),
                "deleted_count": int(row[6]),
                "affected_screening_count": int(row[7]),
                "status": str(row[8]),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "items": items,
            }
        )
    return {
        "total": total,
        "limit": safe_limit,
        "offset": safe_offset,
        "filters": {"status": safe_status, "date_from": safe_date_from, "date_to": safe_date_to},
        "events": events,
    }
