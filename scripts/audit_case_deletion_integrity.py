#!/usr/bin/env python3
"""Read-only consistency audit for case-deletion events and linked records."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import get_db_path

REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_JSON = REPORTS_DIR / "case_deletion_integrity_latest.json"
DEFAULT_MD = REPORTS_DIR / "case_deletion_integrity_latest.md"
ALLOWED_EVENT_STATUSES = {"started", "completed", "partial", "no_match"}
ALLOWED_ITEM_STATUSES = {"matched", "deleted", "not_found"}


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone() is not None


def _parse_sqlite_timestamp(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def audit_deletion_integrity(
    db_path: Path,
    *,
    stale_started_minutes: int = 10,
    now: datetime | None = None,
) -> dict[str, Any]:
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issues: list[dict[str, Any]] = []
    required_tables = {
        "past_cases",
        "screening_records",
        "case_deletion_events",
        "case_deletion_event_items",
    }
    if not db_path.exists():
        return {
            "status": "FAIL",
            "checked_at": checked_at.isoformat(),
            "db_path": str(db_path),
            "summary": {"events": 0, "issues": 1},
            "issues": [{"code": "database_missing", "severity": "error", "detail": str(db_path)}],
        }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        missing_tables = sorted(table for table in required_tables if not _table_exists(conn, table))
        if missing_tables:
            return {
                "status": "FAIL",
                "checked_at": checked_at.isoformat(),
                "db_path": str(db_path),
                "summary": {"events": 0, "issues": 1},
                "database_integrity": integrity,
                "issues": [
                    {
                        "code": "audit_tables_missing",
                        "severity": "error",
                        "detail": ", ".join(missing_tables),
                    }
                ],
            }

        events = conn.execute(
            """
            SELECT event_id, occurred_at, requested_count, matched_count, deleted_count,
                   affected_screening_count, status
              FROM case_deletion_events
             ORDER BY occurred_at, event_id
            """
        ).fetchall()
        status_counts: dict[str, int] = {}
        for event in events:
            event_id = str(event["event_id"])
            event_status = str(event["status"])
            status_counts[event_status] = status_counts.get(event_status, 0) + 1
            items = conn.execute(
                "SELECT case_id, item_status FROM case_deletion_event_items WHERE event_id=?",
                (event_id,),
            ).fetchall()
            item_statuses = [str(item["item_status"]) for item in items]
            matched_items = sum(status in {"matched", "deleted"} for status in item_statuses)
            deleted_items = sum(status == "deleted" for status in item_statuses)
            linked_screening = int(
                conn.execute(
                    "SELECT COUNT(*) FROM screening_records WHERE deletion_event_id=?",
                    (event_id,),
                ).fetchone()[0]
            )
            invalid_linked_screening = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM screening_records
                     WHERE deletion_event_id=? AND COALESCE(record_state, 'active') <> 'parent_deleted'
                    """,
                    (event_id,),
                ).fetchone()[0]
            )
            remaining_parents = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                      FROM case_deletion_event_items i
                      JOIN past_cases p ON p.id=i.case_id
                     WHERE i.event_id=? AND i.item_status='deleted'
                    """,
                    (event_id,),
                ).fetchone()[0]
            )

            comparisons = (
                ("requested_count_mismatch", int(event["requested_count"]), len(items)),
                ("matched_count_mismatch", int(event["matched_count"]), matched_items),
                ("deleted_count_mismatch", int(event["deleted_count"]), deleted_items),
                ("screening_count_mismatch", int(event["affected_screening_count"]), linked_screening),
            )
            for code, recorded, observed in comparisons:
                if recorded != observed:
                    issues.append(
                        {
                            "code": code,
                            "severity": "error",
                            "event_id": event_id,
                            "recorded": recorded,
                            "observed": observed,
                        }
                    )
            invalid_item_statuses = sorted(set(item_statuses) - ALLOWED_ITEM_STATUSES)
            if invalid_item_statuses:
                issues.append(
                    {
                        "code": "invalid_item_status",
                        "severity": "error",
                        "event_id": event_id,
                        "detail": ", ".join(invalid_item_statuses),
                    }
                )
            if event_status not in ALLOWED_EVENT_STATUSES:
                issues.append(
                    {
                        "code": "invalid_event_status",
                        "severity": "error",
                        "event_id": event_id,
                        "detail": event_status,
                    }
                )
            if event_status == "completed" and not (
                int(event["matched_count"]) > 0
                and int(event["deleted_count"]) == int(event["matched_count"])
            ):
                issues.append({"code": "invalid_completed_event", "severity": "error", "event_id": event_id})
            if event_status == "no_match" and any(
                int(event[name]) != 0
                for name in ("matched_count", "deleted_count", "affected_screening_count")
            ):
                issues.append({"code": "invalid_no_match_event", "severity": "error", "event_id": event_id})
            if event_status == "partial":
                issues.append({"code": "partial_deletion", "severity": "warning", "event_id": event_id})
            if event_status == "started":
                occurred_at = _parse_sqlite_timestamp(str(event["occurred_at"]))
                age_minutes = (
                    (checked_at - occurred_at).total_seconds() / 60 if occurred_at is not None else None
                )
                if age_minutes is None or age_minutes > stale_started_minutes:
                    issues.append(
                        {
                            "code": "stale_started_event",
                            "severity": "error",
                            "event_id": event_id,
                            "age_minutes": round(age_minutes, 1) if age_minutes is not None else None,
                        }
                    )
            if invalid_linked_screening:
                issues.append(
                    {
                        "code": "linked_screening_not_parent_deleted",
                        "severity": "error",
                        "event_id": event_id,
                        "observed": invalid_linked_screening,
                    }
                )
            if remaining_parents:
                issues.append(
                    {
                        "code": "deleted_parent_reappeared",
                        "severity": "warning",
                        "event_id": event_id,
                        "observed": remaining_parents,
                    }
                )

        orphan_event_links = int(
            conn.execute(
                """
                SELECT COUNT(*)
                  FROM screening_records s
             LEFT JOIN case_deletion_events e ON e.event_id=s.deletion_event_id
                 WHERE s.deletion_event_id IS NOT NULL AND e.event_id IS NULL
                """
            ).fetchone()[0]
        )
        if orphan_event_links:
            issues.append(
                {
                    "code": "screening_event_without_audit",
                    "severity": "error",
                    "observed": orphan_event_links,
                }
            )
        if integrity != "ok":
            issues.append({"code": "sqlite_integrity_failed", "severity": "error", "detail": integrity})

    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    status = "FAIL" if error_count else "WARN" if warning_count else "PASS"
    return {
        "status": status,
        "checked_at": checked_at.isoformat(),
        "db_path": str(db_path),
        "database_integrity": integrity,
        "summary": {
            "events": len(events),
            "event_statuses": status_counts,
            "errors": error_count,
            "warnings": warning_count,
            "issues": len(issues),
            "orphan_screening_event_links": orphan_event_links,
        },
        "issues": issues,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# 案件削除監査 整合性レポート",
        "",
        f"- 判定: **{payload.get('status', 'FAIL')}**",
        f"- 検査時刻: `{payload.get('checked_at', '')}`",
        f"- DB: `{payload.get('db_path', '')}`",
        f"- SQLite integrity: `{payload.get('database_integrity', 'unknown')}`",
        f"- 削除イベント: {summary.get('events', 0)}件",
        f"- エラー: {summary.get('errors', 0)}件 / 警告: {summary.get('warnings', 0)}件",
        f"- 状態別: `{json.dumps(summary.get('event_statuses', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 検査項目",
        "",
        "- イベントの要求・一致・削除件数と対象明細の一致",
        "- `affected_screening_count` と同一 `deletion_event_id` の審査記録数の一致",
        "- 関連審査記録が `parent_deleted` であること",
        "- 削除済み案件IDが `past_cases` に残存・再出現していないこと",
        "- 監査イベントのない `screening_records.deletion_event_id` がないこと",
        "- 長時間 `started`、`partial`、不正な状態値がないこと",
        "",
        "## 検出事項",
        "",
    ]
    issues = payload.get("issues") or []
    if not issues:
        lines.append("- なし")
    else:
        for issue in issues:
            event = f" / event `{issue['event_id']}`" if issue.get("event_id") else ""
            detail = json.dumps(
                {key: value for key, value in issue.items() if key not in {"code", "severity", "event_id"}},
                ensure_ascii=False,
                sort_keys=True,
            )
            lines.append(f"- **{issue.get('severity')}** `{issue.get('code')}`{event}: `{detail}`")
    lines.extend(["", "_読み取り専用監査です。自動修復・削除・復元は行いません。_", ""])
    return "\n".join(lines)


def write_reports(payload: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    stamp = datetime.fromisoformat(str(payload["checked_at"])).astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
    (output_json.parent / f"case_deletion_integrity_{stamp}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_md.parent / f"case_deletion_integrity_{stamp}.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(get_db_path()))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--stale-started-minutes", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="return 1 when status is WARN or FAIL")
    args = parser.parse_args()

    payload = audit_deletion_integrity(
        args.db,
        stale_started_minutes=max(1, args.stale_started_minutes),
    )
    write_reports(payload, args.output_json, args.output_md)
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    return 1 if args.strict and payload["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
