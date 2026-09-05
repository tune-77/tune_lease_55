#!/usr/bin/env python3
"""Backfill missing screening_records rows from past_cases."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screening_recorder import record_screening_result


DEFAULT_DB_PATH = "data/lease_data.db"
DEFAULT_SINCE = "2026-06-05"


def _parse_since(value: str) -> str:
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--since must be YYYY-MM-DD") from exc


def _score_value(value: Any) -> tuple[float | None, str | None]:
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, bool):
        return None, "not_numeric"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None, "not_numeric"
    if not math.isfinite(score):
        return None, "not_numeric"
    if not 0.0 <= score <= 100.0:
        return None, "out_of_range"
    return score, None


def _optional_number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _prefer_result_value(data: dict[str, Any], key: str) -> Any:
    result = data.get("result")
    if isinstance(result, dict) and result.get(key) is not None:
        return result.get(key)
    return data.get(key)


def _load_rows(db_path: str, since: str) -> tuple[list[sqlite3.Row], set[str]]:
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, timestamp, registration_date, data
              FROM past_cases
             WHERE substr(timestamp, 1, 10) >= ?
                OR substr(registration_date, 1, 10) >= ?
             ORDER BY timestamp, id
            """,
            (since, since),
        ).fetchall()
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='screening_records'"
        ).fetchone()
        existing_case_ids = (
            {
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT case_id FROM screening_records WHERE case_id IS NOT NULL"
                )
            }
            if table_exists
            else set()
        )
    return rows, existing_case_ids


def backfill(db_path: str, since: str, apply: bool) -> int:
    rows, existing_case_ids = _load_rows(db_path, since)
    skipped_existing = 0
    skipped_invalid = 0
    skipped_reasons: Counter[str] = Counter()
    insertable = 0
    inserted = 0
    errors = 0

    for row in rows:
        case_id = row["id"]
        case_id_text = str(case_id) if case_id is not None else ""
        if case_id_text in existing_case_ids:
            skipped_existing += 1
            continue
        if not case_id_text:
            skipped_invalid += 1
            skipped_reasons["case_id_missing"] += 1
            continue
        if not row["timestamp"]:
            skipped_invalid += 1
            skipped_reasons["timestamp_missing"] += 1
            continue

        try:
            data = json.loads(row["data"])
        except (TypeError, json.JSONDecodeError):
            skipped_invalid += 1
            skipped_reasons["data_invalid_json"] += 1
            continue
        if not isinstance(data, dict):
            skipped_invalid += 1
            skipped_reasons["data_not_object"] += 1
            continue

        total_score, total_error = _score_value(
            _prefer_result_value(data, "score")
        )
        asset_score, asset_error = _score_value(
            _prefer_result_value(data, "asset_score")
        )
        if total_error:
            skipped_reasons[f"total_score_{total_error}"] += 1
        if asset_error:
            skipped_reasons[f"asset_score_{asset_error}"] += 1
        if total_error or asset_error:
            skipped_invalid += 1
            continue

        insertable += 1
        if not apply:
            continue

        result = record_screening_result(
            case_id=case_id_text,
            screened_at=row["timestamp"],
            total_score=total_score,
            asset_score=asset_score,
            tenant_score=_optional_number(
                _prefer_result_value(data, "score_borrower")
            ),
            q_risk_score=_optional_number(
                _prefer_result_value(data, "quantum_risk")
            ),
            competitor_pressure_score=None,
            outcome=None,
            input_snapshot=data.get("inputs"),
            source="api_backfill",
            db_path=db_path,
        )
        if result.get("success"):
            inserted += 1
            existing_case_ids.add(case_id_text)
        else:
            errors += 1
            print(
                f"ERROR case_id={case_id_text}: {result.get('error')}",
                file=sys.stderr,
            )

    print("screening_records backfill summary")
    print(f"  mode: {'apply' if apply else 'dry-run'}")
    print(f"  since: {since}")
    print(f"  scanned: {len(rows)}")
    if apply:
        print(f"  inserted: {inserted}")
    else:
        print(f"  would_insert: {insertable}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  skipped_required_or_invalid: {skipped_invalid}")
    for reason, count in sorted(skipped_reasons.items()):
        print(f"    {reason}: {count}")
    print(f"  errors: {errors}")
    return 0 if errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing screening_records from past_cases."
    )
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write records (default: dry-run)",
    )
    parser.add_argument("--since", type=_parse_since, default=DEFAULT_SINCE)
    args = parser.parse_args()

    try:
        return backfill(args.db_path, args.since, args.apply)
    except (FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
