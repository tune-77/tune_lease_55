from __future__ import annotations

import pytest

from scripts import migrate_to_cloud_sql as migrator


def _sqlite_schema() -> list[dict]:
    return [
        {"name": "id", "type": "INTEGER", "notnull": 1, "pk": 1},
        {"name": "name", "type": "TEXT", "notnull": 0, "pk": 0},
        {"name": "score", "type": "REAL", "notnull": 0, "pk": 0},
    ]


def test_schema_mismatches_accepts_matching_schema() -> None:
    existing = [
        {"name": "id", "type": "bigint", "notnull": True},
        {"name": "name", "type": "text", "notnull": False},
        {"name": "score", "type": "double precision", "notnull": False},
    ]

    assert migrator._schema_mismatches(existing, _sqlite_schema()) == []


def test_schema_mismatches_detects_column_order_difference() -> None:
    existing = [
        {"name": "name", "type": "text", "notnull": False},
        {"name": "id", "type": "bigint", "notnull": True},
        {"name": "score", "type": "double precision", "notnull": False},
    ]

    mismatches = migrator._schema_mismatches(existing, _sqlite_schema())

    assert mismatches
    assert "columns existing=" in mismatches[0]


def test_schema_mismatches_detects_type_difference() -> None:
    existing = [
        {"name": "id", "type": "text", "notnull": True},
        {"name": "name", "type": "text", "notnull": False},
        {"name": "score", "type": "double precision", "notnull": False},
    ]

    mismatches = migrator._schema_mismatches(existing, _sqlite_schema())

    assert "id.type existing=text expected=bigint" in mismatches


def test_validate_existing_table_schema_raises_on_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        migrator,
        "_get_existing_pg_schema",
        lambda _cur, _table: [{"name": "id", "type": "text", "notnull": True}],
    )

    with pytest.raises(RuntimeError, match="スキーマが SQLite と一致しません"):
        migrator._validate_existing_table_schema(object(), "past_cases", _sqlite_schema())
