#!/usr/bin/env python3
"""Attach PostgreSQL sequences to integer primary-key columns after SQLite import."""

from __future__ import annotations

import os


def main() -> None:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    import psycopg2

    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.table_name, c.column_name
        FROM information_schema.columns c
        JOIN information_schema.table_constraints tc
          ON tc.table_schema = c.table_schema
         AND tc.table_name = c.table_name
         AND tc.constraint_type = 'PRIMARY KEY'
        JOIN information_schema.key_column_usage kcu
          ON kcu.table_schema = tc.table_schema
         AND kcu.table_name = tc.table_name
         AND kcu.constraint_name = tc.constraint_name
         AND kcu.column_name = c.column_name
        WHERE c.table_schema = current_schema()
          AND c.column_default IS NULL
          AND c.data_type IN ('bigint', 'integer')
          AND (
            SELECT COUNT(*)
            FROM information_schema.key_column_usage kcu2
            WHERE kcu2.table_schema = tc.table_schema
              AND kcu2.table_name = tc.table_name
              AND kcu2.constraint_name = tc.constraint_name
          ) = 1
        ORDER BY c.table_name, c.column_name
        """
    )
    targets = cur.fetchall()
    for table, column in targets:
        seq = f"{table}_{column}_seq"
        cur.execute(f'CREATE SEQUENCE IF NOT EXISTS "{seq}"')
        cur.execute(f'SELECT COALESCE(MAX("{column}"), 0) FROM "{table}"')
        max_id = int(cur.fetchone()[0] or 0)
        if max_id > 0:
            cur.execute("SELECT setval(%s, %s, true)", (seq, max_id))
        else:
            cur.execute("SELECT setval(%s, 1, false)", (seq,))
        cur.execute(f'ALTER SEQUENCE "{seq}" OWNED BY "{table}"."{column}"')
        cur.execute(
            f'ALTER TABLE "{table}" ALTER COLUMN "{column}" SET DEFAULT nextval(%s)',
            (seq,),
        )
        print(f"{table}.{column} -> {seq} next>{max_id}")
    cur.execute('ALTER TABLE "funding_rates" ALTER COLUMN "term_years" DROP DEFAULT')
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
