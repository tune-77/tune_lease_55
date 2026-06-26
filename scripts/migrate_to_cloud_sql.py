#!/usr/bin/env python3
"""
scripts/migrate_to_cloud_sql.py  —  REV-161
SQLite (lease_data.db) → Cloud SQL (PostgreSQL) 初回マイグレーション

使い方:
    export DATABASE_URL="postgresql://user:pass@/dbname?host=/cloudsql/proj:region:inst"
    python scripts/migrate_to_cloud_sql.py

    # SQLite パスを明示する場合:
    SQLITE_DB_PATH=/path/to/lease_data.db python scripts/migrate_to_cloud_sql.py

冪等性:
    - PK がある列: INSERT ... ON CONFLICT (pk_cols) DO NOTHING
    - PK がない列: INSERT のみ（重複は許容）
    既存行はスキップされるため何度実行しても安全。

注意:
    lease_data.db は .gitignore で除外済み。このスクリプト自体もファイルに触れない。
"""

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SQLITE_DB_PATH = Path(os.environ.get("SQLITE_DB_PATH", REPO_ROOT / "data" / "lease_data.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")
SKIP_TABLES = {"sqlite_sequence"}
# ────────────────────────────────────────────────────────────────────────────────


def _check_prereqs() -> None:
    if not DATABASE_URL:
        print("❌  DATABASE_URL が設定されていません。")
        print('    export DATABASE_URL="postgresql://user:pass@host:5432/dbname"')
        sys.exit(1)
    if not SQLITE_DB_PATH.exists():
        print(f"❌  SQLite DB が見つかりません: {SQLITE_DB_PATH}")
        sys.exit(1)
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        print("❌  psycopg2-binary が未インストールです。")
        print("    pip install psycopg2-binary --break-system-packages")
        sys.exit(1)


def _get_tables(sqlite_conn: sqlite3.Connection) -> list[str]:
    cur = sqlite_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [r[0] for r in cur.fetchall() if r[0] not in SKIP_TABLES]


def _get_schema(sqlite_conn: sqlite3.Connection, table: str) -> list[dict]:
    cur = sqlite_conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    return [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in cur.fetchall()]


def _sqlite_type_to_pg(sqlite_type: str) -> str:
    t = sqlite_type.upper().strip()
    if t in ("INTEGER", "INT"):
        return "BIGINT"
    if t in ("REAL", "FLOAT", "DOUBLE"):
        return "DOUBLE PRECISION"
    if t in ("BOOLEAN", "BOOL"):
        return "BOOLEAN"
    if t in ("BLOB",):
        return "BYTEA"
    return "TEXT"


def _ensure_table(pg_cur, table: str, schema: list[dict]) -> None:
    from psycopg2 import sql

    col_defs = []
    for col in schema:
        pg_type = _sqlite_type_to_pg(col["type"])
        pk = " PRIMARY KEY" if col["pk"] == 1 else ""
        not_null = " NOT NULL" if col["notnull"] else ""
        col_defs.append(
            sql.SQL("{}{}{}{}").format(
                sql.Identifier(col["name"]),
                sql.SQL(f" {pg_type}"),
                sql.SQL(pk),
                sql.SQL(not_null),
            )
        )
    ddl = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
        sql.Identifier(table),
        sql.SQL(", ").join(col_defs),
    )
    pg_cur.execute(ddl)


def _migrate_table(pg_conn, pg_cur, sqlite_conn: sqlite3.Connection, table: str) -> int:
    from psycopg2 import sql
    import psycopg2.extras

    schema = _get_schema(sqlite_conn, table)
    _ensure_table(pg_cur, table, schema)

    sqlite_cur = sqlite_conn.cursor()
    sqlite_cur.execute(f'SELECT * FROM "{table}"')
    rows = sqlite_cur.fetchall()
    if not rows:
        return 0

    pk_cols = [c["name"] for c in schema if c["pk"] > 0]
    col_idents = [sql.Identifier(c["name"]) for c in schema]
    placeholders = [sql.Placeholder() for _ in schema]

    if pk_cols:
        conflict_clause = sql.SQL("ON CONFLICT ({}) DO NOTHING").format(
            sql.SQL(", ").join(sql.Identifier(c) for c in pk_cols)
        )
    else:
        conflict_clause = sql.SQL("")

    insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES ({}) {}").format(
        sql.Identifier(table),
        sql.SQL(", ").join(col_idents),
        sql.SQL(", ").join(placeholders),
        conflict_clause,
    )

    def coerce(row):
        result = []
        for val, col in zip(row, schema):
            if val is None:
                result.append(None)
            elif col["type"].upper() in ("BOOLEAN", "BOOL"):
                result.append(bool(val))
            else:
                result.append(val)
        return result

    batch = [coerce(r) for r in rows]
    psycopg2.extras.execute_batch(pg_cur, insert_sql.as_string(pg_conn), batch, page_size=200)
    return len(batch)


def main() -> None:
    _check_prereqs()

    import psycopg2
    import psycopg2.extras

    started_at = datetime.now()
    print("=" * 60)
    print("  SQLite → Cloud SQL マイグレーション (REV-161)")
    print("=" * 60)
    print(f"  Source : {SQLITE_DB_PATH}")
    print(f"  Target : {DATABASE_URL.split('@')[0]}@***")
    print(f"  開始   : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    sqlite_conn = sqlite3.connect(str(SQLITE_DB_PATH), timeout=15)
    pg_conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    pg_conn.autocommit = False
    pg_cur = pg_conn.cursor()

    tables = _get_tables(sqlite_conn)
    print(f"移行対象テーブル: {len(tables)} 件\n")

    total = 0
    try:
        for table in tables:
            count = _migrate_table(pg_conn, pg_cur, sqlite_conn, table)
            status = f"{count} 行" if count else "0 行（スキップ or 空）"
            print(f"  ✅  {table}: {status}")
            total += count
        pg_conn.commit()
    except Exception as e:
        pg_conn.rollback()
        print(f"\n  ❌  エラー発生、ロールバックしました: {e}")
        raise
    finally:
        sqlite_conn.close()
        pg_cur.close()
        pg_conn.close()

    elapsed = (datetime.now() - started_at).total_seconds()
    print(f"\n[完了] {elapsed:.1f} 秒 / 合計 {total} 行を INSERT")
    print("✅  全テーブルの移行が完了しました。")


if __name__ == "__main__":
    main()
