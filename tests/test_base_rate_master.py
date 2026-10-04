from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import base_rate_master


def test_get_conn_serializes_concurrent_term_column_migration(tmp_path, monkeypatch):
    """同じ不足列を見た接続同士が二重に ALTER TABLE しない。"""
    db_path = tmp_path / "base-rate-race.db"
    first_schema_reads = threading.Barrier(2)
    real_connect = sqlite3.connect

    class SynchronizedConnection:
        def __init__(self, connection):
            self._connection = connection
            self._first_schema_read = True

        def execute(self, sql, parameters=()):
            cursor = self._connection.execute(sql, parameters)
            if self._first_schema_read and sql.lstrip().upper().startswith("PRAGMA TABLE_INFO"):
                # 両接続に migration 前の同じスキーマを確実に見せる。
                rows = cursor.fetchall()
                self._first_schema_read = False
                first_schema_reads.wait(timeout=2)
                return rows
            return cursor

        def __getattr__(self, name):
            return getattr(self._connection, name)

    def synchronized_connect(*args, **kwargs):
        return SynchronizedConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(base_rate_master, "_DB_PATH", db_path)
    monkeypatch.setattr(base_rate_master.sqlite3, "connect", synchronized_connect)

    def open_and_close_connection(_worker):
        base_rate_master._get_conn().close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(open_and_close_connection, range(2)))

    with real_connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(base_rate_master)")}

    assert set(base_rate_master.TERM_COLS) <= columns
