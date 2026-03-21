# -*- coding: utf-8 -*-
"""
tests/test_data_cases.py
========================
data_cases.py の基本動作テスト。
SQLite DB は一時ファイルにリダイレクトして実行する。
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

# プロジェクトルートを sys.path に追加
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """data_cases.DB_PATH を一時ファイルに差し替える。"""
    db = tmp_path / "test_cases.db"
    # テーブル作成
    with closing(sqlite3.connect(str(db))) as conn:
        conn.execute("""
            CREATE TABLE past_cases (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                data      TEXT    NOT NULL
            )
        """)
        conn.commit()

    import data_cases
    monkeypatch.setattr(data_cases, "DB_PATH", str(db))
    return str(db)


def test_load_all_cases_empty(tmp_db):
    import data_cases
    result = data_cases.load_all_cases()
    assert result == []


def test_save_and_load_case(tmp_db):
    import data_cases
    import sqlite3
    from contextlib import closing

    case = {"company_name": "テスト株式会社", "score": 75.0, "industry": "製造業"}

    # load_all_cases は past_cases.data (JSON) を読む。直接 INSERT してテスト。
    import json
    with closing(sqlite3.connect(tmp_db)) as conn:
        conn.execute(
            "INSERT INTO past_cases (timestamp, data) VALUES (?, ?)",
            ("2026-01-01T00:00:00", json.dumps(case))
        )
        conn.commit()

    loaded = data_cases.load_all_cases()
    assert len(loaded) == 1
    assert loaded[0]["company_name"] == "テスト株式会社"
    assert loaded[0]["score"] == 75.0


def test_load_all_cases_multiple(tmp_db):
    import data_cases
    import sqlite3
    import json
    from contextlib import closing

    cases = [
        {"company_name": f"会社{i}", "score": float(i * 10)}
        for i in range(5)
    ]
    with closing(sqlite3.connect(tmp_db)) as conn:
        for case in cases:
            conn.execute(
                "INSERT INTO past_cases (timestamp, data) VALUES (?, ?)",
                ("2026-01-01T00:00:00", json.dumps(case))
            )
        conn.commit()

    loaded = data_cases.load_all_cases()
    assert len(loaded) == 5


def test_load_all_cases_cached_exists():
    """load_all_cases_cached が callable として存在することを確認。"""
    import data_cases
    assert callable(data_cases.load_all_cases_cached)
