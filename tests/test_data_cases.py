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


def test_save_and_load_case(tmp_db, monkeypatch):
    import data_cases
    import sqlite3
    import json
    from contextlib import closing

    monkeypatch.setattr(data_cases, "DB_PATH", tmp_db)

    case = {"id": "test_id_1", "company_name": "テスト株式会社", "score": 85.5}

    # load_all_cases は past_cases.data (JSON) を読む。直接 INSERT してテスト。
    with closing(sqlite3.connect(tmp_db)) as conn:
        conn.execute(
            "INSERT INTO past_cases (timestamp, data) VALUES (?, ?)",
            ("2026-01-01T00:00:00", json.dumps(case))
        )
        conn.commit()

    loaded = data_cases.load_all_cases()
    assert len(loaded) == 1
    assert loaded[0]["company_name"] == "テスト株式会社"
    assert loaded[0]["score"] == 85.5


def test_load_all_cases_multiple(tmp_db, monkeypatch):
    import data_cases
    import sqlite3
    import json
    from contextlib import closing

    monkeypatch.setattr(data_cases, "DB_PATH", tmp_db)

    cases = [
        {"id": f"test_id_{i}", "company_name": f"会社{i}", "score": float(i * 10)}
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


def test_enrich_rate_fields_infers_base_rate(monkeypatch):
    import data_cases

    def fake_get_base_rate_by_term(month=None, lease_term_months=60):
        assert month == "2025-10"
        assert lease_term_months == 60
        return 1.93

    import base_rate_master
    monkeypatch.setattr(base_rate_master, "get_base_rate_by_term", fake_get_base_rate_by_term)

    case = {
        "timestamp": "2025-10-01T00:00:00",
        "base_rate_at_time": 0,
        "final_rate": 2.50,
        "inputs": {"lease_term": 60},
    }
    data_cases._enrich_rate_fields(case)

    assert case["base_rate_at_time"] == pytest.approx(1.93)
    assert case["winning_spread"] == pytest.approx(0.57)
