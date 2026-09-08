"""Exercise the real delete functions against SQLite without starting AI services."""
import ast
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
import sys
import types
from unittest.mock import Mock

import pytest
from starlette.exceptions import HTTPException

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def deletion(tmp_path, monkeypatch):
    db = tmp_path / "cases.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE past_cases (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO past_cases VALUES ('exists')")
    @contextmanager
    def connection():
        conn = sqlite3.connect(db)
        try:
            yield conn
        finally:
            conn.close()
    cache = Mock()
    storage_ns = {"os": os, "DB_PATH": str(db), "_cloud_db_enabled": lambda: False,
        "_db_placeholder": lambda: "?", "_case_db_connection": connection,
        "refresh_stats_caches": cache, "logger": logging.getLogger(__name__)}
    delete = load_function("data_cases.py", "delete_case", storage_ns)
    module = types.ModuleType("data_cases")
    module.delete_case = delete
    monkeypatch.setitem(sys.modules, "data_cases", module)
    score, event = Mock(return_value=False), Mock(return_value=False)
    ns = {"logger": logging.getLogger(__name__), "HTTPException": HTTPException,
        "_git_push_db": Mock(),
        "_parse_cloudrun_score_case_id": lambda x: 1 if x == "cloudrun_score:1" else None,
        "_parse_cloudrun_event_case_id": lambda x: "e" if x == "cloudrun_event:e" else "",
        "_reject_cloudrun_score_pending_case": score,
        "_reject_cloudrun_event_pending_case": event}
    endpoint = load_function("api/main.py", "delete_case", ns)
    return endpoint, storage_ns, db, score, event


def test_success_deletes_row_and_queues_persistence(deletion):
    endpoint, _, db, _, _ = deletion
    tasks = Mock()
    assert endpoint("exists", tasks)["message"] == "Deleted"
    tasks.add_task.assert_called_once()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()[0] == 0


def test_absent_case_returns_404_without_persistence(deletion):
    endpoint, _, _, _, _ = deletion
    tasks = Mock()
    with pytest.raises(HTTPException) as exc:
        endpoint("absent", tasks)
    assert exc.value.status_code == 404
    tasks.add_task.assert_not_called()


def test_db_failure_returns_503_and_preserves_row(deletion):
    endpoint, ns, db, _, _ = deletion
    ns["_case_db_connection"] = Mock(side_effect=sqlite3.OperationalError("locked"))
    tasks = Mock()
    with pytest.raises(HTTPException) as exc:
        endpoint("exists", tasks)
    assert exc.value.status_code == 503
    tasks.add_task.assert_not_called()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()[0] == 1


def test_missing_database_returns_503(deletion):
    endpoint, _, db, _, _ = deletion
    db.unlink()
    with pytest.raises(HTTPException) as exc:
        endpoint("exists", Mock())
    assert exc.value.status_code == 503


def test_cache_failure_after_commit_is_still_success(deletion):
    endpoint, ns, db, _, _ = deletion
    ns["refresh_stats_caches"].side_effect = OSError("cache unavailable")
    assert endpoint("exists", Mock())["message"] == "Deleted"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()[0] == 0


@pytest.mark.parametrize("case_id,index", [("cloudrun_score:1", 3), ("cloudrun_event:e", 4)])
def test_pending_rejection_errors_are_not_success(deletion, case_id, index):
    endpoint = deletion[0]
    deletion[index].side_effect = OSError("write failed")
    tasks = Mock()
    with pytest.raises(HTTPException) as exc:
        endpoint(case_id, tasks)
    assert exc.value.status_code == 503
    deletion[index].assert_called_once_with(case_id, raise_on_error=True)
    tasks.add_task.assert_not_called()
