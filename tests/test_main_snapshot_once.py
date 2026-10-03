import json

import silent_failure_log as sfl
from api.main import _snapshot_once


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_snapshot_once_records_unuploaded_and_exceptions(tmp_path, monkeypatch):
    sfl._last_write.clear()
    path = tmp_path / "sf.jsonl"
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(path))

    _snapshot_once("backup.test.ok", lambda: {"enabled": True, "uploaded": True})
    _snapshot_once("backup.test.disabled", lambda: {"enabled": False, "uploaded": False})
    assert _rows(path) == []

    _snapshot_once("backup.test.lock", lambda: {"enabled": True, "uploaded": False, "reason": "lock_timeout: held by 山田"})

    def boom():
        raise RuntimeError("x")

    _snapshot_once("backup.test.boom", boom)  # 例外を外へ出さない（定期スレッドを止めない）
    rows = _rows(path)
    assert [(r["component"], r.get("detail"), r["exc_type"]) for r in rows] == [
        ("backup.test.lock", "lock_timeout", ""),
        ("backup.test.boom", None, "RuntimeError"),
    ]
