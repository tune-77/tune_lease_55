import json

import silent_failure_log as sfl
from api.background_executor import background_executor


def _save_note():
    raise OSError("disk full")


def test_background_failure_is_recorded(tmp_path, monkeypatch):
    sfl._last_write.clear()
    path = tmp_path / "sf.jsonl"
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(path))
    background_executor.submit(_save_note).exception(timeout=5)
    background_executor.submit(lambda: 1).result(timeout=5)
    background_executor.submit(lambda: None).result(timeout=5)
    import time

    for _ in range(50):
        if path.exists():
            break
        time.sleep(0.02)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(r["component"], r["exc_type"]) for r in rows] == [
        ("background.tests.test_background_executor_records_failures._save_note", "OSError")
    ]
