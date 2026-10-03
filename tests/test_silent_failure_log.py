import datetime as dt
import json

import silent_failure_log as sfl


def _reset():
    sfl._last_write.clear()
    sfl._suppressed.clear()


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_record_keeps_type_not_message(tmp_path, monkeypatch):
    _reset()
    path = tmp_path / "sf.jsonl"
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(path))
    sfl.record_silent_failure("memory.chat.save", "save_failed", ValueError("山田太郎 090-1234"), detail="要約保存")
    (row,) = _rows(path)
    assert row["component"] == "memory.chat.save"
    assert row["exc_type"] == "ValueError"
    assert row["critical"] is True
    assert row["where"].startswith("tests/test_silent_failure_log.py:")
    assert "山田" not in path.read_text(encoding="utf-8")


def test_repeats_are_counted_not_written(tmp_path, monkeypatch):
    _reset()
    path = tmp_path / "sf.jsonl"
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(path))
    for _ in range(5):
        sfl.record_silent_failure("misc.x", "swallowed", OSError())
    assert len(_rows(path)) == 1
    sfl._last_write.clear()
    sfl.record_silent_failure("misc.x", "swallowed", OSError())
    assert _rows(path)[-1]["repeat"] == 4


def test_off_and_unwritable_never_raise(tmp_path, monkeypatch):
    _reset()
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", "off")
    sfl.record_silent_failure("misc.x", "swallowed")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(blocker / "sub" / "sf.jsonl"))
    sfl.record_silent_failure("misc.y", "swallowed")


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_morning_lines_quiet_for_known_minor(tmp_path):
    now = dt.datetime(2026, 10, 3, 6, 0).astimezone()
    old = (now - dt.timedelta(days=2)).isoformat()
    new = (now - dt.timedelta(hours=1)).isoformat()
    path = tmp_path / "sf.jsonl"
    _write(path, [{"ts": old, "component": "misc.a", "kind": "swallowed"}, {"ts": new, "component": "misc.a", "kind": "swallowed"}])
    assert sfl.morning_report_lines(path, now=now) == []


def test_morning_lines_flag_critical_new_and_volume(tmp_path):
    now = dt.datetime(2026, 10, 3, 6, 0).astimezone()
    new = (now - dt.timedelta(hours=1)).isoformat()
    path = tmp_path / "sf.jsonl"
    _write(path, [{"ts": new, "component": "backup.obsidian", "kind": "save_failed", "repeat": 2}])
    lines = sfl.morning_report_lines(path, now=now)
    assert "3件" in lines[0]
    assert any("重要部品" in line and "backup.obsidian" in line for line in lines)
    assert any("新しい種類" in line for line in lines)

    old = (now - dt.timedelta(days=2)).isoformat()
    _write(path, [{"ts": old, "component": "misc.a", "kind": "swallowed"}, {"ts": new, "component": "misc.a", "kind": "swallowed", "repeat": 25}])
    lines = sfl.morning_report_lines(path, now=now)
    assert "26件" in lines[0] and "多い順" in lines[1]


def test_launchd_failure_lines_only_nonzero_project_jobs():
    listing = "PID\tStatus\tLabel\n-\t0\tcom.tunelease.ok\n-\t1\tcom.tunelease.branch-cleanup\n123\t2\tcom.lease.slackbot\n-\t1\tcom.apple.other\n"
    (line,) = sfl.launchd_failure_lines(listing)
    assert "`branch-cleanup`=1" in line
    assert "`com.lease.slackbot`=2（常駐" in line
    assert "com.apple" not in line and "`ok`" not in line
    assert sfl.launchd_failure_lines("PID\tStatus\tLabel\n-\t0\tcom.tunelease.ok\n") == []
