from pathlib import Path

import pytest

from api import shion_tasks


def test_shion_task_ledger_create_update_and_complete(tmp_path: Path):
    path = tmp_path / "shion_tasks.jsonl"

    task = shion_tasks.create_task(
        title="決勝前に最後の30秒文面を磨く",
        due_at="2026-07-20",
        note="業務ごとに育つ静かな紫苑を入れる",
        source="test",
        reminder=True,
        tags=["pitch", "shion"],
        path=path,
    )

    assert task["status"] == "open"
    assert task["title"] == "決勝前に最後の30秒文面を磨く"
    assert task["due_at"] == "2026-07-20"
    assert path.read_text(encoding="utf-8").count("task_created") == 1

    updated = shion_tasks.update_task(
        task["id"],
        due_at="2026-07-21T09:00:00+09:00",
        note="最後の30秒だけに絞る",
        path=path,
    )

    assert updated["due_at"] == "2026-07-21T09:00:00+09:00"
    assert updated["note"] == "最後の30秒だけに絞る"

    done = shion_tasks.set_task_status(task["id"], "done", path=path)
    assert done["status"] == "done"
    assert done["completed_at"]

    assert shion_tasks.list_tasks(status="open", path=path) == []
    assert shion_tasks.list_tasks(status="done", path=path)[0]["id"] == task["id"]


def test_shion_task_ledger_rejects_invalid_due_at(tmp_path: Path):
    with pytest.raises(ValueError):
        shion_tasks.create_task(title="無効な期限", due_at="明日の朝", path=tmp_path / "tasks.jsonl")


def test_shion_task_ledger_rejects_missing_task(tmp_path: Path):
    with pytest.raises(KeyError):
        shion_tasks.set_task_status("task_missing", "done", path=tmp_path / "tasks.jsonl")
