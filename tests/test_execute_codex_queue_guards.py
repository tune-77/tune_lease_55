"""自律実行ガード（planning/shion_autonomy_guards.md）。

- キルスイッチ CODEX_QUEUE_DISABLED で即時停止
- 日次実行上限 CODEX_QUEUE_DAILY_LIMIT で持ち越し
- 連続失敗 CODEX_QUEUE_MAX_CONSECUTIVE_FAILURES で残りを停止
"""

from __future__ import annotations

import json
import sys

import pytest

from scripts import execute_codex_queue as executor


@pytest.fixture()
def env(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(executor, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(executor, "record_status", lambda *a, **k: None)
    monkeypatch.setattr(executor, "_get_gemini_api_key", lambda root: "")
    monkeypatch.delenv("CODEX_QUEUE_DISABLED", raising=False)
    monkeypatch.delenv("CODEX_QUEUE_DAILY_LIMIT", raising=False)
    monkeypatch.delenv("CODEX_QUEUE_MAX_CONSECUTIVE_FAILURES", raising=False)
    return tmp_path


def _write_queue(tmp_path, n_items: int):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "items": [
                    {"id": f"REV-{500 + i}", "title": f"候補{i}", "prompt": f"p{i}"}
                    for i in range(n_items)
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return queue_path


def _run_main(monkeypatch, queue_path, output_path):
    monkeypatch.setattr(
        sys, "argv",
        ["execute_codex_queue.py", "--queue", str(queue_path), "--output", str(output_path)],
    )
    executor.main()


def test_kill_switch_stops_everything(env, monkeypatch, capsys):
    queue_path = _write_queue(env, 2)
    monkeypatch.setenv("CODEX_QUEUE_DISABLED", "1")
    calls = []
    monkeypatch.setattr(executor, "run_item", lambda item, **k: calls.append(item) or {})

    _run_main(monkeypatch, queue_path, env / "reports" / "out.json")

    assert calls == []
    assert not (env / "reports" / "out.json").exists()
    assert "キルスイッチ" in capsys.readouterr().out


def test_daily_limit_carries_over(env, monkeypatch):
    queue_path = _write_queue(env, 3)
    monkeypatch.setenv("CODEX_QUEUE_DAILY_LIMIT", "2")
    # 本日すでに1件実行済み → 残り枠1
    import datetime as dt
    date_tag = dt.date.today().strftime("%Y%m%d")
    (env / "reports" / f"codex_queue_result_{date_tag}_early.json").write_text(
        json.dumps({"results": [{"id": "REV-000", "exit_code": 0}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    executed = []
    monkeypatch.setattr(
        executor, "run_item",
        lambda item, **k: executed.append(str(item["id"])) or {
            "id": item["id"], "title": item["title"], "exit_code": 0,
            "stdout": "", "stderr": "", "backend": "claude",
            "started_at": "", "finished_at": "",
        },
    )

    _run_main(monkeypatch, queue_path, env / "reports" / "out.json")

    assert executed == ["REV-500"]
    report = json.loads((env / "reports" / "out.json").read_text(encoding="utf-8"))
    assert report["guards"]["carried_over"] == ["REV-501", "REV-502"]
    assert report["guards"]["already_executed_today"] == 1


def test_consecutive_failures_abort_remaining(env, monkeypatch):
    queue_path = _write_queue(env, 3)
    monkeypatch.setenv("CODEX_QUEUE_DAILY_LIMIT", "10")
    monkeypatch.setenv("CODEX_QUEUE_MAX_CONSECUTIVE_FAILURES", "2")
    executed = []

    def fail_item(item, **k):
        executed.append(str(item["id"]))
        return {
            "id": item["id"], "title": item["title"], "exit_code": 1,
            "stdout": "", "stderr": "boom", "backend": "none",
            "started_at": "", "finished_at": "",
        }

    monkeypatch.setattr(executor, "run_item", fail_item)

    with pytest.raises(SystemExit):
        _run_main(monkeypatch, queue_path, env / "reports" / "out.json")

    assert executed == ["REV-500", "REV-501"]  # 3件目は実行されない
    report = json.loads((env / "reports" / "out.json").read_text(encoding="utf-8"))
    assert report["guards"]["aborted_by_consecutive_failures"] is True
    assert report["total"] == 2
