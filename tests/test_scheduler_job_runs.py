import datetime as dt
import json
from types import SimpleNamespace

from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler

import api.scheduler as scheduler_module


def test_schedule_catchup_jobs_for_missing_completed_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    now = dt.datetime(2026, 10, 5, 5, 0)
    runs_path = tmp_path / "scheduler_job_runs.jsonl"
    runs_path.write_text(
        json.dumps(
            {
                "ts": now.replace(hour=3).isoformat(timespec="seconds"),
                "job_id": "shion_feedback_loop_daily",
                "status": "ok",
                "result": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scheduler = BackgroundScheduler(timezone="Asia/Tokyo")

    scheduled = scheduler_module._schedule_catchup_jobs(scheduler, now)

    assert scheduled == [
        "crystallization_daily",
        "shion_usage_loop_daily",
        "shion_memory_decay_daily",
        "shion_inactivity_decay_daily",
        "chat_summary_refresh_daily",
    ]
    assert {job.id for job in scheduler.get_jobs()} == {f"{job_id}_catchup" for job_id in scheduled}


def test_schedule_catchup_jobs_before_daily_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "scheduler_job_runs.jsonl").write_text("", encoding="utf-8")
    scheduler = BackgroundScheduler(timezone="Asia/Tokyo")

    assert scheduler_module._schedule_catchup_jobs(scheduler, dt.datetime(2026, 10, 5, 1, 0)) == []
    assert scheduler.get_jobs() == []


def test_schedule_catchup_jobs_skips_first_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    scheduler = BackgroundScheduler(timezone="Asia/Tokyo")

    assert scheduler_module._schedule_catchup_jobs(scheduler, dt.datetime(2026, 10, 5, 5, 0)) == []
    assert scheduler.get_jobs() == []


def test_schedule_catchup_jobs_skips_when_ledger_is_unreadable(monkeypatch):
    class UnreadableLedger:
        def exists(self) -> bool:
            return True

        def read_text(self, encoding: str) -> str:
            raise OSError("permission denied")

    monkeypatch.setattr(scheduler_module, "_job_runs_path", lambda: UnreadableLedger())
    scheduler = BackgroundScheduler(timezone="Asia/Tokyo")

    assert scheduler_module._schedule_catchup_jobs(scheduler, dt.datetime(2026, 10, 5, 5, 0)) == []
    assert scheduler.get_jobs() == []


def test_feedback_loop_reports_gemini_failure(monkeypatch):
    import api.feedback_pattern_loop as feedback_loop

    monkeypatch.setattr(
        feedback_loop,
        "generate_proposals",
        lambda: {"generated": False, "reason": "Gemini生成に失敗: Timeout", "proposals": []},
    )
    monkeypatch.setattr(feedback_loop, "evaluate_proposal_impact", lambda: {"evaluated": 0})

    result = scheduler_module.run_shion_feedback_loop()

    assert result["status"] == "error"
    assert result["reason"] == "Gemini生成に失敗: Timeout"


def test_scheduler_listener_records_catchup_result(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    event = SimpleNamespace(
        job_id="shion_usage_loop_daily_catchup",
        retval={"status": "no_proposals", "reason": "利用データなし"},
        exception=None,
        code=EVENT_JOB_EXECUTED,
    )

    scheduler_module._scheduler_job_listener(event)

    record = json.loads((tmp_path / "scheduler_job_runs.jsonl").read_text(encoding="utf-8"))
    assert record["job_id"] == "shion_usage_loop_daily"
    assert record["status"] == "no_proposals"
    assert record["result"] == event.retval


def test_scheduler_listener_does_not_record_missed_job(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    event = SimpleNamespace(job_id="shion_usage_loop_daily", code=EVENT_JOB_MISSED)

    scheduler_module._scheduler_job_listener(event)

    assert not (tmp_path / "scheduler_job_runs.jsonl").exists()
