from __future__ import annotations

import json
from pathlib import Path

from scripts.notify_case_deletion_integrity_slack import (
    build_slack_payload,
    notify_if_abnormal,
)


VALID_WEBHOOK = "https://hooks.slack.com/services/T12345678/B12345678/abcdefghijklmnopqrstuvwxyz"


def _report(status: str = "FAIL") -> dict:
    return {
        "status": status,
        "summary": {"events": 2, "errors": 1, "warnings": 0},
        "issues": [
            {
                "code": "deleted_count_mismatch",
                "severity": "error",
                "event_id": "del_test",
                "recorded": 2,
                "observed": 1,
            }
        ],
    }


def test_pass_clears_active_incident_without_sending(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"active_fingerprint": "old", "last_sent_at": "before"}), encoding="utf-8")
    calls = []

    result, _detail = notify_if_abnormal(
        _report("PASS"),
        state_path=state,
        webhook_url=VALID_WEBHOOK,
        sender=lambda *_args: calls.append(True) or (True, "ok"),
    )

    saved = json.loads(state.read_text(encoding="utf-8"))
    assert result == "healthy_skip"
    assert calls == []
    assert saved["active_fingerprint"] == ""
    assert saved["last_status"] == "PASS"


def test_abnormal_alert_is_sent_once_until_recovery(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    calls = []

    def sender(_webhook: str, payload: dict) -> tuple[bool, str]:
        calls.append(payload)
        return True, "ok"

    first, _ = notify_if_abnormal(
        _report(), state_path=state, webhook_url=VALID_WEBHOOK, sender=sender
    )
    duplicate, _ = notify_if_abnormal(
        _report(), state_path=state, webhook_url=VALID_WEBHOOK, sender=sender
    )
    notify_if_abnormal(_report("PASS"), state_path=state, webhook_url="", sender=sender)
    recurrence, _ = notify_if_abnormal(
        _report(), state_path=state, webhook_url=VALID_WEBHOOK, sender=sender
    )

    assert (first, duplicate, recurrence) == ("sent", "duplicate_skip", "sent")
    assert len(calls) == 2
    assert "自動修復" in calls[0]["text"]


def test_dry_run_and_missing_webhook_do_not_mark_alert_sent(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    dry_run, detail = notify_if_abnormal(
        _report(), state_path=state, webhook_url="", dry_run=True
    )
    missing, _ = notify_if_abnormal(_report(), state_path=state, webhook_url="")

    assert dry_run == "dry_run"
    assert "案件削除監査" in detail
    assert missing == "webhook_missing"
    assert not state.exists()


def test_slack_payload_limits_issue_details() -> None:
    report = _report()
    report["issues"] = [
        {"code": f"issue_{index}", "severity": "error", "event_id": f"del_{index}"}
        for index in range(7)
    ]
    text = build_slack_payload(report)["text"]

    assert "issue_4" in text
    assert "issue_5" not in text
    assert "他 2 件" in text
