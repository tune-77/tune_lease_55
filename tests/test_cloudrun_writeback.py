from __future__ import annotations

import json

from api import cloudrun_writeback as writeback


def test_build_cloudrun_input_event_redacts_pii() -> None:
    event = writeback.build_cloudrun_input_event(
        event_type="score_calculated",
        surface="score",
        payload={
            "company_name": "秘密商事",
            "industry_sub": "情報サービス業",
            "nested": {"email": "a@example.com", "asset_name": "サーバー"},
        },
    )

    assert event["payload"]["company_name"] == "[REDACTED]"
    assert event["payload"]["industry_sub"] == "情報サービス業"
    assert event["payload"]["nested"]["email"] == "[REDACTED]"
    assert event["payload"]["nested"]["asset_name"] == "サーバー"


def test_record_cloudrun_input_event_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("CLOUDRUN_INPUT_WRITEBACK_ENABLED", "false")

    result = writeback.record_cloudrun_input_event(
        event_type="test",
        surface="unit",
        payload={"x": 1},
    )

    assert result["ok"] is False
    assert result["skipped"] is True
    assert result["reason"] == "writeback_disabled"


def test_fallback_writes_jsonl(tmp_path, monkeypatch) -> None:
    fallback_path = tmp_path / "failures.jsonl"
    monkeypatch.setattr(writeback, "LOCAL_FALLBACK_PATH", fallback_path)
    entry = writeback.build_cloudrun_input_event(
        event_type="test",
        surface="unit",
        payload={"company_name": "秘密商事"},
    )

    writeback._fallback(entry, "boom")

    rows = [json.loads(line) for line in fallback_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["writeback_error"] == "boom"
    assert rows[0]["payload"]["company_name"] == "[REDACTED]"
