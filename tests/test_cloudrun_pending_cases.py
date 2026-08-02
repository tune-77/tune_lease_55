import json

from api.cloudrun_pending_cases import (
    cloudrun_score_pending_item,
    cloudrun_score_pending_item_from_event,
    list_cloudrun_score_pending_cases_from_events,
    loads_dict,
    parse_cloudrun_event_case_id,
    parse_cloudrun_score_case_id,
)


def test_parse_cloudrun_case_ids_and_loads_dict():
    assert parse_cloudrun_score_case_id("cloudrun_score:12") == 12
    assert parse_cloudrun_score_case_id("cloudrun_score:0") is None
    assert parse_cloudrun_score_case_id("x:12") is None
    assert parse_cloudrun_event_case_id("cloudrun_event:event-1") == "event-1"
    assert parse_cloudrun_event_case_id("x:event-1") == ""
    assert loads_dict({"a": 1}) == {"a": 1}
    assert loads_dict(json.dumps({"a": 1})) == {"a": 1}
    assert loads_dict("not-json") == {}


def test_cloudrun_score_pending_item_prefers_row_and_result_fields():
    item = cloudrun_score_pending_item({
        "id": 7,
        "created_at": "2026-08-02T01:02:03Z",
        "score": "",
        "event_id": "evt-7",
        "inputs_json": json.dumps({"company_name": "A社", "company_no": "C001"}),
        "result_json": json.dumps({"score_base": 58, "industry_sub": "工作機械"}),
        "return_review_status": "",
    })

    assert item["id"] == "cloudrun_score:7"
    assert item["company_name"] == "A社"
    assert item["score"] == 58
    assert item["industry"] == "工作機械"
    assert item["registration_date"] == "2026-08-02"


def test_list_cloudrun_score_pending_cases_from_events_excludes_registered_or_rejected():
    events = [
        {
            "event_type": "score_calculated",
            "event_id": "old",
            "ts": "2026-08-01T00:00:00Z",
            "payload": {"inputs": {"company_name": "古い"}, "result": {"score": 40}},
        },
        {
            "event_type": "score_calculated",
            "event_id": "done",
            "ts": "2026-08-02T00:00:00Z",
            "payload": {"inputs": {"company_name": "登録済み"}, "result": {"score": 50}},
        },
        {
            "event_type": "case_result_registered",
            "payload": {"source_event_id": "done"},
        },
        {
            "event_type": "score_full_calculated",
            "event_id": "new",
            "ts": "2026-08-03T00:00:00Z",
            "payload": {"inputs": {"company_name": "新しい"}, "result": {"score": 60}},
        },
    ]

    rows = list_cloudrun_score_pending_cases_from_events(events, limit=10)

    assert [row["cloudrun_event_id"] for row in rows] == ["new", "old"]
    assert rows[0]["id"] == "cloudrun_event:new"
    assert cloudrun_score_pending_item_from_event({"event_type": "other"}) is None
