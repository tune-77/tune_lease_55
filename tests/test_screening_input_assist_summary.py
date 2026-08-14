from api.routers.feedback_loop import _summarize_screening_input_assist_events


def test_screening_input_assist_summary_counts_copy_and_submit_metrics():
    rows = [
        {"action": "input_started", "session_id": "s1"},
        {"action": "search_click", "session_id": "s1"},
        {
            "action": "copied",
            "session_id": "s1",
            "ts": "2026-08-15T09:00:00+00:00",
            "copied_field_count": 4,
            "confirm_count": 2,
        },
        {"action": "score_submitted", "session_id": "s1", "changed_after_copy_count": 1, "elapsed_ms": 90_000},
        {"action": "search_click", "session_id": "s2"},
        {"action": "score_submitted", "session_id": "s2", "changed_after_copy_count": 0, "elapsed_ms": 30_000},
        {"action": "search_click", "session_id": "s3"},
        {
            "action": "copied",
            "session_id": "s3",
            "ts": "2026-08-15T09:05:00+00:00",
            "copied_field_count": 6,
            "confirm_count": 3,
        },
    ]

    summary = _summarize_screening_input_assist_events(rows)

    assert summary["event_count"] == 8
    assert summary["session_count"] == 3
    assert summary["by_action"] == {
        "input_started": 1,
        "search_click": 3,
        "copied": 2,
        "score_submitted": 2,
    }
    assert summary["search_count"] == 3
    assert summary["copy_count"] == 2
    assert summary["copy_rate"] == 0.667
    assert summary["score_submit_count"] == 2
    assert summary["submitted_after_copy_count"] == 1
    assert summary["submitted_after_copy_rate"] == 0.5
    assert summary["avg_copied_fields"] == 5.0
    assert summary["avg_confirm_fields"] == 2.5
    assert summary["avg_changed_after_copy"] == 0.5
    assert summary["avg_elapsed_after_copy_ms"] == 90_000


def test_screening_input_assist_summary_handles_empty_rows():
    summary = _summarize_screening_input_assist_events([])

    assert summary["event_count"] == 0
    assert summary["session_count"] == 0
    assert summary["by_action"] == {}
    assert summary["copy_rate"] is None
    assert summary["submitted_after_copy_rate"] is None
    assert summary["avg_elapsed_after_copy_ms"] is None
