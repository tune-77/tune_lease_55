import datetime as dt
import json

from lease_intelligence_activity import (
    ALLOWED_ACTIONS,
    ALLOWED_SURFACES,
    observe_user_behavior,
    record_user_activity,
    suggest_related_feature,
)


def test_activity_log_accepts_only_bounded_events_and_dedupes(tmp_path):
    log = tmp_path / "activity.jsonl"

    assert record_user_activity(
        "home",
        "page_view",
        event_id="home-1",
        occurred_at="2026-06-12T09:00:00",
        log_path=log,
    )
    assert not record_user_activity(
        "home",
        "page_view",
        event_id="home-1",
        occurred_at="2026-06-12T09:01:00",
        log_path=log,
    )
    assert not record_user_activity("desktop", "key_log", log_path=log)
    rows = log.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert "question" not in json.loads(rows[0])


def test_observation_stores_categories_not_question_text(tmp_path):
    activity = tmp_path / "activity.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    metrics = tmp_path / "metrics.json"
    activity.write_text(
        json.dumps(
            {
                "timestamp": "2026-06-12T08:00:00",
                "surface": "improvement_log",
                "action": "page_view",
                "event_id": "one",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    secret_question = "レンタカーと車検について詳しく知りたい"
    prompts.write_text(
        json.dumps(
            {
                "timestamp": "2026-06-12T10:00:00",
                "surface": "next_chat_general",
                "question": secret_question,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics.write_text(
        json.dumps(
            {"days": {"2026-06-12": {"views": 2, "judgment_changes": 1}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    observation = observe_user_behavior(
        "2026-06-13",
        activity_log=activity,
        prompt_log=prompts,
        news_metrics=metrics,
    )

    serialized = json.dumps(observation, ensure_ascii=False)
    assert observation["observed"]
    assert observation["actions"]["chat_message"] == 1
    assert observation["actions"]["news_view"] == 2
    assert "車・移動" in [item["label"] for item in observation["interests"]]
    assert secret_question not in serialized
    assert "システムがどう改善されるか" in observation["understanding"]


def test_frontend_surfaces_are_all_allowed():
    # フロントエンドが送る surface がサイレントに捨てられないことの番人テスト
    for surface in (
        "home",
        "chat",
        "improvement_log",
        "lease_intelligence_dialogue",
        "simulator:screening",
        "simulator:lease-intelligence",
    ):
        assert surface in ALLOWED_SURFACES


def test_simulator_activity_is_recorded_not_silently_dropped(tmp_path):
    # LeasePaymentSimulator (screening / lease-intelligence 埋め込み) が送る
    # action がサイレントに捨てられないことの番人テスト
    log = tmp_path / "activity.jsonl"
    for action in ("simulator_view", "simulator_input_changed"):
        assert action in ALLOWED_ACTIONS

    assert record_user_activity(
        "simulator:screening",
        "simulator_view",
        event_id="simulator-view-1",
        occurred_at="2026-07-31T09:00:00",
        log_path=log,
    )
    assert record_user_activity(
        "simulator:lease-intelligence",
        "simulator_input_changed",
        event_id="simulator-input-1",
        occurred_at="2026-07-31T09:01:00",
        log_path=log,
    )
    rows = log.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def test_dialogue_visit_is_recorded_and_reflected_in_understanding(tmp_path):
    log = tmp_path / "activity.jsonl"

    recorded = record_user_activity(
        surface="lease_intelligence_dialogue",
        action="page_view",
        event_id="lease-intelligence-activity:dialogue:2026-06-12",
        occurred_at="2026-06-12T09:00:00",
        log_path=log,
    )
    assert recorded is True

    observation = observe_user_behavior(
        "2026-06-13",
        activity_log=log,
        prompt_log=tmp_path / "missing_prompts.jsonl",
        news_metrics=tmp_path / "missing_metrics.json",
    )

    assert observation["observed"] is True
    assert observation["surfaces"]["lease_intelligence_dialogue"] == 1
    assert "対話室" in observation["understanding"]


def test_suggest_related_feature_none_when_no_pattern_matches(tmp_path):
    log = tmp_path / "activity.jsonl"
    assert suggest_related_feature(activity_log=log, today=dt.date(2026, 8, 1)) is None


def test_suggest_related_feature_flags_heavy_improvement_log_use(tmp_path):
    log = tmp_path / "activity.jsonl"
    for i in range(5):
        record_user_activity(
            "improvement_log",
            "page_view",
            event_id=f"improvement-log-{i}",
            occurred_at=f"2026-07-2{i}T09:00:00",
            log_path=log,
        )
    suggestion = suggest_related_feature(activity_log=log, today=dt.date(2026, 8, 1))
    assert suggestion is not None
    assert suggestion["surface"] == "lease_intelligence_dialogue"
    assert suggestion["link"] == "/lease-intelligence"


def test_suggest_related_feature_stays_quiet_once_target_already_used(tmp_path):
    log = tmp_path / "activity.jsonl"
    for i in range(5):
        record_user_activity(
            "improvement_log",
            "page_view",
            event_id=f"improvement-log-{i}",
            occurred_at=f"2026-07-2{i}T09:00:00",
            log_path=log,
        )
    record_user_activity(
        "lease_intelligence_dialogue",
        "page_view",
        event_id="dialogue-already-visited-1",
        occurred_at="2026-07-26T09:00:00",
        log_path=log,
    )
    record_user_activity(
        "lease_intelligence_dialogue",
        "page_view",
        event_id="dialogue-already-visited-2",
        occurred_at="2026-07-27T09:00:00",
        log_path=log,
    )
    assert suggest_related_feature(activity_log=log, today=dt.date(2026, 8, 1)) is None


def test_suggest_related_feature_ignores_events_outside_window(tmp_path):
    log = tmp_path / "activity.jsonl"
    for i in range(5):
        record_user_activity(
            "improvement_log",
            "page_view",
            event_id=f"improvement-log-old-{i}",
            occurred_at=f"2026-01-0{i + 1}T09:00:00",
            log_path=log,
        )
    assert suggest_related_feature(activity_log=log, today=dt.date(2026, 8, 1)) is None
