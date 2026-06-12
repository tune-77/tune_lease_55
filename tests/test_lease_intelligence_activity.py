import json

from lease_intelligence_activity import observe_user_behavior, record_user_activity


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
