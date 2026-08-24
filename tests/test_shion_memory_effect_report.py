from scripts import build_shion_memory_effect_report as report


def test_build_memory_effect_report_counts_layers_and_review_candidates():
    index = {
        "records": [
            {
                "id": "mem_1",
                "content": "長期判断軸",
                "memory_layer": "long_term",
                "memory_type": "judgment_memory",
                "status": "active",
            },
            {
                "id": "mem_2",
                "content": "古い記憶",
                "memory_layer": "long_term",
                "memory_type": "factual_memory",
                "status": "stale",
            },
            {
                "id": "mem_3",
                "content": "永続原則",
                "memory_layer": "persistent",
                "memory_type": "value_memory",
                "status": "active",
            },
        ]
    }
    usage = [{"refs": ["mem_1"], "question": "審査判断", "impact_hints": [{"id": "mem_1"}]}]

    payload = report.build_report(index, usage, [])

    assert payload["summary"]["used_memory_ids"] == 1
    assert payload["summary"]["usage_by_layer"]["long_term"] == 1
    assert payload["summary"]["usage_events_with_impact_hints"] == 1
    assert payload["review_candidates"][0]["id"] == "mem_2"
    assert payload["unused_persistent"][0]["id"] == "mem_3"


def test_build_memory_effect_report_scores_answer_utility():
    index = {
        "records": [
            {
                "id": "mem_helpful",
                "content": "境界案件では条件付き承認を検討する。",
                "memory_layer": "long_term",
                "memory_type": "judgment_memory",
                "status": "active",
                "domain": "lease_screening",
                "use_when": "案件審査で使う",
            },
            {
                "id": "mem_needs_feedback",
                "content": "古いが頻出する運用メモ。",
                "memory_layer": "long_term",
                "memory_type": "technical_memory",
                "status": "active",
            },
            {
                "id": "mem_stale_used",
                "content": "古い判断軸。",
                "memory_layer": "long_term",
                "memory_type": "judgment_memory",
                "status": "stale",
            },
        ]
    }
    usage = [
        {
            "refs": ["mem_helpful", "mem_needs_feedback", "mem_stale_used"],
            "route": "case_screening",
            "question": "境界案件の確認",
            "impact_hints": [{"id": "mem_helpful"}],
        },
        {
            "refs": ["mem_helpful", "mem_needs_feedback"],
            "route": "case_screening",
            "question": "条件付き承認の確認",
            "impact_hints": [{"id": "mem_helpful"}],
        },
        {"refs": ["mem_needs_feedback"], "route": "implementation", "question": "運用確認"},
    ]

    payload = report.build_report(index, usage, [])

    assert payload["summary"]["utility_by_state"]["likely_helpful"] == 1
    assert payload["summary"]["utility_by_state"]["needs_feedback"] == 1
    assert payload["summary"]["utility_by_state"]["needs_review"] == 1
    assert payload["likely_helpful"][0]["id"] == "mem_helpful"
    assert payload["likely_helpful"][0]["domain"] == "lease_screening"
    assert payload["needs_feedback"][0]["id"] == "mem_needs_feedback"
    assert payload["possible_noise"][0]["id"] == "mem_stale_used"
    assert payload["needs_feedback_triage"]["record_count"] == 1
    assert payload["needs_feedback_triage"]["batch_count"] == 1


def test_build_memory_effect_report_uses_explicit_memory_feedback():
    index = {
        "records": [
            {
                "id": "mem_validated",
                "content": "銀行支援は対象リースへの直接性を見る。",
                "memory_layer": "long_term",
                "memory_type": "judgment_memory",
                "status": "active",
            },
            {
                "id": "mem_challenged",
                "content": "古い条件設定。",
                "memory_layer": "long_term",
                "memory_type": "judgment_memory",
                "status": "active",
            },
        ]
    }
    usage = [
        {"refs": ["mem_validated"], "memory_feedback": "helped", "route": "case_screening"},
        {"refs": ["mem_challenged"], "memory_feedback": "challenged", "route": "case_screening"},
    ]

    payload = report.build_report(index, usage, [])

    validated = next(item for item in payload["likely_helpful"] if item["id"] == "mem_validated")
    challenged = next(item for item in payload["possible_noise"] if item["id"] == "mem_challenged")
    assert validated["utility_state"] == "validated"
    assert validated["explicit_feedback"]["helped"] == 1
    assert challenged["utility_state"] == "challenged"
    assert challenged["explicit_feedback"]["challenged"] == 1


def test_needs_feedback_triage_groups_all_records_not_only_top_twenty():
    records = [
        {
            "id": f"mem_{idx}",
            "content": f"確認待ち記憶 {idx}",
            "memory_layer": "long_term",
            "memory_type": "judgment_memory",
            "status": "active",
            "domain": "credit",
        }
        for idx in range(25)
    ]
    usage = [
        {"refs": [f"mem_{idx}"], "route": "case_screening", "question": f"質問 {idx}"}
        for idx in range(25)
        for _ in range(3)
    ]

    payload = report.build_report({"records": records}, usage, [])

    assert len(payload["needs_feedback"]) == 20
    assert payload["needs_feedback_triage"]["record_count"] == 25
    assert payload["needs_feedback_triage"]["batch_count"] == 1
    assert payload["needs_feedback_triage"]["top_batches"][0]["count"] == 25
