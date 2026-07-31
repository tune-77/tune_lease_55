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
