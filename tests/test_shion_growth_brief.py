from scripts import build_shion_growth_brief as brief


def test_build_growth_brief_combines_memory_and_judgment_reports():
    payload = brief.build_brief(
        target_date="2026-07-31",
        memory_effect={
            "summary": {
                "records": 10,
                "used_memory_ids": 3,
                "usage_events": 5,
                "usage_events_with_impact_hints": 0,
            },
            "top_used": [{"id": "mem_1", "memory_layer": "long_term", "status": "active", "used_count": 2, "content": "判断軸"}],
            "review_candidates": [{"id": "mem_old", "status": "stale", "content": "古い"}],
        },
        judgment_ab={
            "summary": {"candidate_pairs": 1},
            "pairs": [{"concept": "asset_life", "asset_a": {"rule_id": "a"}, "asset_b": {"rule_id": "b"}}],
        },
        persistent_audit={"summary": {"findings": 0}},
        field_review={"summary": {"grow": 1, "review": 2, "sleeping": 3}},
    )

    assert payload["summary"]["memory_records"] == 10
    assert payload["summary"]["judgment_asset_ab_pairs"] == 1
    assert payload["summary"]["judgment_assets_sleeping"] == 3
    assert any("判断資産A/B候補" in action for action in payload["actions"])
    assert any("impact_hints" in action for action in payload["actions"])
