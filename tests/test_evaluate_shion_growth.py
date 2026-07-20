from scripts import evaluate_shion_growth as growth_eval


def _snapshot(
    date,
    *,
    score=60.0,
    active_rules=8,
    user_evidence=12,
    concepts=8,
    risk_axes=5,
    field_validation=0.0,
    negative_signal=0.0,
):
    return {
        "date": date,
        "score": score,
        "components": {
            "reuse_proxy": 60.0,
            "judgment_change_proxy": 67.0,
            "human_alignment_proxy": 70.0,
            "field_validation": field_validation,
            "negative_signal": negative_signal,
        },
        "counts": {
            "active_rules": active_rules,
            "user_evidence": user_evidence,
            "concepts": concepts,
            "risk_axes": risk_axes,
        },
    }


def test_inventory_growth_without_field_feedback_is_not_full_growth():
    payload = growth_eval.build_growth_evaluation(
        start_date="2026-07-01",
        end_date="2026-07-31",
        growth_rows=[
            _snapshot("2026-07-01", active_rules=2, user_evidence=2, concepts=3),
            _snapshot("2026-07-31", active_rules=10, user_evidence=18, concepts=10),
        ],
        feedback_rows=[],
    )

    assert payload["judgment"]["code"] == "inventory_only"
    assert "実案件" in payload["judgment"]["summary"]
    assert payload["dimensions"]["inventory"]["active_rules_delta"] == 8
    assert payload["dimensions"]["field_validation"]["used"] == 0


def test_field_feedback_can_prove_partial_or_full_growth():
    payload = growth_eval.build_growth_evaluation(
        start_date="2026-07-01",
        end_date="2026-07-31",
        growth_rows=[
            _snapshot("2026-07-01", score=55.0, active_rules=4, user_evidence=4, concepts=4),
            _snapshot(
                "2026-07-31",
                score=72.0,
                active_rules=12,
                user_evidence=20,
                concepts=11,
                field_validation=35.0,
            ),
        ],
        feedback_rows=[
            {"rule_id": "rule-a", "outcome": "helped", "case_id": "case-1", "used_at": "2026-07-20T10:00:00"},
            {"rule_id": "rule-b", "outcome": "used", "case_id": "case-2", "used_at": "2026-07-21T10:00:00"},
            {"rule_id": "rule-c", "outcome": "helped", "case_id": "case-3", "used_at": "2026-07-22T10:00:00"},
        ],
    )

    assert payload["judgment"]["code"] in {"grown", "partial"}
    assert payload["dimensions"]["field_validation"]["used"] == 3
    assert payload["dimensions"]["field_validation"]["helped"] == 2
    assert payload["evidence"]["feedback"]["distinct_cases"] == 3


def test_negative_signal_or_large_score_drop_marks_regression():
    payload = growth_eval.build_growth_evaluation(
        start_date="2026-07-01",
        end_date="2026-07-31",
        growth_rows=[
            _snapshot("2026-07-01", score=75.0, negative_signal=0.0),
            _snapshot("2026-07-31", score=60.0, negative_signal=80.0),
        ],
    )

    assert payload["judgment"]["code"] == "regressed"


def test_markdown_contains_judgment_dimensions_and_next_actions():
    payload = growth_eval.build_growth_evaluation(
        start_date="2026-07-01",
        end_date="2026-07-31",
        growth_rows=[_snapshot("2026-07-31")],
        feedback_rows=[],
    )

    markdown = growth_eval.build_markdown(payload)

    assert "# Shion Growth Evaluation" in markdown
    assert "Result:" in markdown
    assert "Field validation" in markdown
    assert "Next Actions" in markdown
