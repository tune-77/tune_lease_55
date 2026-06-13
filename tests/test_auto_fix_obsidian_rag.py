from scripts.auto_fix_obsidian_rag import is_safe_improvement, run_auto_fix


def _summary(*, hit_at_k: int, hit_at_1: int, forbidden: int, passed_ids: set[str]) -> dict:
    total = 3
    return {
        "total": total,
        "hit_at_k": hit_at_k,
        "hit_at_1": hit_at_1,
        "mrr": hit_at_1 / total,
        "forbidden_cases": forbidden,
        "passed": hit_at_k == total and forbidden == 0,
        "cases": [
            {"id": case_id, "rank": 1 if case_id in passed_ids else 0, "forbidden_paths": [], "passed": case_id in passed_ids}
            for case_id in ("a", "b", "c")
        ],
    }


def test_safe_improvement_rejects_regression_of_existing_pass():
    baseline = _summary(hit_at_k=2, hit_at_1=2, forbidden=0, passed_ids={"a", "b"})
    candidate = _summary(hit_at_k=2, hit_at_1=2, forbidden=0, passed_ids={"a", "c"})

    assert is_safe_improvement(baseline, candidate) is False


def test_run_auto_fix_does_nothing_when_baseline_is_healthy():
    healthy = _summary(hit_at_k=3, hit_at_1=3, forbidden=0, passed_ids={"a", "b", "c"})
    calls = []

    report = run_auto_fix(
        cases=[],
        config={"keyword_pool_multiplier": 4},
        evaluate_config=lambda config: calls.append(config) or healthy,
    )

    assert report["status"] == "healthy"
    assert report["selected"] is None
    assert len(calls) == 1


def test_run_auto_fix_selects_candidate_without_regressing_passes():
    baseline = _summary(hit_at_k=2, hit_at_1=2, forbidden=0, passed_ids={"a", "b"})
    improved = _summary(hit_at_k=3, hit_at_1=3, forbidden=0, passed_ids={"a", "b", "c"})
    baseline["cases"][2]["rank"] = 0
    cases = [
        {"id": "a", "expected_path_any": ["リース知識/a.md"], "forbidden_path_any": []},
        {"id": "b", "expected_path_any": ["リース知識/b.md"], "forbidden_path_any": []},
        {"id": "c", "expected_path_any": ["03-知識_業界/c.md"], "forbidden_path_any": []},
    ]

    report = run_auto_fix(
        cases=cases,
        config={"keyword_pool_multiplier": 4, "preferred_path_boosts": {}, "low_priority_path_penalties": {}},
        evaluate_config=lambda config: improved if config.get("keyword_pool_multiplier", 4) > 4 else baseline,
    )

    assert report["status"] == "applied"
    assert report["selected"]["summary"]["passed"] is True
