import pytest

from scoring_core import _apply_demo_food_service_score_cap


def test_demo_food_service_score_is_capped_to_strong_warning_band() -> None:
    score, adj = _apply_demo_food_service_score_cap({"company_no": "900303"}, 71.2)

    assert score == pytest.approx(35.0)
    assert adj == pytest.approx(-36.2)


def test_demo_food_service_score_can_match_by_company_name() -> None:
    score, adj = _apply_demo_food_service_score_cap({"company_name": "デモフードサービス"}, 62.8)

    assert score == pytest.approx(35.0)
    assert adj == pytest.approx(-27.8)


def test_demo_food_service_cap_does_not_raise_lower_score() -> None:
    score, adj = _apply_demo_food_service_score_cap({"company_no": "900303"}, 34.5)

    assert score == pytest.approx(34.5)
    assert adj == pytest.approx(0.0)


def test_demo_food_service_cap_does_not_touch_other_cases() -> None:
    score, adj = _apply_demo_food_service_score_cap({"company_no": "123456"}, 71.2)

    assert score == pytest.approx(71.2)
    assert adj == pytest.approx(0.0)
