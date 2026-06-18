from __future__ import annotations

import sys
from types import ModuleType


def _import_case_network(monkeypatch):
    fake_data_cases = ModuleType("data_cases")
    fake_data_cases.load_past_cases = lambda: []
    monkeypatch.setitem(sys.modules, "data_cases", fake_data_cases)
    sys.modules.pop("case_network", None)

    import case_network

    return case_network


def test_score_of_uses_saved_case_fallbacks(monkeypatch):
    case_network = _import_case_network(monkeypatch)

    assert case_network._score_of({"result": {"hantei_score": 72.5}}) == 72.5
    assert case_network._score_of({"score_borrower": "61.2"}) == 61.2
    assert round(case_network._score_of({"scoring_result": {"ai_prob": 0.145}}), 1) == 14.5


def test_build_network_data_uses_nested_fields(monkeypatch):
    case_network = _import_case_network(monkeypatch)

    monkeypatch.setattr(
        case_network,
        "load_past_cases",
        lambda: [
            {
                "id": "a",
                "inputs": {"industry_major": "建設", "industry_sub": "総合工事"},
                "result": {"hantei_score": 70},
                "final_status": "成約",
            },
            {
                "id": "b",
                "inputs": {"industry_major": "建設", "industry_sub": "総合工事"},
                "result": {"hantei_score": 72},
                "final_status": "失注",
            },
        ],
    )

    data = case_network.build_network_data(None)

    assert data["summary"]["total"] == 2
    assert data["summary"]["won"] == 1
    assert data["summary"]["lost"] == 1
    assert data["nodes"][0]["industry_sub"] == "総合工事"
    assert data["edges"]
