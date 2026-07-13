import json

from scripts.run_shion_memory_asset_experiment import run_experiment


def test_memory_asset_experiment_converts_memory_to_judgment(tmp_path):
    timeline = tmp_path / "timeline.json"
    timeline.write_text(
        json.dumps(
            {
                "memory_layers": {
                    "mid_term": {
                        "items": [
                            "Q_risk and RAG should separate credit concern from competition or contract risk."
                        ]
                    },
                    "long_term": {
                        "promotion_candidates": [
                            "Q_risk should be used as a discovery signal, not an automatic score deduction.",
                            "事前判断は人間の仮説、結果登録は実際の結果、差分は判断基準の精度検証。",
                        ]
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = run_experiment(timeline)

    assert payload["summary"]["passes_minimum_bar"] is True
    assert payload["summary"]["case_count"] == 3
    for item in payload["cases"]:
        aware = item["memory_aware"]
        assert aware["uses_explicit_memory_bridge"] is False
        assert len(aware["questions"]) <= 3
        assert aware["risk_origin"]
        assert "前回" not in aware["opening"]


def test_memory_asset_experiment_separates_case_risk_origins(tmp_path):
    timeline = tmp_path / "timeline.json"
    timeline.write_text('{"memory_layers": {}}', encoding="utf-8")

    payload = run_experiment(timeline)
    by_id = {item["case"]["case_id"]: item["memory_aware"]["risk_origin"] for item in payload["cases"]}

    assert by_id["food_new_store"] == "credit_and_contract_recovery"
    assert by_id["logistics_route_expansion"] == "competition_or_contract"
    assert by_id["precision_machine_capacity"] == "repayment_source_and_asset_purpose"
