from scripts import build_judgment_asset_graph as graph


def test_build_graph_data_links_rules_to_axes_domains_evidence_and_cases():
    canonical = {
        "rules": [
            {
                "id": "rule-1",
                "status": "active",
                "concept": "asset_life",
                "canonical_statement": "期間と出口を確認する。",
                "evidence_count": 3,
                "user_evidence_count": 1,
                "confidence": 0.9,
                "risk_axis": ["asset_life", "cash_flow"],
                "domains": ["lease_screening"],
                "evidence_paths": ["Projects/tune_lease_55/AI Chat/2026-07-01.md"],
            }
        ]
    }

    payload = graph.build_graph_data(
        canonical=canonical,
        feedback_rows=[
            {
                "rule_id": "rule-1",
                "outcome": "helped",
                "case_id": "case-001",
                "note": "稟議コメントに使えた",
                "used_at": "2026-07-19T12:00:00",
            }
        ],
        growth_evaluation={"judgment": {"label": "育った", "score": 78.0}},
    )

    node_ids = {node["id"] for node in payload["nodes"]}
    edge_types = {edge["type"] for edge in payload["edges"]}

    assert "rule:rule-1" in node_ids
    assert "risk:asset_life" in node_ids
    assert "risk:cash_flow" in node_ids
    assert "domain:lease_screening" in node_ids
    assert "case:case-001" in node_ids
    assert {"risk_axis", "domain", "evidence", "helped"}.issubset(edge_types)
    assert payload["summary"]["rules"] == 1
    assert payload["summary"]["cases"] == 1
    assert payload["summary"]["growth_label"] == "育った"


def test_inactive_rules_are_not_rendered():
    payload = graph.build_graph_data(
        canonical={
            "rules": [
                {"id": "active", "status": "active", "concept": "keep"},
                {"id": "draft", "status": "candidate", "concept": "skip"},
            ]
        }
    )

    node_ids = {node["id"] for node in payload["nodes"]}

    assert "rule:active" in node_ids
    assert "rule:draft" not in node_ids


def test_build_html_is_offline_and_embeds_graph_payload():
    payload = graph.build_graph_data(
        canonical={"rules": [{"id": "rule-1", "status": "active", "concept": "asset_life"}]}
    )

    html = graph.build_html(payload)

    assert "<!doctype html>" in html
    assert "Judgment Asset Graph" in html
    assert "const graph =" in html
    assert "https://" not in html
    assert "rule:rule-1" in html
