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


def test_build_graph_data_renders_explicit_lineage_edges_and_metadata():
    payload = graph.build_graph_data(
        canonical={
            "rules": [
                {
                    "id": "rule-parent",
                    "status": "active",
                    "concept": "asset_life",
                    "canonical_statement": "機械の経済的寿命を見る。",
                },
                {
                    "id": "rule-child",
                    "status": "active",
                    "concept": "asset_life_and_residual",
                    "canonical_statement": "経済的寿命に加えて残価と出口を見る。",
                    "parent_ids": ["rule-parent"],
                    "derivation_reason": "実案件フィードバックで残価観点を追加",
                },
            ]
        }
    )

    by_id = {node["id"]: node for node in payload["nodes"]}
    lineage_edges = [edge for edge in payload["edges"] if edge["type"] == "lineage"]

    assert by_id["rule:rule-child"]["parent_ids"] == ["rule-parent"]
    assert by_id["rule:rule-child"]["derivation_reason"] == "実案件フィードバックで残価観点を追加"
    assert by_id["rule:rule-child"]["lineage_depth"] == 1
    assert lineage_edges == [
        {
            "source": "rule:rule-parent",
            "target": "rule:rule-child",
            "type": "lineage",
            "label": "実案件フィードバックで残価観点を追加",
            "weight": 2.4,
            "color": "#0f766e",
        }
    ]
    assert payload["summary"]["lineage_edges"] == 1
    assert payload["summary"]["lineage_roots"] == 1
    assert payload["summary"]["lineage_derived"] == 1


def test_build_graph_data_infers_same_concept_lineage_for_legacy_assets():
    payload = graph.build_graph_data(
        canonical={
            "rules": [
                {
                    "id": "rule-early",
                    "status": "active",
                    "concept": "demo_renewal_asset",
                    "canonical_statement": "公開デモ案件は更新資産の妥当性を見る。",
                    "created_at": "2026-07-01T00:00:00",
                },
                {
                    "id": "rule-later",
                    "status": "active",
                    "concept": "demo_renewal_asset",
                    "canonical_statement": "更新資産は既存設備との置換関係も見る。",
                    "created_at": "2026-07-20T00:00:00",
                },
            ]
        }
    )

    by_id = {node["id"]: node for node in payload["nodes"]}
    lineage_edges = [edge for edge in payload["edges"] if edge["type"] == "lineage"]

    assert by_id["rule:rule-later"]["parent_ids"] == ["rule-early"]
    assert by_id["rule:rule-later"]["derivation_reason"] == "同じ判断テーマから派生"
    assert lineage_edges[0]["source"] == "rule:rule-early"
    assert lineage_edges[0]["target"] == "rule:rule-later"


def test_build_graph_data_preserves_missing_explicit_parent_ids_without_edge():
    payload = graph.build_graph_data(
        canonical={
            "rules": [
                {
                    "id": "rule-child",
                    "status": "active",
                    "concept": "asset_life",
                    "canonical_statement": "親が未収録でも系統メタデータは保持する。",
                    "parent_ids": ["external-parent"],
                    "derivation_reason": "外部ノートから派生",
                }
            ]
        }
    )

    by_id = {node["id"]: node for node in payload["nodes"]}
    lineage_edges = [edge for edge in payload["edges"] if edge["type"] == "lineage"]

    assert by_id["rule:rule-child"]["parent_ids"] == ["external-parent"]
    assert by_id["rule:rule-child"]["derivation_reason"] == "外部ノートから派生"
    assert lineage_edges == []
    assert payload["summary"]["lineage_derived"] == 1


def test_build_html_is_offline_and_embeds_graph_payload():
    payload = graph.build_graph_data(
        canonical={"rules": [{"id": "rule-1", "status": "active", "concept": "asset_life"}]}
    )

    html = graph.build_html(payload)

    assert "<!doctype html>" in html
    assert "判断資産系統樹" in html
    assert "親子関係" in html
    assert "const graph =" in html
    assert "https://" not in html
    assert "rule:rule-1" in html
