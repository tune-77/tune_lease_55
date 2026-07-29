from pathlib import Path

from scripts import build_shion_architecture_layer_audit as audit


PRESENT_PATHS = {
    "frontend/src/app/shion-eval-health/page.tsx",
    "frontend/src/app/judgment-asset-graph/page.tsx",
    "scripts/build_judgment_asset_graph.py",
    "scripts/promote_canonical_judgment_rules.py",
    "scripts/record_judgment_asset_feedback.py",
    "scripts/preflight_pr_guard.py",
    "frontend/src/app/screening/page.tsx",
    "frontend/src/app/lease-intelligence/page.tsx",
    "frontend/src/app/improvement-log/page.tsx",
}


def test_build_audit_maps_shion_to_three_layers_without_runtime_side_effects():
    payload = audit.build_audit(
        target_date="2026-07-29",
        canonical={
            "rules": [
                {"id": "root", "status": "active", "concept": "asset_life"},
                {"id": "child", "status": "active", "concept": "asset_life", "parent_ids": ["root"]},
                {"id": "legacy-a", "status": "active", "concept": "demo_renewal_asset"},
                {"id": "legacy-b", "status": "active", "concept": "demo_renewal_asset"},
            ]
        },
        field_review={"summary": {"active_rules": 2, "sleeping": 2}},
        growth={
            "judgment": {"label": "要点検"},
            "dimensions": {"field_validation": {"score": 0}},
        },
        memory_effectiveness={"summary": {"used": 4}},
        loop_engineering={"status": "attention"},
        project_root=Path("/repo"),
        present_paths=PRESENT_PATHS,
    )

    assert payload["mode"] == "read_only_architecture_audit"
    assert payload["guardrail"] == audit.GUARDRAIL
    assert payload["overall_status"] == "attention"
    assert payload["summary"]["active_judgment_assets"] == 4
    assert payload["summary"]["lineage_assets"] == 2
    assert payload["summary"]["field_feedback_sleeping"] == 2
    assert [layer["id"] for layer in payload["layers"]] == ["harness", "agents", "improver"]
    assert any("実案件フィードバック未記録" in risk for risk in payload["layers"][2]["risks"])
    assert any("既存の回答生成" in guardrail for guardrail in payload["guardrails"])


def test_build_markdown_is_japanese_and_marks_safe_sequence():
    payload = audit.build_audit(
        target_date="2026-07-29",
        canonical={"rules": [{"id": "rule", "status": "active", "concept": "cash_flow"}]},
        field_review={"summary": {"active_rules": 1, "sleeping": 0}},
        growth={"judgment": {"label": "育った"}, "dimensions": {"field_validation": {"score": 20}}},
        memory_effectiveness={"summary": {"used": 1}},
        loop_engineering={"status": "ok"},
        project_root=Path("/repo"),
        present_paths=PRESENT_PATHS,
    )

    md = audit.build_markdown(payload)

    assert "# 紫苑 アーキテクチャ層監査" in md
    assert "harness: 型と安全ゲート" in md
    assert "agents: 判断を作るAI" in md
    assert "improver: 判断資産を育てる層" in md
    assert "既存の回答生成、RAG、スコアリング、日次改善パイプラインには接続しません" in md
    assert "Safe Sequence" in md
