import json

from scripts import build_canonical_judgment_rules as canonical


def test_build_canonical_rules_groups_similar_materials():
    materials = [
        {
            "claim": "リース期間は法定耐用年数だけでなく経済的寿命と再販可能性も確認する。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source_role": "assistant",
            "confidence": 0.82,
            "risk_axis": ["asset_life"],
            "evidence_path": "Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-10.md",
            "private": False,
        },
        {
            "claim": "ラーメン屋の厨房機器はリース期間5年が多い。覚えておいて。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source_role": "user",
            "confidence": 0.78,
            "risk_axis": ["asset_life", "industry_risk"],
            "evidence_path": "Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-11.md",
            "private": False,
        },
        {
            "claim": "銀行支援は本件リースへの直接支援か確認します。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source_role": "assistant",
            "confidence": 0.74,
            "risk_axis": ["support_specificity"],
            "evidence_path": "Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-12.md",
            "private": False,
        },
        {
            "claim": "Private Reflection raw body should not be included.",
            "material_type": "judgment_rule",
            "domain": "ai_agent_ops",
            "source_role": "assistant",
            "confidence": 0.9,
            "risk_axis": [],
            "evidence_path": "Private Reflection/2026-07-12.md",
            "private": True,
        },
    ]

    rules = canonical.build_canonical_rules(materials)
    concepts = {item["concept"] for item in rules}
    asset_rule = next(item for item in rules if item["concept"] == "asset_life_and_residual")

    assert "asset_life_and_residual" in concepts
    assert "support_specificity" in concepts
    assert asset_rule["evidence_count"] == 2
    assert asset_rule["user_evidence_count"] == 1
    assert asset_rule["status"] == "accepted_preview"
    assert all(item["private"] is False for item in rules)
    assert "Private Reflection" not in json.dumps(rules, ensure_ascii=False)


def test_build_canonical_rules_keeps_vertex_only_group_as_candidate():
    materials = [
        {
            "claim": f"リース期間は法定耐用年数だけでなく経済的寿命と再販可能性も確認する（{idx}）。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source": "vertex_distilled_review",
            "source_role": "vertex_review",
            "confidence": 0.7,
            "risk_axis": ["asset_life"],
            "evidence_path": "Research/Vertex Distilled/2026-08-08-sample.md",
            "private": False,
        }
        for idx in range(3)
    ]

    rules = canonical.build_canonical_rules(materials)
    asset_rule = next(item for item in rules if item["concept"] == "asset_life_and_residual")

    assert asset_rule["evidence_count"] == 3
    assert asset_rule["user_evidence_count"] == 0
    assert asset_rule["status"] == "candidate"


def test_build_canonical_rules_mixed_source_group_still_auto_promotes():
    materials = [
        {
            "claim": f"リース期間は法定耐用年数だけでなく経済的寿命と再販可能性も確認する（{idx}）。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source": "vertex_distilled_review",
            "source_role": "vertex_review",
            "confidence": 0.7,
            "risk_axis": ["asset_life"],
            "evidence_path": "Research/Vertex Distilled/2026-08-08-sample.md",
            "private": False,
        }
        for idx in range(2)
    ]
    materials.append(
        {
            "claim": "リース期間は法定耐用年数だけでなく経済的寿命と再販可能性も確認する（会話由来）。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source": "Projects/tune_lease_55/Lease Intelligence/Dialogue",
            "source_role": "assistant",
            "confidence": 0.8,
            "risk_axis": ["asset_life"],
            "evidence_path": "Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-11.md",
            "private": False,
        }
    )

    rules = canonical.build_canonical_rules(materials)
    asset_rule = next(item for item in rules if item["concept"] == "asset_life_and_residual")

    assert asset_rule["evidence_count"] == 3
    assert asset_rule["status"] == "accepted_preview"


def test_canonical_markdown_declares_preview_only():
    rules = [
        {
            "concept": "support_specificity",
            "status": "candidate",
            "evidence_count": 1,
            "canonical_statement": "銀行支援は対象リースへの直接性を確認する。",
            "material_type": "judgment_rule",
            "confidence": 0.8,
            "user_evidence_count": 0,
            "risk_axis": ["support_specificity"],
            "sample_claims": ["銀行支援は本件リースへの直接支援か確認します。"],
            "evidence_paths": ["Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-12.md"],
        }
    ]

    md = canonical._markdown(rules)

    assert "Preview only" in md
    assert "Not connected to RAG" in md
    assert "compressed into representative rules" in md
