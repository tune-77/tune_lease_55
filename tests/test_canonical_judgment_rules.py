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


def test_build_canonical_rules_attaches_three_layers():
    materials = [
        {
            "claim": "リース期間は法定耐用年数だけでなく経済的寿命と再販可能性も確認する。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source_role": "user",
            "confidence": 0.8,
            "risk_axis": ["asset_life"],
            "evidence_path": "Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-11.md",
            "private": False,
        },
        {
            "claim": "残価は満了後の再リース先まで見て決める。",
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "source_role": "assistant",
            "confidence": 0.75,
            "risk_axis": ["asset_life"],
            "evidence_path": "Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-12.md",
            "private": False,
        },
    ]

    rules = canonical.build_canonical_rules(materials)
    asset_rule = next(item for item in rules if item["concept"] == "asset_life_and_residual")
    layers = asset_rule["layers"]

    # L0 は関連性判定用の短文。L1/L2 より必ず軽い。
    assert 0 < len(layers["l0_abstract"]) <= canonical._L0_MAX_CHARS + 1
    assert layers["tokens_estimate"]["l0"] < layers["tokens_estimate"]["l1"]
    assert layers["tokens_estimate"]["l1"] < layers["tokens_estimate"]["l2"]

    # L1 は ER 図どおり「適用条件 / 失敗条件 / 質問観点」を持つ。
    assert "適用条件:" in layers["l1_overview"]
    assert "失敗条件:" in layers["l1_overview"]
    assert "質問観点:" in layers["l1_overview"]

    # L2 は一次エビデンスまで含む。
    assert "Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-11.md" in layers["l2_details"]


def test_layers_for_unknown_concept_degrade_without_hints():
    layers = canonical.build_layers(
        {
            "concept": "unmapped_concept",
            "canonical_statement": "出典のない判断はそのまま使わない。",
            "risk_axis": [],
            "sample_claims": [],
            "evidence_paths": [],
        }
    )

    assert layers["l0_abstract"] == "出典のない判断はそのまま使わない。"
    assert "適用条件:" not in layers["l1_overview"]
    assert "リスク軸: n/a" in layers["l1_overview"]
    assert "- (なし)" in layers["l2_details"]


def test_l0_abstract_truncates_long_statement():
    statement = "あ" * 200 + "。"
    l0 = canonical._l0_abstract(statement)

    assert len(l0) <= canonical._L0_MAX_CHARS
    assert l0.endswith("…")


def test_canonical_markdown_tolerates_rules_without_layers():
    rules = [
        {
            "concept": "support_specificity",
            "status": "candidate",
            "evidence_count": 1,
            "canonical_statement": "銀行支援は対象リースへの直接性を確認する。",
            "material_type": "judgment_rule",
            "confidence": 0.8,
            "user_evidence_count": 0,
            "risk_axis": [],
            "sample_claims": [],
            "evidence_paths": [],
        }
    ]

    md = canonical._markdown(rules)

    assert "- L0: n/a" in md
    assert "Layer tokens (L0/L1/L2 total): 0/0/0" in md
