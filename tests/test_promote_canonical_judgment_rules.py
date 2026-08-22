from scripts import promote_canonical_judgment_rules as promote


def test_promote_only_accepted_preview_rules_and_updates_existing():
    preview_rules = [
        {
            "id": "rule_asset",
            "status": "accepted_preview",
            "preview": True,
            "private": False,
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "concept": "asset_life_and_residual",
            "canonical_statement": "リース期間は経済的寿命と換金性も確認する。",
            "evidence_count": 3,
            "user_evidence_count": 1,
            "confidence": 0.9,
            "risk_axis": ["asset_life"],
            "sample_claims": ["sample"],
            "evidence_paths": ["path/a.md"],
        },
        {
            "id": "rule_candidate",
            "status": "candidate",
            "preview": True,
            "private": False,
            "canonical_statement": "まだ候補。",
        },
        {
            "id": "rule_private",
            "status": "accepted_preview",
            "preview": True,
            "private": True,
            "canonical_statement": "private。",
        },
    ]
    existing = {
        "rules": [
            {
                "id": "rule_asset",
                "status": "active",
                "domain": "lease_screening",
                "concept": "asset_life_and_residual",
                "created_at": "2026-07-01T00:00:00",
                "canonical_statement": "リース期間は経済的寿命と換金性も確認する。",
            },
            {
                "id": "rule_keep",
                "status": "active",
                "created_at": "2026-07-01T00:00:00",
                "canonical_statement": "既存ルールはpreviewに無くても残す。",
            },
        ]
    }

    store = promote.promote_rules(preview_rules, existing, now="2026-07-12T12:00:00")
    by_id = {rule["id"]: rule for rule in store["rules"]}

    assert store["summary"]["active_rules"] == 2
    assert store["summary"]["promoted"] == 0
    assert store["summary"]["updated"] == 1
    assert store["summary"]["skipped"] == 2
    assert by_id["rule_asset"]["canonical_statement"] == "リース期間は経済的寿命と換金性も確認する。"
    assert by_id["rule_asset"]["created_at"] == "2026-07-01T00:00:00"
    assert by_id["rule_asset"]["updated_at"] == "2026-07-12T12:00:00"
    assert "rule_keep" in by_id
    assert "rule_candidate" not in by_id
    assert "rule_private" not in by_id


def test_promote_merges_same_semantic_rule_across_material_types():
    preview_rules = [
        {
            "id": "rule_support_judgment",
            "status": "accepted_preview",
            "preview": True,
            "private": False,
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "concept": "support_specificity",
            "canonical_statement": "銀行支援は対象リースへの直接性を確認する。",
            "evidence_count": 4,
            "user_evidence_count": 0,
            "confidence": 0.82,
            "risk_axis": ["support_specificity"],
            "sample_claims": ["直接支援か確認する。"],
            "evidence_paths": ["path/a.md"],
        },
        {
            "id": "rule_support_risk",
            "status": "accepted_preview",
            "preview": True,
            "private": False,
            "material_type": "risk_signal",
            "domain": "lease_screening",
            "concept": "support_specificity",
            "canonical_statement": "銀行支援は対象リースへの直接性を確認する。",
            "evidence_count": 3,
            "user_evidence_count": 1,
            "confidence": 0.84,
            "risk_axis": ["cash_flow"],
            "sample_claims": ["補助金は過信しない。"],
            "evidence_paths": ["path/b.md"],
        },
    ]

    store = promote.promote_rules(preview_rules, {"rules": []}, now="2026-07-12T12:00:00")

    assert store["summary"]["active_rules"] == 1
    rule = store["rules"][0]
    assert rule["evidence_count"] == 7
    assert rule["user_evidence_count"] == 1
    assert rule["confidence"] == 0.84
    assert rule["material_types"] == ["judgment_rule", "risk_signal"]
    assert rule["evidence_paths"] == ["path/a.md", "path/b.md"]


def test_promote_preserves_lineage_parent_ids_and_derivation_reason():
    preview_rules = [
        {
            "id": "rule_child",
            "status": "accepted_preview",
            "preview": True,
            "private": False,
            "material_type": "judgment_rule",
            "domain": "lease_screening",
            "concept": "asset_life_and_residual",
            "canonical_statement": "経済的寿命に加えて残価と出口を見る。",
            "parent_ids": ["rule_parent"],
            "derivation_reason": "実案件フィードバックで残価観点を追加",
            "evidence_count": 2,
        }
    ]

    store = promote.promote_rules(preview_rules, {"rules": []}, now="2026-07-12T12:00:00")

    rule = store["rules"][0]
    assert rule["parent_ids"] == ["rule_parent"]
    assert rule["derivation_reason"] == "実案件フィードバックで残価観点を追加"


def test_promote_keeps_existing_lineage_when_preview_updates_same_rule():
    preview_rules = [
        {
            "id": "rule_child",
            "status": "accepted_preview",
            "preview": True,
            "private": False,
            "concept": "asset_life_and_residual",
            "canonical_statement": "経済的寿命に加えて残価と出口を見る。",
            "evidence_count": 4,
        }
    ]
    existing = {
        "rules": [
            {
                "id": "rule_child",
                "status": "active",
                "concept": "asset_life_and_residual",
                "canonical_statement": "経済的寿命に加えて残価と出口を見る。",
                "parent_ids": ["rule_parent"],
                "derivation_reason": "過去の審査メモから派生",
                "created_at": "2026-07-01T00:00:00",
            }
        ]
    }

    store = promote.promote_rules(preview_rules, existing, now="2026-07-12T12:00:00")

    rule = store["rules"][0]
    assert rule["parent_ids"] == ["rule_parent"]
    assert rule["derivation_reason"] == "過去の審査メモから派生"


def test_promoted_markdown_declares_active_store_not_obsidian():
    store = {
        "summary": {"active_rules": 1, "promoted": 1, "updated": 0, "skipped": 0},
        "rules": [
            {
                "concept": "support_specificity",
                "evidence_count": 3,
                "user_evidence_count": 1,
                "canonical_statement": "銀行支援は直接性を確認する。",
                "material_type": "judgment_rule",
                "confidence": 0.88,
                "risk_axis": ["support_specificity"],
            }
        ],
    }

    md = promote._markdown(store)

    assert "Only accepted_preview rules are promoted" in md
    assert "Obsidian is not modified" in md
    assert "judgment_memory" in md
    assert "親判断:" in md
    assert "派生理由:" in md


def _layered_preview_rule(*, material_type: str, sample_claim: str, evidence_path: str) -> dict:
    return {
        "id": f"rule_asset_{material_type}",
        "status": "accepted_preview",
        "preview": True,
        "private": False,
        "material_type": material_type,
        "domain": "lease_screening",
        "concept": "asset_life_and_residual",
        "canonical_statement": "リース期間・残価判断では経済的寿命と出口を合わせて確認する。",
        "evidence_count": 2,
        "user_evidence_count": 1,
        "confidence": 0.86,
        "risk_axis": ["asset_life"],
        "sample_claims": [sample_claim],
        "evidence_paths": [evidence_path],
        "layers": {
            "l0_abstract": "リース期間・残価判断では経済的寿命と出口を合わせて確認する。",
            "l1_overview": "判断: リース期間・残価判断では経済的寿命と出口を合わせて確認する。\n適用条件: リース期間・残価・満了後の出口を決める時",
            "l2_details": f"代表クレーム:\n- {sample_claim}\n一次エビデンス:\n- {evidence_path}",
            "tokens_estimate": {"l0": 12, "l1": 30, "l2": 40},
        },
    }


def test_promotion_carries_layers_into_active_store():
    preview_rules = [
        _layered_preview_rule(
            material_type="judgment_rule",
            sample_claim="経済的寿命は法定耐用年数より短いことがある。",
            evidence_path="Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-11.md",
        )
    ]

    store = promote.promote_rules(preview_rules, {"schema_version": 1, "rules": []})
    rule = next(item for item in store["rules"] if item["concept"] == "asset_life_and_residual")

    assert rule["layers"]["l0_abstract"].startswith("リース期間・残価判断では")
    assert "適用条件:" in rule["layers"]["l1_overview"]
    assert rule["layers"]["tokens_estimate"]["l0"] > 0


def test_promotion_rebuilds_l2_from_merged_evidence():
    """L2 は sample_claims / evidence_paths の描画なので、マージ後の実データと一致させる。"""
    preview_rules = [
        _layered_preview_rule(
            material_type="judgment_rule",
            sample_claim="経済的寿命は法定耐用年数より短いことがある。",
            evidence_path="Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-11.md",
        ),
        _layered_preview_rule(
            material_type="risk_signal",
            sample_claim="満了後の再販先が無い物件は残価を寝かせない。",
            evidence_path="Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-12.md",
        ),
    ]

    store = promote.promote_rules(preview_rules, {"schema_version": 1, "rules": []})
    rule = next(item for item in store["rules"] if item["concept"] == "asset_life_and_residual")

    # 同じ concept + canonical_statement なので1本にマージされる
    assert rule["material_types"] == ["judgment_rule", "risk_signal"]
    for claim in rule["sample_claims"]:
        assert claim in rule["layers"]["l2_details"]
    for path in rule["evidence_paths"]:
        assert path in rule["layers"]["l2_details"]


def test_promotion_without_preview_layers_still_builds_l2():
    preview_rule = _layered_preview_rule(
        material_type="judgment_rule",
        sample_claim="経済的寿命は法定耐用年数より短いことがある。",
        evidence_path="Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-11.md",
    )
    preview_rule.pop("layers")

    store = promote.promote_rules([preview_rule], {"schema_version": 1, "rules": []})
    rule = next(item for item in store["rules"] if item["concept"] == "asset_life_and_residual")

    assert rule["layers"]["l0_abstract"] == ""
    assert "経済的寿命は法定耐用年数より短いことがある。" in rule["layers"]["l2_details"]
