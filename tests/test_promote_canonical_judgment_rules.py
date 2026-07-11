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
