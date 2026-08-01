from scripts import promote_curator_inbox_candidates as promote


def _curator_report(candidates):
    return {"inbox_candidates": candidates}


def _candidate(candidate_id="cur_abc123", claim="厨房設備は換金性を確認する。", confidence=0.85):
    return {
        "candidate_id": candidate_id,
        "type": "judgment_rule",
        "claim": claim,
        "source": "Projects/tune_lease_55/AI Chat/2026-07-29.md",
        "use_when": "厨房機器のリース判断をするとき",
        "confidence": confidence,
        "status_suggestion": "review",
    }


def test_blocked_when_mana_not_allow():
    result = promote.build_promotion(
        curator_report=_curator_report([_candidate()]),
        mana_report={"status": "hold"},
        decisions=[{"candidate_id": "cur_abc123", "decision": "adopt"}],
        existing_preview=None,
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "mana_not_allow"
    assert result["promoted"] == []


def test_blocked_when_curator_report_missing():
    result = promote.build_promotion(
        curator_report=None,
        mana_report={"status": "allow"},
        decisions=[{"candidate_id": "cur_abc123", "decision": "adopt"}],
        existing_preview=None,
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "curator_report_missing"


def test_adopts_only_explicitly_marked_candidate():
    result = promote.build_promotion(
        curator_report=_curator_report([_candidate("cur_a"), _candidate("cur_b", claim="別の候補")]),
        mana_report={"status": "allow"},
        decisions=[{"candidate_id": "cur_a", "decision": "adopt"}],
        existing_preview=None,
        now="2026-08-01T09:00:00",
    )
    assert result["status"] == "applied"
    assert result["promoted"] == ["cur_a"]
    rules = result["preview"]["canonical_rules"]
    assert len(rules) == 1
    rule = rules[0]
    assert rule["id"] == "curator_cur_a"
    assert rule["status"] == "accepted_preview"
    assert rule["preview"] is True
    assert rule["private"] is False
    assert rule["canonical_statement"] == "厨房設備は換金性を確認する。"
    assert rule["material_type"] == "judgment_rule"
    assert rule["evidence_paths"] == ["Projects/tune_lease_55/AI Chat/2026-07-29.md"]


def test_candidate_not_in_latest_report_is_skipped_not_promoted():
    result = promote.build_promotion(
        curator_report=_curator_report([_candidate("cur_a")]),
        mana_report={"status": "allow"},
        decisions=[{"candidate_id": "cur_stale", "decision": "adopt"}],
        existing_preview=None,
    )
    assert result["status"] == "applied"
    assert result["promoted"] == []
    assert result["skipped"] == [
        {"candidate_id": "cur_stale", "reason": "candidate_not_in_latest_curator_report"}
    ]


def test_last_decision_for_a_candidate_wins():
    result = promote.build_promotion(
        curator_report=_curator_report([_candidate("cur_a")]),
        mana_report={"status": "allow"},
        decisions=[
            {"candidate_id": "cur_a", "decision": "adopt"},
            {"candidate_id": "cur_a", "decision": "reject"},
        ],
        existing_preview=None,
    )
    assert result["promoted"] == []


def test_rerun_is_idempotent_and_preserves_other_existing_rules():
    existing_preview = {
        "schema_version": 1,
        "canonical_rules": [
            {"id": "other_rule", "status": "candidate", "canonical_statement": "既存の別ルール"},
        ],
    }
    decisions = [{"candidate_id": "cur_a", "decision": "adopt"}]
    curator_report = _curator_report([_candidate("cur_a")])
    mana_report = {"status": "allow"}

    first = promote.build_promotion(
        curator_report=curator_report,
        mana_report=mana_report,
        decisions=decisions,
        existing_preview=existing_preview,
    )
    second = promote.build_promotion(
        curator_report=curator_report,
        mana_report=mana_report,
        decisions=decisions,
        existing_preview=first["preview"],
    )

    ids_first = sorted(r["id"] for r in first["preview"]["canonical_rules"])
    ids_second = sorted(r["id"] for r in second["preview"]["canonical_rules"])
    assert ids_first == ids_second == ["curator_cur_a", "other_rule"]
