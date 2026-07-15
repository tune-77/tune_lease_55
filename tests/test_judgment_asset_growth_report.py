import json

from scripts import judgment_asset_growth_report as growth


def test_build_growth_snapshot_uses_current_judgment_asset_signals():
    curator = {
        "materials_count": 12,
        "material_counts": {"judgment_rule": 7, "risk_signal": 5},
        "mana_status": "allow",
        "inbox_candidates": [{"claim": "a"}, {"claim": "b"}],
    }
    mana = {"status": "allow", "findings": []}
    canonical = {
        "rules": [
            {
                "status": "active",
                "concept": "asset_life",
                "domain": "lease_screening",
                "evidence_count": 4,
                "user_evidence_count": 1,
                "confidence": 0.9,
                "risk_axis": ["asset_life", "cash_flow"],
            },
            {
                "status": "active",
                "concept": "support",
                "domain": "lease_screening",
                "evidence_count": 3,
                "user_evidence_count": 0,
                "confidence": 0.82,
                "risk_axis": ["support_specificity"],
            },
        ]
    }

    snapshot = growth.build_growth_snapshot(
        target_date="2026-07-15",
        curator=curator,
        mana=mana,
        canonical=canonical,
        feedback_rows=[
            {
                "rule_id": "business-plan",
                "outcome": "helped",
                "case_id": "case-001",
            }
        ],
    )

    assert snapshot["score_name"] == "Judgment Asset Growth Score"
    assert snapshot["score"] > 0
    assert snapshot["counts"]["active_rules"] == 2
    assert snapshot["counts"]["user_evidence"] == 1
    assert snapshot["components"]["negative_signal"] == 0
    assert "reuse_proxy" in snapshot["components"]
    assert "field_validation" in snapshot["components"]


def test_history_update_is_idempotent_per_date(tmp_path):
    path = tmp_path / "history.jsonl"
    first = {"date": "2026-07-14", "score": 41.0}
    today_v1 = {"date": "2026-07-15", "score": 55.0}
    today_v2 = {"date": "2026-07-15", "score": 56.5}

    growth._write_history(path, first)
    growth._write_history(path, today_v1)
    rows = growth._write_history(path, today_v2)

    assert [row["date"] for row in rows] == ["2026-07-14", "2026-07-15"]
    assert rows[-1]["score"] == 56.5
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_markdown_contains_component_and_trend_graph():
    snapshot = {
        "date": "2026-07-15",
        "score": 56.5,
        "components": {
            "coverage": 60.0,
            "reuse_proxy": 50.0,
            "judgment_change_proxy": 70.0,
            "human_alignment_proxy": 40.0,
            "field_validation": 24.0,
            "negative_signal": 0.0,
        },
        "counts": {
            "materials_count": 51,
            "inbox_candidates": 20,
            "active_rules": 8,
            "risk_axes": 5,
            "concepts": 7,
            "user_evidence": 10,
        },
        "notes": ["測定のみ"],
        "field_feedback": {
            "totals": {"used": 1, "helped": 1, "challenged": 0, "rejected": 0},
            "unused_active_rules": 2,
            "rules": [
                {
                    "rule_id": "rule-1",
                    "used_count": 1,
                    "helped_count": 1,
                    "challenged_count": 0,
                    "last_used_case": "case-001",
                }
            ],
        },
    }
    payload = {
        "mode": "local_measurement_only",
        "guardrail": "no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write",
        "latest": snapshot,
        "history": [{"date": "2026-07-15", "score": 56.5}],
    }

    markdown = growth.build_markdown(payload)

    assert "# Judgment Asset Growth Score" in markdown
    assert "Reuse proxy" in markdown
    assert "Field validation" in markdown
    assert "Top used rules" in markdown
    assert "2026-07-15" in markdown
    assert "█" in markdown


def test_field_feedback_rewards_helped_and_penalizes_challenges():
    rules = [
        {"id": "rule-a", "status": "active"},
        {"id": "rule-b", "status": "active"},
    ]

    helped = growth.summarize_field_feedback(
        [
            {"rule_id": "rule-a", "outcome": "used"},
            {"rule_id": "rule-a", "outcome": "helped", "case_id": "case-001"},
        ],
        rules,
    )
    challenged = growth.summarize_field_feedback(
        [
            {"rule_id": "rule-a", "outcome": "challenged"},
            {"rule_id": "rule-b", "outcome": "rejected"},
        ],
        rules,
    )

    assert helped["score"] > challenged["score"]
    assert helped["totals"]["helped"] == 1
    assert challenged["totals"]["challenged"] == 1
    assert challenged["totals"]["rejected"] == 1
    assert helped["rules"][0]["last_used_case"] == "case-001"


def test_main_writes_latest_json_markdown_and_history(tmp_path):
    curator = tmp_path / "curator.json"
    mana = tmp_path / "mana.json"
    canonical = tmp_path / "canonical.json"
    history = tmp_path / "history.jsonl"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"

    curator.write_text(
        json.dumps(
            {
                "materials_count": 1,
                "material_counts": {"judgment_rule": 1},
                "inbox_candidates": [{"claim": "x"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    mana.write_text(json.dumps({"status": "allow", "findings": []}), encoding="utf-8")
    canonical.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "status": "active",
                        "concept": "asset_life",
                        "domain": "lease_screening",
                        "evidence_count": 2,
                        "user_evidence_count": 1,
                        "confidence": 0.9,
                        "risk_axis": ["asset_life"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshot = growth.build_growth_snapshot(
        target_date="2026-07-15",
        curator=growth._read_json(curator),
        mana=growth._read_json(mana),
        canonical=growth._read_json(canonical),
    )
    rows = growth._write_history(history, snapshot)
    payload = growth.build_payload(snapshot, rows)
    growth._write_json(output_json, payload)
    output_md.write_text(growth.build_markdown(payload), encoding="utf-8")

    assert json.loads(output_json.read_text(encoding="utf-8"))["latest"]["date"] == "2026-07-15"
    assert "Judgment Asset Growth Score" in output_md.read_text(encoding="utf-8")
    assert history.exists()
