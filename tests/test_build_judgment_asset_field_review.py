import json

from scripts import build_judgment_asset_field_review as review


def _canonical_payload():
    return {
        "rules": [
            {
                "id": "rule-helped",
                "status": "active",
                "concept": "business_plan",
                "canonical_statement": "受注根拠と返済原資を確認する。",
                "confidence": 0.91,
                "user_evidence_count": 2,
            },
            {
                "id": "rule-challenged",
                "status": "active",
                "concept": "asset_exit",
                "canonical_statement": "満了後出口を確認する。",
                "confidence": 0.88,
                "user_evidence_count": 1,
            },
            {
                "id": "rule-sleeping",
                "status": "active",
                "concept": "support_specificity",
                "canonical_statement": "支援の直接性を確認する。",
                "confidence": 0.82,
                "user_evidence_count": 0,
            },
            {
                "id": "rule-demoted",
                "status": "demoted",
                "concept": "demo_ops",
                "canonical_statement": "デモ説明を整える。",
                "confidence": 0.8,
                "user_evidence_count": 4,
            },
        ]
    }


def test_build_review_buckets_active_assets_by_field_feedback():
    payload = review.build_review(
        target_date="2026-07-25",
        canonical=_canonical_payload(),
        feedback_rows=[
            {
                "rule_id": "rule-helped",
                "outcome": "helped",
                "case_id": "case-001",
                "note": "稟議条件に使えた",
            },
            {"rule_id": "rule-challenged", "outcome": "challenged"},
            {"rule_id": "rule-demoted", "outcome": "helped"},
            {"rule_id": "missing", "outcome": "used"},
        ],
    )

    assert payload["summary"]["active_rules"] == 3
    assert payload["summary"]["grow"] == 1
    assert payload["summary"]["review"] == 1
    assert payload["summary"]["sleeping"] == 1
    assert payload["summary"]["hold"] == 0
    assert payload["summary"]["unknown_feedback"] == 2
    assert payload["buckets"]["grow"][0]["rule_id"] == "rule-helped"
    assert payload["buckets"]["review"][0]["rule_id"] == "rule-challenged"
    assert payload["buckets"]["sleeping"][0]["rule_id"] == "rule-sleeping"


def test_neutral_or_used_only_feedback_goes_to_hold():
    payload = review.build_review(
        target_date="2026-07-25",
        canonical={
            "rules": [
                {
                    "id": "rule-neutral",
                    "status": "active",
                    "concept": "neutral_rule",
                    "canonical_statement": "確認する。",
                }
            ]
        },
        feedback_rows=[{"rule_id": "rule-neutral", "outcome": "neutral"}],
    )

    assert payload["summary"]["hold"] == 1
    assert payload["buckets"]["hold"][0]["counts"]["neutral"] == 1


def test_markdown_contains_operational_buckets():
    payload = review.build_review(
        target_date="2026-07-25",
        canonical=_canonical_payload(),
        feedback_rows=[{"rule_id": "rule-helped", "outcome": "helped", "note": "使えた"}],
    )

    markdown = review.build_markdown(payload)

    assert "# Judgment Asset Field Review" in markdown
    assert "## 伸ばす" in markdown
    assert "## 見直す" in markdown
    assert "## 眠ってる" in markdown
    assert "business_plan" in markdown


def test_main_writes_json_and_markdown(tmp_path):
    canonical = tmp_path / "canonical.json"
    feedback_jsonl = tmp_path / "feedback.jsonl"
    output_json = tmp_path / "review.json"
    output_md = tmp_path / "review.md"
    canonical.write_text(json.dumps(_canonical_payload(), ensure_ascii=False), encoding="utf-8")
    feedback_jsonl.write_text(
        json.dumps({"rule_id": "rule-helped", "outcome": "helped"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    payload = review.build_review(
        target_date="2026-07-25",
        canonical=review._read_json(canonical),
        feedback_rows=review._read_jsonl(feedback_jsonl),
    )
    review._write_json(output_json, payload)
    output_md.write_text(review.build_markdown(payload), encoding="utf-8")

    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]["grow"] == 1
    assert "Judgment Asset Field Review" in output_md.read_text(encoding="utf-8")
