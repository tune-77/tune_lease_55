import json
from pathlib import Path

from scripts import build_memory_engineering_report as mer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_report_tracks_write_promotion_read_and_forgetting(tmp_path):
    candidate_path = tmp_path / "data" / "candidates.jsonl"
    _write_jsonl(
        candidate_path,
        [
            {"status": "needs_human_review", "title": "候補A", "body": "判断に使う候補です"},
            {"status": "rejected", "title": "候補B", "body": "捨てた候補です"},
        ],
    )

    payload = mer.build_report(
        canonical_preview={
            "canonical_rules": [
                {"id": "r1", "status": "accepted_preview"},
                {"id": "r2", "status": "candidate"},
            ]
        },
        canonical_active={"rules": [{"id": "r1", "status": "active"}]},
        memory_index={
            "records": [
                {
                    "id": "mem_used",
                    "content": "使われた判断記憶",
                    "memory_type": "judgment_memory",
                    "status": "active",
                    "source_path": "memory/a.md",
                },
                {
                    "id": "mem_unused",
                    "content": "まだ使われていない判断記憶",
                    "memory_type": "judgment_memory",
                    "status": "active",
                    "created_at": "2026-06-01",
                    "source_path": "memory/b.md",
                },
                {
                    "id": "mem_stale",
                    "content": "古い記憶",
                    "memory_type": "technical_memory",
                    "status": "stale",
                    "source_path": "memory/c.md",
                },
            ]
        },
        memory_usage_rows=[
            {"ts": "2026-08-02T10:00:00", "route": "case_screening", "refs": ["mem_used"]}
        ],
        jsonl_sources=[("test_candidates", candidate_path)],
        contradictions={"candidates": [{"a": {"id": "x"}, "b": {"id": "y"}}]},
        retrieval_graph={
            "summary": {
                "notes": 100,
                "edges": 140,
                "estimated_raw_tokens": 1000,
                "estimated_index_tokens": 100,
                "estimated_token_reduction_ratio": 0.9,
            }
        },
        target_date="2026-08-09",
    )

    summary = payload["summary"]
    assert summary["write_path_records"] == 4
    assert summary["active_canonical_rules"] == 1
    assert summary["latest_preview_promoted_to_active"] == 1
    assert summary["open_human_review_records"] == 1
    assert summary["recent_unique_memory_refs"] == 1
    assert summary["maintenance_status_records"] == 1
    assert summary["contradiction_candidates"] == 1
    assert payload["maintenance_path"]["forgetting_review_sample"][0]["id"] == "mem_unused"
    assert payload["hardware_pressure_proxy"]["estimated_tokens_avoided"] == 900
    assert summary["write_policy_metadata_completion_rate"] == 0.0
    assert payload["write_path"]["write_policy"]["missing_by_field"]["provenance"] == 2
    assert any(item["action"] == "memory_entry_metadata_gate" for item in payload["recommendations"])
    assert any(item["action"] == "contradiction_review" for item in payload["recommendations"])


def test_review_state_reduces_open_review_count(tmp_path):
    candidate_path = tmp_path / "data" / "candidates.jsonl"
    _write_jsonl(
        candidate_path,
        [
            {"id": "a1", "status": "candidate", "title": "候補A"},
            {"id": "a2", "status": "candidate", "title": "候補B"},
        ],
    )

    payload = mer.build_report(
        canonical_preview={"canonical_rules": []},
        canonical_active={"rules": []},
        memory_index={"records": []},
        memory_usage_rows=[],
        jsonl_sources=[("test_candidates", candidate_path)],
        review_state={
            "reviews": {
                "test_candidates__a1": {
                    "status": "rejected",
                    "reviewed_at": "2026-08-09T10:00:00",
                }
            }
        },
        contradictions={},
        retrieval_graph={},
        target_date="2026-08-09",
    )

    assert payload["write_path"]["sources"][0]["by_status"]["rejected"] == 1
    assert payload["summary"]["open_human_review_records"] == 1


def test_write_policy_metadata_completion_counts_complete_rows(tmp_path):
    candidate_path = tmp_path / "data" / "candidates.jsonl"
    _write_jsonl(
        candidate_path,
        [
            {
                "id": "a1",
                "status": "candidate",
                "title": "候補A",
                "importance": 0.8,
                "confidence": 0.9,
                "trust_level": 0.5,
                "provenance": {"source": "test"},
            },
            {"id": "a2", "status": "candidate", "title": "候補B", "importance": 0.8},
        ],
    )

    summary = mer.summarize_jsonl_source("test_candidates", candidate_path)

    assert summary["write_policy_metadata"]["complete_records"] == 1
    assert summary["write_policy_metadata"]["missing_by_field"] == {
        "confidence": 1,
        "provenance": 1,
        "trust_level": 1,
    }


def test_markdown_contains_four_lenses():
    payload = {
        "generated_at": "2026-08-09T10:00:00",
        "mode": "read_only_memory_engineering_observation",
        "guardrail": "no changes",
        "summary": {
            "write_path_records": 1,
            "active_canonical_rules": 1,
            "write_amplification_per_active_rule": 1.0,
            "open_human_review_records": 0,
            "memory_records": 1,
            "recent_memory_usage_events": 1,
            "recent_unique_memory_refs": 1,
            "maintenance_status_records": 0,
            "contradiction_candidates": 0,
            "latest_preview_accepted": 1,
            "latest_preview_promoted_to_active": 1,
            "latest_preview_promotion_rate": 1.0,
        },
        "write_path": {
            "sources": [
                {
                    "name": "source",
                    "records": 1,
                    "open_review_records": 0,
                    "estimated_tokens": 10,
                }
            ],
            "canonical_preview": {"records": 0, "by_status": {}},
        },
        "maintenance_path": {
            "by_status": {"active": 1},
            "by_type": {"judgment_memory": 1},
            "forgetting_review_sample": [],
        },
        "hardware_pressure_proxy": {
            "notes": 10,
            "edges": 20,
            "estimated_raw_tokens": 100,
            "estimated_index_tokens": 20,
            "estimated_token_reduction_ratio": 0.8,
        },
        "recommendations": [],
    }

    markdown = mer.build_markdown(payload)
    assert "## Stanford Lens: Write Cost" in markdown
    assert "## Microsoft Lens: Utility Density" in markdown
    assert "## Anthropic Lens: Control" in markdown
    assert "## Nvidia Lens: Retrieval Pressure" in markdown


def test_write_outputs_creates_json_and_markdown(tmp_path):
    payload = {
        "generated_at": "2026-08-09T10:00:00",
        "mode": "read_only_memory_engineering_observation",
        "guardrail": "no changes",
        "summary": {
            "write_path_records": 0,
            "active_canonical_rules": 0,
            "write_amplification_per_active_rule": None,
            "open_human_review_records": 0,
            "memory_records": 0,
            "recent_memory_usage_events": 0,
            "recent_unique_memory_refs": 0,
            "maintenance_status_records": 0,
            "contradiction_candidates": 0,
            "latest_preview_accepted": 0,
            "latest_preview_promoted_to_active": 0,
            "latest_preview_promotion_rate": None,
        },
        "write_path": {"sources": [], "canonical_preview": {"records": 0, "by_status": {}}},
        "maintenance_path": {"by_status": {}, "by_type": {}, "forgetting_review_sample": []},
        "hardware_pressure_proxy": {},
        "recommendations": [],
    }
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    mer.write_outputs(payload, json_path, md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["mode"] == payload["mode"]
    assert "# Memory Engineering Report" in md_path.read_text(encoding="utf-8")
