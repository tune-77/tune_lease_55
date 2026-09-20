import json
from pathlib import Path

import scripts.typesafe_rag_usage_report as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_summarize_computes_exclusion_rate_and_fallback_errors(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "status": "applied",
                "candidate_count": 8,
                "accepted_count": 3,
                "excluded_count": 5,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
            {
                "status": "applied",
                "candidate_count": 4,
                "accepted_count": 4,
                "excluded_count": 0,
                "usage": {"input_tokens": 50, "output_tokens": 10},
            },
            {"status": "fallback", "error_type": "TypeSafeRagError"},
        ],
    )

    entries = report.read_entries(log_path)
    summary = report.summarize(entries)

    assert summary["samples"] == 3
    assert summary["applied_count"] == 2
    assert summary["fallback_count"] == 1
    assert summary["candidate_total"] == 12
    assert summary["accepted_total"] == 7
    assert summary["excluded_total"] == 5
    assert summary["exclusion_rate"] == round(5 / 12, 3)
    assert summary["input_tokens_total"] == 150
    assert summary["output_tokens_total"] == 30
    assert summary["fallback_errors"] == [("TypeSafeRagError", 1)]


def test_summarize_handles_missing_log_file(tmp_path):
    entries = report.read_entries(tmp_path / "does_not_exist.jsonl")
    summary = report.summarize(entries)

    assert summary["samples"] == 0
    assert summary["exclusion_rate"] == 0.0
    assert summary["fallback_errors"] == []


def test_render_includes_key_numbers():
    summary = report.summarize(
        [
            {
                "status": "applied",
                "candidate_count": 10,
                "accepted_count": 6,
                "excluded_count": 4,
                "usage": {},
            }
        ]
    )

    output = report.render(summary)

    assert "Applied: 1 / Fallback: 0" in output
    assert "除外率 40.0%" in output
