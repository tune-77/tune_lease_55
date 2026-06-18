from __future__ import annotations

import json


def test_loop_metrics_handles_missing_recursive_report(tmp_path):
    from scripts.loop_metrics import build_loop_metrics

    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "applied_count": 1,
                "needs_review_count": 2,
                "failed_count": 0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    prompt_log = tmp_path / "prompt.jsonl"
    prompt_log.write_text(
        json.dumps(
            {
                "surface": "consultation",
                "pdca_applied": True,
                "response_len": 100,
                "response_diff_from_previous": "+changed",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_loop_metrics(
        latest_report_path=latest,
        recursive_report_path=tmp_path / "missing_recursive.json",
        prompt_log_path=prompt_log,
    )

    assert report["status"] == "warn"
    assert report["sources"]["recursive_report"]["available"] is False
    assert report["health"]["source_coverage_rate"] == 66.7
    assert report["improvement_loop"]["needs_review_count"] == 2
    assert report["prompt_feedback_loop"]["pdca_rate"] == 100.0
    assert any("recursive_self_improvement_latest.json" in item for item in report["recommendations"])


def test_loop_metrics_writes_json_and_markdown(tmp_path):
    from scripts.loop_metrics import build_loop_metrics, write_outputs

    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "applied_count": 2,
                "needs_review_count": 1,
                "failed_count": 0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    recursive = tmp_path / "recursive.json"
    recursive.write_text(
        json.dumps(
            {
                "canonical_candidate_count": 4,
                "ranked_queue_count": 1,
                "suppressed_count": 1,
                "measurement_summary": {
                    "pdca_rate": 50.0,
                    "response_changed_rate": 25.0,
                    "repeat_issue_rate": 10.0,
                    "reuse_rate": 20.0,
                    "noise_rate": 25.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    prompt_log = tmp_path / "prompt.jsonl"
    prompt_log.write_text("", encoding="utf-8")

    report = build_loop_metrics(
        latest_report_path=latest,
        recursive_report_path=recursive,
        prompt_log_path=prompt_log,
    )
    out_json = tmp_path / "loop.json"
    out_md = tmp_path / "loop.md"
    write_outputs(report, output_json=out_json, output_md=out_md)

    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["recursive_loop"]["ranked_queue_count"] == 1
    assert "Loop Engineering Health" in out_md.read_text(encoding="utf-8")

