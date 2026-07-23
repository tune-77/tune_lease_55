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
        model_paths=(),
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
        model_paths=(),
    )
    out_json = tmp_path / "loop.json"
    out_md = tmp_path / "loop.md"
    write_outputs(report, output_json=out_json, output_md=out_md)

    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["recursive_loop"]["ranked_queue_count"] == 1
    assert "scoring_coefficients" in saved
    assert "Loop Engineering Health" in out_md.read_text(encoding="utf-8")


def test_loop_metrics_flags_prompt_rows_with_zero_pdca_rate(tmp_path):
    from scripts.loop_metrics import build_loop_metrics

    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"applied_count": 0, "needs_review_count": 0, "failed_count": 0}), encoding="utf-8")
    recursive = tmp_path / "recursive.json"
    recursive.write_text(json.dumps({"measurement_summary": {}}, ensure_ascii=False), encoding="utf-8")
    prompt_log = tmp_path / "prompt.jsonl"
    prompt_log.write_text(
        json.dumps(
            {
                "surface": "consultation",
                "pdca_applied": False,
                "response_len": 100,
                "response_diff_from_previous": "",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_loop_metrics(
        latest_report_path=latest,
        recursive_report_path=recursive,
        prompt_log_path=prompt_log,
        model_paths=(),
    )

    assert report["status"] == "attention"
    assert report["prompt_feedback_loop"]["total"] == 1
    assert report["prompt_feedback_loop"]["pdca_rate"] == 0.0
    assert any("PDCA反映率が0%" in item for item in report["recommendations"])


def test_loop_metrics_does_not_flag_healthy_dedup_suppression(tmp_path):
    """noise_rate が高くても churn_rate が0（＝健全な重複排除）なら滞留として警告しない。"""
    from scripts.loop_metrics import build_loop_metrics

    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"applied_count": 1, "needs_review_count": 0, "failed_count": 0}), encoding="utf-8")
    recursive = tmp_path / "recursive.json"
    recursive.write_text(
        json.dumps({"measurement_summary": {"noise_rate": 100.0, "churn_rate": 0.0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    prompt_log = tmp_path / "prompt.jsonl"
    prompt_log.write_text("", encoding="utf-8")

    report = build_loop_metrics(
        latest_report_path=latest,
        recursive_report_path=recursive,
        prompt_log_path=prompt_log,
        model_paths=(),
    )

    assert not any("churn" in item for item in report["recommendations"])
    assert report["recursive_loop"]["measurement_summary"]["churn_rate"] == 0.0


def test_loop_metrics_flags_high_churn(tmp_path):
    """churn_rate が高いときは滞留として警告する。"""
    from scripts.loop_metrics import build_loop_metrics

    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"applied_count": 0, "needs_review_count": 0, "failed_count": 0}), encoding="utf-8")
    recursive = tmp_path / "recursive.json"
    recursive.write_text(
        json.dumps({"measurement_summary": {"noise_rate": 100.0, "churn_rate": 80.0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    prompt_log = tmp_path / "prompt.jsonl"
    prompt_log.write_text("", encoding="utf-8")

    report = build_loop_metrics(
        latest_report_path=latest,
        recursive_report_path=recursive,
        prompt_log_path=prompt_log,
        model_paths=(),
    )

    assert any("churn" in item for item in report["recommendations"])


def test_scoring_coeff_health_flags_all_zero_required_coefficients(tmp_path):
    from scripts.loop_metrics import build_scoring_coeff_health

    overrides = tmp_path / "coeff_overrides.json"
    auto = tmp_path / "coeff_auto.json"
    model = tmp_path / "model.pkl"
    required = {
        "intercept": 0,
        "sales_log": 0,
        "op_profit": 0,
        "ord_profit": 0,
        "net_income": 0,
        "bank_credit_log": 0,
        "lease_credit_log": 0,
    }
    overrides.write_text(json.dumps({"全体_既存先": required}, ensure_ascii=False), encoding="utf-8")
    auto.write_text(
        json.dumps(
            {
                "_auto_weight_borrower": 0.5,
                "_auto_weight_asset": 0.5,
                "_auto_weight_quant": 0.5,
                "_auto_weight_qual": 0.5,
                "_auto_blend_w_main": 0.5,
                "_auto_blend_w_bench": 0.3,
                "_auto_blend_w_ind": 0.2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    model.write_bytes(b"not a pickle")

    health = build_scoring_coeff_health(
        coeff_overrides_path=overrides,
        coeff_auto_path=auto,
        model_paths=(model,),
    )

    assert health["status"] == "attention"
    codes = {issue["code"] for issue in health["issues"]}
    assert "coeff_required_all_zero" in codes
    assert "model_load_failed" in codes
