from __future__ import annotations

import json


def test_recursive_self_improvement_builds_queue_and_suppresses_duplicates(tmp_path, monkeypatch):
    from scripts import recursive_self_improvement as rsi
    import pipeline_ledger

    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(pipeline_ledger, "LEDGER_PATH", ledger_path)
    pipeline_ledger.record(
        "answer_truncation",
        "applied",
        "回答途切れの改善",
        canonical_key="answer_truncation",
    )

    report = {
        "date": "2026-06-12",
        "generated_at": "2026-06-12 12:00:00",
        "applied": [
            {
                "id": "REV-001",
                "title": "回答途切れの改善",
                "description": "回答が途中で切れる。",
                "detail": "回答が途中で切れる。",
            }
        ],
        "needs_review": [
            {
                "id": "REV-002",
                "title": "回答途切れの改善",
                "description": "回答が途中で切れる。",
                "detail": "回答が途中で切れる。",
            },
            {
                "id": "REV-003",
                "title": "送信ボタンの文言を「保存」に変更する",
                "description": "ボタン文言を短く統一する。",
                "target_module": "frontend/src/app/page.tsx",
            },
            {
                "id": "REV-004",
                "title": "スコアリングの閾値を32に変更する",
                "description": "評価閾値を見直す。",
                "target_module": "scoring_core.py",
            },
        ],
    }
    prompt_rows = [
        {
            "surface": "consultation",
            "pdca_applied": True,
            "response_len": 100,
            "prompt_base_len": 90,
            "prompt_final_len": 120,
            "prompt_diff": "+rule",
            "response_diff_from_previous": "+more detail",
        },
        {
            "surface": "consultation",
            "pdca_applied": False,
            "response_len": 80,
            "prompt_base_len": 90,
            "prompt_final_len": 90,
            "prompt_diff": "",
            "response_diff_from_previous": "",
        },
    ]

    bundle = rsi.build_recursive_self_improvement(
        report,
        prompt_feedback_log=prompt_rows,
        workspace_root=tmp_path,
    )

    assert bundle["canonical_candidate_count"] == 3
    assert bundle["ranked_queue_count"] == 1
    assert bundle["suppressed_count"] == 1
    assert bundle["measurement_summary"]["pdca_rate"] == 50.0
    assert bundle["measurement_summary"]["response_changed_rate"] == 50.0
    assert bundle["measurement_summary"]["repeat_issue_rate"] > 0
    assert bundle["measurement_summary"]["reuse_rate"] > 0
    assert bundle["suppressions"][0]["reason"].startswith("ledger=applied")
    assert bundle["ranked_queue"][0]["title"] == "送信ボタンの文言を「保存」に変更する"


def test_recursive_self_improvement_writes_outputs_and_augments_latest(tmp_path):
    from scripts import recursive_self_improvement as rsi

    bundle = {
        "generated_at": "2026-06-12T12:00:00",
        "date": "2026-06-12",
        "source_report": "reports/latest.json",
        "canonical_candidate_count": 1,
        "ranked_queue_count": 1,
        "suppressed_count": 0,
        "measurement_summary": {
            "pdca_rate": 100.0,
            "response_changed_rate": 100.0,
            "repeat_issue_rate": 0.0,
            "reuse_rate": 100.0,
            "noise_rate": 0.0,
            "prompt_total": 1,
            "prompt_previous_diff_count": 1,
        },
        "ranked_queue": [
            {
                "id": "REV-003",
                "title": "送信ボタンの文言を「保存」に変更する",
                "recommended_order": 1,
                "priority_score": 42,
            }
        ],
        "suppressions": [],
        "ledger_events": [],
        "grouped_improvements": [],
        "canonical_candidates": [],
        "prompt_feedback_summary": {},
        "input_counts": {},
    }
    out_json = tmp_path / "recursive.json"
    out_md = tmp_path / "recursive.md"
    latest_json = tmp_path / "recursive_latest.json"
    latest_md = tmp_path / "recursive_latest.md"
    augment = tmp_path / "latest.json"
    augment.write_text(json.dumps({"date": "2026-06-12"}, ensure_ascii=False), encoding="utf-8")

    rsi.write_recursive_outputs(
        bundle,
        output_json=out_json,
        output_md=out_md,
        latest_json=latest_json,
        latest_md=latest_md,
        augment_report=augment,
    )

    assert out_json.exists()
    assert out_md.exists()
    assert latest_json.exists()
    assert latest_md.exists()
    augmented = json.loads(augment.read_text(encoding="utf-8"))
    assert "recursive_self_improvement" in augmented
    assert augmented["recursive_self_improvement"]["canonical_candidate_count"] == 1
