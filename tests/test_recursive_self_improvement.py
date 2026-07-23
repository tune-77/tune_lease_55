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


def test_recursive_self_improvement_suppresses_deleted_items(tmp_path, monkeypatch):
    from scripts import recursive_self_improvement as rsi
    import pipeline_ledger

    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(pipeline_ledger, "LEDGER_PATH", ledger_path)

    report = {
        "date": "2026-07-19",
        "needs_review": [
            {
                "id": "REV-101",
                "title": "削除済み候補",
                "description": "ユーザーが改善ログで削除した項目。",
                "target_module": "frontend/src/app/page.tsx",
            }
        ],
    }
    deleted_key = rsi.canonical_key("削除済み候補", "ユーザーが改善ログで削除した項目。")
    pipeline_ledger.record(
        deleted_key,
        "deleted",
        "削除済み候補",
        reason="UI経由で改善ログから削除",
        canonical_key=deleted_key,
    )

    bundle = rsi.build_recursive_self_improvement(
        report,
        prompt_feedback_log=[],
        workspace_root=tmp_path,
    )

    assert bundle["ranked_queue_count"] == 0
    assert bundle["suppressed_count"] == 1
    assert bundle["suppressions"][0]["canonical_key"] == deleted_key
    assert bundle["suppressions"][0]["reason"].startswith("ledger=deleted")
    assert bundle["canonical_candidates"][0]["state"] == "suppressed"


def test_record_ledger_events_skips_suppressed(tmp_path, monkeypatch):
    """suppressed イベントを台帳へ再記録しない（クールダウン恒久リセットの防止）。"""
    from scripts import recursive_self_improvement as rsi
    import pipeline_ledger

    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(pipeline_ledger, "LEDGER_PATH", ledger_path)

    events = [
        {"key": "k_supp", "status": "suppressed", "title": "抑制", "canonical_key": "k_supp"},
        {"key": "k_nr", "status": "needs_review", "title": "要レビュー", "canonical_key": "k_nr"},
        {"key": "k_val", "status": "validated", "title": "検証済み", "canonical_key": "k_val"},
    ]
    recorded = rsi.record_ledger_events(events)

    assert recorded == 2  # suppressed は記録しない
    entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    statuses = {e["key"]: e["status"] for e in entries}
    assert "k_supp" not in statuses
    assert statuses["k_nr"] == "needs_review"
    assert statuses["k_val"] == "validated"


def test_recursive_distinguishes_healthy_dedup_from_churn(tmp_path, monkeypatch):
    """applied 抑制=健全な重複排除、needs_review クールダウン抑制=滞留(churn) を区別する。"""
    from scripts import recursive_self_improvement as rsi
    import pipeline_ledger

    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(pipeline_ledger, "LEDGER_PATH", ledger_path)

    healthy_key = rsi.canonical_key("A項目", "説明A")
    pipeline_ledger.record(healthy_key, "applied", "A項目", canonical_key=healthy_key)
    churn_key = rsi.canonical_key("B項目", "説明B")
    pipeline_ledger.record(churn_key, "needs_review", "B項目", canonical_key=churn_key)

    report = {
        "date": "2026-07-23",
        "needs_review": [
            {"id": "R1", "title": "A項目", "description": "説明A", "target_module": "frontend/src/app/page.tsx"},
            {"id": "R2", "title": "B項目", "description": "説明B", "target_module": "frontend/src/app/page.tsx"},
        ],
    }
    bundle = rsi.build_recursive_self_improvement(report, prompt_feedback_log=[], workspace_root=tmp_path)

    ms = bundle["measurement_summary"]
    assert bundle["suppressed_count"] == 2
    assert ms["suppressed_healthy_count"] == 1
    assert ms["suppressed_churn_count"] == 1
    assert ms["churn_rate"] == 50.0
    healthy_flags = {s["canonical_key"]: s["healthy"] for s in bundle["suppressions"]}
    assert healthy_flags[healthy_key] is True
    assert healthy_flags[churn_key] is False


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
