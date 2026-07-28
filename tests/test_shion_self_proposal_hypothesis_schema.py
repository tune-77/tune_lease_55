from __future__ import annotations

import json


def test_usage_loop_saves_verifiable_hypothesis_schema(tmp_path, monkeypatch):
    import api.usage_loop_engineering as usage

    proposals_path = tmp_path / "usage_loop_proposals.jsonl"
    monkeypatch.setattr(usage, "_PROPOSALS_PATH", proposals_path)
    monkeypatch.setattr(
        usage,
        "aggregate_usage",
        lambda days=30: {
            "window_days": days,
            "total_events": 3,
            "most_used": [{"path": "/", "visit_count": 2, "last_visited": "2026-07-26T03:00:00"}],
            "least_used": [{"path": "/help", "visit_count": 1, "last_visited": "2026-07-26T03:01:00"}],
        },
    )
    monkeypatch.setattr(
        usage,
        "_call_gemini",
        lambda _prompt: json.dumps(
            [
                {
                    "title": "審査導線を短縮",
                    "target_page": "/",
                    "hypothesis": "トップから審査へ直接入れると審査開始までの迷いが減る",
                    "evidence": "/ が2回、/help が1回利用されている",
                    "proposed_change": "トップに審査開始ボタンを置く",
                    "success_metric": "/ から /screening への遷移率",
                    "verification_plan": "変更前後7日で遷移ログを比較",
                    "risk": "トップが混雑する",
                    "priority": "high",
                }
            ],
            ensure_ascii=False,
        ),
    )

    result = usage.generate_proposals()
    saved = json.loads(proposals_path.read_text(encoding="utf-8").splitlines()[0])

    assert result["generated"] is True
    assert saved["proposal_schema"] == "shion_self_hypothesis_v1"
    assert saved["human_decision_status"] == "needs_human_review"
    assert saved["hypothesis"].startswith("トップから審査へ")
    assert saved["success_metric"] == "/ から /screening への遷移率"


def test_feedback_pattern_saves_verifiable_hypothesis_schema(tmp_path, monkeypatch):
    import api.feedback_pattern_loop as feedback

    proposals_path = tmp_path / "feedback_pattern_proposals.jsonl"
    monkeypatch.setattr(feedback, "_PROPOSALS_PATH", proposals_path)
    monkeypatch.setattr(
        feedback,
        "aggregate_feedback",
        lambda: {
            "total_feedback": 2,
            "rating_counts": {"thin": 2},
            "negative_examples": [{"rating": "thin", "route": "screening", "message": "薄い", "response": "一般論", "comment": ""}],
        },
    )
    monkeypatch.setattr(
        feedback,
        "aggregate_experience_signals",
        lambda: {"total_events": 1, "weak_signal_count": 1, "weak_examples": []},
    )
    monkeypatch.setattr(
        feedback,
        "call_gemini_json",
        lambda _prompt: [
            {
                "title": "根拠を先に出す",
                "pattern": "一般論に見える回答が薄いと評価される",
                "hypothesis": "根拠を先に出すとthin評価が減る",
                "evidence": "thinが2件、弱シグナルが1件",
                "proposed_change": "回答冒頭に案件固有の根拠を1つ置く",
                "success_metric": "thin評価率の低下",
                "verification_plan": "変更前後7日でhuman_response_feedbackを比較",
                "risk": "冒頭が長くなる",
            }
        ],
    )

    result = feedback.generate_proposals()
    saved = json.loads(proposals_path.read_text(encoding="utf-8").splitlines()[0])

    assert result["generated"] is True
    assert saved["proposal_schema"] == "shion_self_hypothesis_v1"
    assert saved["hypothesis"] == "根拠を先に出すとthin評価が減る"
    assert saved["verification_plan"].startswith("変更前後7日")


def test_report_attachment_preserves_hypothesis_fields(tmp_path, monkeypatch):
    import scripts.attach_shion_self_proposals_to_report as attach

    source_path = tmp_path / "usage.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "title": "審査導線を短縮",
                "target_page": "/",
                "hypothesis": "トップから審査へ直接入れると迷いが減る",
                "evidence": "/ が70回利用",
                "proposed_change": "トップに審査開始ボタンを置く",
                "success_metric": "/screening 遷移率",
                "verification_plan": "前後7日比較",
                "risk": "トップの情報量増加",
                "priority": "high",
                "proposal_schema": "shion_self_hypothesis_v1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        attach,
        "SOURCES",
        [{"path": source_path, "source": "usage_loop", "kind": "画面利用", "summary_keys": ("hypothesis", "evidence")}],
    )

    section = attach.collect_shion_self_proposals(limit=5)

    assert section["count"] == 1
    assert section["counts_by_layer"]["usage_based"] == 1
    assert section["items"][0]["evidence_layer"] == "usage_based"
    assert section["items"][0]["proposal_schema"] == "shion_self_hypothesis_v1"
    assert section["items"][0]["hypothesis"].startswith("トップから審査へ")
    assert section["items"][0]["success_metric"] == "/screening 遷移率"


def test_report_attachment_classifies_system_audit_layer(tmp_path, monkeypatch):
    import scripts.attach_shion_self_proposals_to_report as attach

    source_path = tmp_path / "lease_system_gap_analysis.json"
    source_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-27T09:00:00",
                "gaps": [
                    {
                        "id": "GAP-001",
                        "title": "DB品質監査が不足",
                        "priority": "high",
                        "category": "data-quality",
                        "evidence": ["監査レポートで欠損が見つかった"],
                        "impact": "審査判断の説明性が下がる",
                        "recommended_action": "DB品質監査をレポート化する",
                        "suggested_program": "scripts/audit_db_quality.py",
                        "guardrail": "本体DBを直接変更しない",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        attach,
        "SOURCES",
        [
            {
                "path": source_path,
                "source": "lease_system_gap_analysis",
                "kind": "システム監査",
                "evidence_layer": "system_audit_based",
                "format": "system_gap_report",
                "summary_keys": ("impact", "recommended_action", "guardrail"),
            }
        ],
    )

    section = attach.collect_shion_self_proposals(limit=5)

    assert section["count"] == 1
    assert section["counts_by_layer"]["system_audit_based"] == 1
    assert section["items"][0]["evidence_layer"] == "system_audit_based"
    assert section["items"][0]["kind"] == "システム監査"
    assert section["items"][0]["proposed_change"] == "DB品質監査をレポート化する"


def test_report_attachment_suppresses_resolved_self_proposals(tmp_path, monkeypatch):
    import scripts.attach_shion_self_proposals_to_report as attach

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "improvement_report_20260729.json").write_text(
        json.dumps(
            {
                "applied_improvements": [
                    {
                        "id": "REV-101",
                        "title": "審査導線を短縮",
                        "description": "自己提案から採用済み",
                    }
                ],
                "needs_review": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "cloudrun_improvement_log.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "event_type": "improvement_delete",
                "title": "不要な提案を隠す",
                "body": "既に整理済み",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "usage.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"title": "審査導線を短縮", "hypothesis": "採用済みなので消える"}, ensure_ascii=False),
                json.dumps({"title": "不要な提案を隠す", "hypothesis": "削除済みなので消える"}, ensure_ascii=False),
                json.dumps({"title": "新しい提案だけ残す", "hypothesis": "これは未対応"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(attach, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(attach, "LOCAL_IMPROVEMENT_LOG", log_path)
    monkeypatch.setattr(
        attach,
        "SOURCES",
        [{"path": source_path, "source": "usage_loop", "kind": "画面利用", "summary_keys": ("hypothesis",)}],
    )

    section = attach.collect_shion_self_proposals(limit=5)

    assert section["count"] == 1
    assert section["source_counts_by_kind"]["画面利用"] == 3
    assert section["counts_by_kind"]["画面利用"] == 1
    assert section["suppressed_resolved_count"] == 2
    assert [item["title"] for item in section["items"]] == ["新しい提案だけ残す"]


def test_scheduler_push_preserves_hypothesis_fields(tmp_path, monkeypatch):
    import api.scheduler as scheduler

    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    pushed = scheduler._push_proposals_to_improvement_log(
        [
            {
                "title": "審査導線を短縮",
                "target_page": "/",
                "hypothesis": "トップから審査へ直接入れると迷いが減る",
                "evidence": "/ が70回利用",
                "proposed_change": "トップに審査開始ボタンを置く",
                "success_metric": "/screening 遷移率",
                "verification_plan": "前後7日比較",
                "risk": "トップの情報量増加",
                "priority": "high",
                "status": "proposed",
                "proposal_schema": "shion_self_hypothesis_v1",
                "human_decision_status": "needs_human_review",
            }
        ],
        source="usage_loop",
    )

    rows = (tmp_path / "cloudrun_improvement_log.jsonl").read_text(encoding="utf-8").splitlines()
    saved = json.loads(rows[0])

    assert pushed == 1
    assert saved["surface"] == "shion_self_proposal"
    assert saved["proposal_schema"] == "shion_self_hypothesis_v1"
    assert saved["hypothesis"].startswith("トップから審査へ")
    assert saved["success_metric"] == "/screening 遷移率"
    assert "## 仮説" in saved["body"]
    assert "## 検証方法" in saved["body"]


def test_feedback_pattern_pdca_handles_mixed_timezone_timestamps(tmp_path, monkeypatch):
    import api.feedback_pattern_loop as feedback

    improvement_log = tmp_path / "cloudrun_improvement_log.jsonl"
    feedback_log = tmp_path / "human_response_feedback.jsonl"
    pdca_log = tmp_path / "shion_self_pdca_log.jsonl"
    improvement_log.write_text(
        json.dumps(
            {
                "proposed_by": "shion",
                "surface": "shion_self_proposal",
                "title": "短く答える",
                "ts": "2026-07-29T00:00:00Z",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    feedback_log.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-07-28T23:00:00", "rating": "thin"}, ensure_ascii=False),
                json.dumps({"ts": "2026-07-29T01:00:00+00:00", "rating": "good"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(feedback, "_IMPROVEMENT_LOG_PATH", improvement_log)
    monkeypatch.setattr(feedback, "_FEEDBACK_PATH", feedback_log)
    monkeypatch.setattr(feedback, "_PDCA_LOG_PATH", pdca_log)

    result = feedback.evaluate_proposal_impact()

    assert result["evaluated"] == 1
    assert result["results"][0]["verdict"] == "improved"
    assert pdca_log.exists()
