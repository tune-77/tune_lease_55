from __future__ import annotations

import json
from pathlib import Path

from scripts import build_experience_flywheel_report as flywheel


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_experience_flywheel_gates_review_replay_and_observe():
    report = flywheel.build_experience_flywheel(
        prompt_feedback_rows=[
            {
                "question": "電車はリース？",
                "response_preview": "鉄道車両のリースについて回答",
                "response_diff_from_previous": "+ detail",
                "pdca_applied": True,
                "surface": "next_chat_rag",
            }
        ],
        experience_event_rows=[
            {
                "id": "exp1",
                "ts": "2026-08-14T00:00:00+00:00",
                "route": "case_screening",
                "message_preview": "競合料率が低い案件で合わせるべき？",
                "response_start": "採算と戦略価値を分けて確認する",
                "practical_scene": {"label": "競合・料率条件がある時", "learned_entry_count": 4},
                "memory_refs_count": 2,
                "knowledge_refs_count": 1,
                "judgment_directive": "採算、競合理由、長期価値へ分解する",
            }
        ],
        screening_feedback_rows=[
            {
                "id": "sf1",
                "ts": "2026-08-14T00:01:00+00:00",
                "rating": "使える",
                "comment": "条件付き承認の整理として有効",
                "target": "ringi_policy",
                "score": 65.5,
                "hantei": "条件付き承認",
                "context": {"customer_type": "新規先", "has_asset_context": True},
                "issue_text": "境界スコアを条件設定で承認側へ寄せられるか",
                "ringi_policy_text": "物件保全と銀行支援確認を条件に限定承認",
            }
        ],
        judgment_asset_feedback_rows=[
            {
                "case_id": "case1",
                "rule_id": "rule1",
                "outcome": "neutral",
                "note": "主論点ではなかった",
                "used_at": "2026-08-14T00:02:00",
            }
        ],
        generated_at="2026-08-14T00:03:00+00:00",
    )

    assert report["summary"]["raw_candidates"] == 4
    assert report["summary"]["by_gate"]["promote_to_review"] == 1
    assert report["summary"]["by_gate"]["replay_eval"] == 2
    assert report["summary"]["by_gate"]["observe_only"] == 1
    assert report["promotion_review_candidates"][0]["gate"]["recommended_action"] == "review_screening_judgment_pattern"
    assert {item["source"] for item in report["replay_eval_candidates"]} == {
        "prompt_feedback",
        "judgment_asset_feedback",
    }
    assert report["observe_only_candidates"][0]["source"] == "shion_experience"
    assert any(item["action"] == "build_replay_eval_set" for item in report["recommendations"])


def test_experience_flywheel_quarantines_insufficient_rows_and_collapses_duplicates():
    row = {
        "id": "sf1",
        "rating": "",
        "target": "",
        "issue_text": "",
        "ringi_policy_text": "",
    }
    report = flywheel.build_experience_flywheel(
        screening_feedback_rows=[row, dict(row, id="sf2")],
        generated_at="2026-08-14T00:00:00+00:00",
    )

    assert report["summary"]["raw_candidates"] == 2
    assert report["summary"]["deduped_candidates"] == 1
    assert report["summary"]["duplicates_collapsed"] == 1
    assert report["summary"]["by_gate"]["quarantine"] == 1
    assert report["quarantined_candidates"][0]["gate"]["recommended_action"] == "do_not_learn"


def test_write_outputs_creates_json_and_markdown(tmp_path):
    report = flywheel.build_experience_flywheel(
        judgment_asset_feedback_rows=[
            {
                "case_id": "case1",
                "rule_id": "rule1",
                "outcome": "helped",
                "note": "確認軸として有効",
            }
        ],
        generated_at="2026-08-14T00:00:00+00:00",
    )

    output_json = tmp_path / "experience.json"
    output_md = tmp_path / "experience.md"
    flywheel.write_outputs(report, output_json=output_json, output_md=output_md)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")
    assert payload["mode"] == "read_only_experience_flywheel_gate"
    assert "# Experience Flywheel Report" in markdown
    assert "## Promotion Review" in markdown


def test_experience_flywheel_is_wired_into_daily_post_pipeline():
    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")

    assert "scripts/build_experience_flywheel_report.py" in script
    assert 'log_step "build_experience_flywheel_report"' in script
    field_review_pos = script.index("scripts/build_judgment_asset_field_review.py")
    flywheel_pos = script.index("scripts/build_experience_flywheel_report.py")
    ab_report_pos = script.index("scripts/build_judgment_asset_ab_report.py")
    assert field_review_pos < flywheel_pos < ab_report_pos
