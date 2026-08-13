from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate_experience_replay_historical as hist


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_historical_answers_match_by_source_id_and_score():
    cases = [
        {
            "id": "case1",
            "query": "飲食業をやりたいんだけどリース使う？",
            "required_concepts": [["厨房設備"], ["資金繰り"], ["保全"]],
            "forbidden_claims": ["飲食業なら一律否決"],
            "require_uncertainty": True,
            "_source_id": "hash1",
        }
    ]
    rows = [
        {
            "question_hash": "hash1",
            "question": "別の文",
            "response_text": "要確認。厨房設備、資金繰り、保全を分けて見る。",
        }
    ]

    answers = hist.build_historical_answers(cases, rows)
    report = hist.build_report(
        cases=cases,
        answers=answers,
        eval_set_path=Path("eval.json"),
        prompt_feedback_path=Path("feedback.jsonl"),
    )

    assert answers["case1"]["matched_historical_log"] is True
    assert report["final"]["passed"] == 1
    assert report["missing_historical_answers"] == []


def test_historical_report_marks_missing_answers():
    cases = [
        {
            "id": "case1",
            "query": "メイン先の方がリースやりやすいの？",
            "required_concepts": [["メインバンク"], ["支援姿勢"], ["返済原資"]],
            "forbidden_claims": ["審査不要"],
            "require_uncertainty": False,
            "_source_id": "missing",
        }
    ]

    answers = hist.build_historical_answers(cases, [])
    report = hist.build_report(
        cases=cases,
        answers=answers,
        eval_set_path=Path("eval.json"),
        prompt_feedback_path=Path("feedback.jsonl"),
    )

    assert answers["case1"]["matched_historical_log"] is False
    assert report["missing_historical_answers"][0]["id"] == "case1"


def test_historical_outputs_and_daily_post_wiring(tmp_path):
    report = {
        "generated_at": "2026-08-14T00:00:00",
        "mode": "historical_local_no_network",
        "missing_historical_answers": [],
        "final": {
            "total": 1,
            "passed": 0,
            "pass_rate": 0.0,
            "average_score": 20.0,
            "concept_coverage": 0.0,
            "forbidden_cases": 0,
            "uncertainty_misses": 1,
            "cases": [
                {
                    "id": "case1",
                    "query": "飲食業",
                    "score": 20.0,
                    "passed": False,
                    "concept_results": [{"aliases": ["厨房設備"], "matched": False}],
                    "uncertainty_present": False,
                }
            ],
        },
    }
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    hist.write_outputs(report, output_json=output_json, output_md=output_md)

    assert json.loads(output_json.read_text(encoding="utf-8"))["mode"] == "historical_local_no_network"
    assert "# Experience Replay Historical Quality" in output_md.read_text(encoding="utf-8")

    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")
    assert "scripts/evaluate_experience_replay_historical.py" in script
    assert 'log_step "evaluate_experience_replay_historical" 0' in script
    replay_pos = script.index("scripts/build_experience_replay_eval_set.py")
    historical_pos = script.index("scripts/evaluate_experience_replay_historical.py")
    ab_report_pos = script.index("scripts/build_judgment_asset_ab_report.py")
    assert replay_pos < historical_pos < ab_report_pos
