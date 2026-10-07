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
    assert 'log_step "evaluate_experience_replay_historical" $?' in script
    replay_pos = script.index("scripts/build_experience_replay_eval_set.py")
    historical_pos = script.index("scripts/evaluate_experience_replay_historical.py")
    ab_report_pos = script.index("scripts/build_judgment_asset_ab_report.py")
    assert replay_pos < historical_pos < ab_report_pos


def _run_main(tmp_path, monkeypatch, *, passing: bool, extra: list[str]) -> int:
    import sys

    eval_set = tmp_path / "eval.json"
    eval_set.write_text(
        json.dumps([{"id": "c1", "query": "q", "concepts": [{"aliases": ["返済原資"]}]}], ensure_ascii=False),
        encoding="utf-8",
    )
    feedback = tmp_path / "feedback.jsonl"
    _write_jsonl(feedback, [{"question": "q", "response_text": "返済原資を確認します" if passing else "わからない"}])
    monkeypatch.setattr(
        hist,
        "build_report",
        lambda **_: {
            "missing_historical_answers": [],
            "answer_coverage": {"total_cases": 1, "scored_with_latest": 1, "stale": [], "no_answer": []},
            "final": {
                "total": 1, "passed": 1 if passing else 0, "average_score": 0,
                "concept_coverage": 0, "forbidden_cases": 0, "uncertainty_misses": 0, "cases": [],
            },
        },
    )
    monkeypatch.setattr(hist, "write_outputs", lambda *a, **k: None)
    monkeypatch.setattr(
        sys, "argv",
        ["x", "--eval-set", str(eval_set), "--prompt-feedback", str(feedback),
         "--chat-log", str(tmp_path / "chat.jsonl"), *extra],
    )
    return hist.main()


def test_failed_cases_are_a_quality_warning_not_a_pipeline_failure(tmp_path, monkeypatch, capsys):
    # REV-496: 不合格は品質の指標。8/14 から変わらない 0/10 が毎日パイプライン障害として検出されていた
    assert _run_main(tmp_path, monkeypatch, passing=False, extra=[]) == 0
    assert "warn: 不合格 1/1 件" in capsys.readouterr().out


def test_strict_flag_keeps_failing_exit_code(tmp_path, monkeypatch):
    assert _run_main(tmp_path, monkeypatch, passing=False, extra=["--strict"]) == 1
    assert _run_main(tmp_path, monkeypatch, passing=True, extra=["--strict"]) == 0


def _case(case_id: str, query: str) -> dict:
    return {
        "id": case_id,
        "query": query,
        "required_concepts": [["厨房設備"], ["資金繰り"]],
        "forbidden_claims": [],
        "require_uncertainty": False,
        "_source_id": f"hash-{case_id}",
    }


def test_latest_answer_is_used_over_first_record():
    # REV-497: 以前は同じ質問の最初の記録（6/12）を毎回使い、新しい回答が評価に入らなかった
    cases = [_case("c1", "飲食業をやりたいんだけどリース使う？")]
    feedback = [
        {"question_hash": "hash-c1", "question": "飲食業をやりたいんだけどリース使う？",
         "response_text": "古い回答", "timestamp": "2026-06-12T10:50:24"},
    ]
    chat = [
        {"user_message": "飲食業をやりたいんだけど、リース使う?", "assistant_reply": "厨房設備と資金繰りを見る",
         "ts": "2026-10-06T09:00:00", "surface": "lease_intelligence_dialogue"},
    ]

    answers = hist.build_historical_answers(cases, feedback, chat, fresh_since="2026-09-08")

    assert answers["c1"]["answer"] == "厨房設備と資金繰りを見る"
    assert answers["c1"]["freshness"] == "fresh"
    assert answers["c1"]["answer_source"] == "chat_log:lease_intelligence_dialogue"


def test_only_old_answers_are_not_scored_and_shown_with_date():
    cases = [_case("c1", "電車はリース？"), _case("c2", "レンタカーを借りるには")]
    feedback = [{"question_hash": "hash-c1", "question": "電車はリース？", "response_text": "古い",
                 "timestamp": "2026-06-12T10:52:59"}]

    answers = hist.build_historical_answers(cases, feedback, [], fresh_since="2026-09-08")
    report = hist.build_report(cases=cases, answers=answers, eval_set_path=Path("e"), prompt_feedback_path=Path("f"))

    coverage = report["answer_coverage"]
    assert coverage["scored_with_latest"] == 0
    assert coverage["stale"] == [{"id": "c1", "query": "電車はリース？", "latest_answered_at": "2026-06-12T10:52:59"}]
    assert [item["id"] for item in coverage["no_answer"]] == ["c2"]
    assert report["final"]["total"] == 0


def test_different_question_with_same_template_is_not_matched():
    # 「中古車はリースできる？」を「焼却炉はリースできる？」の回答として採点しない
    cases = [_case("c1", "焼却炉はリースできる？")]
    chat = [{"user_message": "中古車はリースできる？", "assistant_reply": "できます", "ts": "2026-10-01T00:00:00"}]

    answers = hist.build_historical_answers(cases, [], chat, fresh_since="2026-09-08")

    assert answers["c1"]["freshness"] == "none"
