from __future__ import annotations

import json
from pathlib import Path

from scripts import build_experience_replay_eval_set as replay
from scripts.evaluate_answer_quality import evaluate_answers


def test_build_replay_eval_set_filters_casual_and_keeps_lease_cases():
    flywheel = {
        "replay_eval_candidates": [
            {
                "id": "casual",
                "source": "prompt_feedback",
                "context": {"question": "横須賀に釣りに行く"},
                "scores": {"total": 8},
                "gate": {"priority": 70, "reason": "prompt changed"},
            },
            {
                "id": "meta",
                "source": "prompt_feedback",
                "context": {"question": "リース以外の話もできるのね"},
                "scores": {"total": 8},
                "gate": {"priority": 70, "reason": "prompt changed"},
            },
            {
                "id": "lease",
                "source": "prompt_feedback",
                "context": {"question": "うちのリース会社では工事費は3割くらいまで"},
                "scores": {"total": 8},
                "gate": {"priority": 70, "reason": "prompt changed"},
            },
            {
                "id": "business_plan",
                "source": "prompt_feedback",
                "context": {"question": "事業計画はどうやって作る？"},
                "scores": {"total": 8},
                "gate": {"priority": 70, "reason": "prompt changed"},
            },
        ],
        "promotion_review_candidates": [
            {
                "id": "promotion",
                "source": "screening_feedback",
                "context": {"question": "境界スコアを条件付き承認へ寄せる"},
                "scores": {"total": 15},
                "gate": {"priority": 90, "reason": "positive"},
            }
        ],
    }

    cases = replay.build_replay_eval_set(flywheel, limit=10)

    assert [case["query"] for case in cases] == [
        "うちのリース会社では工事費は3割くらいまで",
        "事業計画はどうやって作る？",
    ]
    assert cases[0]["required_concepts"][0] == ["工事費", "付帯工事"]
    assert cases[1]["required_concepts"][0] == ["売上計画", "収支計画"]
    assert all("境界スコア" not in case["query"] for case in cases)


def test_electric_train_is_not_classified_as_car_registration_case():
    concepts, forbidden, uncertainty = replay.concepts_for_query("電車はリース？")

    assert concepts[0] == ["車両", "設備"]
    assert concepts[1] == ["長期契約", "保守"]
    assert "車検切れでも問題ない" not in forbidden
    assert uncertainty is True


def test_replay_eval_set_is_answer_quality_compatible():
    flywheel = {
        "replay_eval_candidates": [
            {
                "id": "subsidy",
                "source": "prompt_feedback",
                "context": {"question": "補助金前提の工作機械リースで確認すべきことを短く教えて"},
                "scores": {"total": 8},
                "gate": {"priority": 70, "reason": "prompt changed"},
            }
        ]
    }
    cases = replay.build_replay_eval_set(flywheel, limit=1)
    answers = {
        cases[0]["id"]: "公募要領で補助対象とリース可否を要確認。採択・未採択で返済原資と支払原資を分けて見る。"
    }

    summary = evaluate_answers(cases, answers)

    assert summary["total"] == 1
    assert summary["passed"] == 1


def test_write_outputs_and_daily_post_wiring(tmp_path):
    cases = [
        {
            "id": "experience_replay_test",
            "query": "飲食業の厨房設備リース、審査で何を見る？",
            "required_concepts": [["厨房設備"], ["資金繰り"], ["保全"]],
            "forbidden_claims": ["飲食業なら一律否決"],
            "require_uncertainty": True,
        }
    ]
    output_json = tmp_path / "eval.json"
    output_md = tmp_path / "eval.md"

    replay.write_outputs(cases, output_json=output_json, output_md=output_md)

    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["id"] == "experience_replay_test"
    assert "# Experience Replay Eval Set" in output_md.read_text(encoding="utf-8")

    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")
    assert "scripts/build_experience_replay_eval_set.py" in script
    assert 'log_step "build_experience_replay_eval_set"' in script
    flywheel_pos = script.index("scripts/build_experience_flywheel_report.py")
    replay_pos = script.index("scripts/build_experience_replay_eval_set.py")
    ab_report_pos = script.index("scripts/build_judgment_asset_ab_report.py")
    assert flywheel_pos < replay_pos < ab_report_pos
