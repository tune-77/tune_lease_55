from __future__ import annotations

import json
from pathlib import Path

from scripts import build_experience_replay_checklist_candidates as checklist


def test_build_checklist_candidates_from_failed_historical_cases():
    report = {
        "eval_set": "api/knowledge/experience_replay_eval_set.json",
        "final": {
            "cases": [
                {
                    "id": "case_food",
                    "query": "飲食業をやりたいんだけどリース使う？",
                    "score": 43.3,
                    "passed": False,
                    "concept_results": [
                        {"aliases": ["厨房設備", "店舗設備"], "matched": False},
                        {"aliases": ["資金繰り", "返済原資"], "matched": True},
                        {"aliases": ["撤退", "中古市場", "保全"], "matched": False},
                    ],
                    "uncertainty_required": True,
                    "uncertainty_present": False,
                },
                {
                    "id": "case_ok",
                    "query": "通ったケース",
                    "score": 100.0,
                    "passed": True,
                    "concept_results": [],
                },
            ]
        },
    }

    payload = checklist.build_checklist_candidates(report)

    assert payload["summary"]["failed_cases"] == 1
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["uncertainty_miss_count"] == 1
    case_candidate = payload["case_checklist_candidates"][0]
    assert case_candidate["topic"] == "飲食業リース"
    assert case_candidate["missing_concepts"] == [
        "厨房設備 / 店舗設備",
        "撤退 / 中古市場 / 保全",
    ]
    assert any("撤退時の物件回収" in item for item in case_candidate["checklist_items"])
    assert any(item["id"] == "global_uncertainty_01" for item in payload["global_checklist_candidates"])


def test_checklist_outputs_and_daily_post_wiring(tmp_path):
    payload = checklist.build_checklist_candidates(
        {
            "final": {
                "cases": [
                    {
                        "id": "case_car",
                        "query": "車検切れのくるまはやばいよね",
                        "score": 76.7,
                        "passed": False,
                        "concept_results": [{"aliases": ["所有者", "使用者"], "matched": False}],
                        "uncertainty_required": True,
                        "uncertainty_present": True,
                    }
                ]
            }
        }
    )
    output_json = tmp_path / "checklist.json"
    output_md = tmp_path / "checklist.md"

    checklist.write_outputs(payload, output_json=output_json, output_md=output_md)

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["mode"] == "experience_replay_response_checklist_candidates"
    assert "# Experience Replay Checklist Candidates" in output_md.read_text(encoding="utf-8")

    script = Path("scripts/run_daily_improvement_post.sh").read_text(encoding="utf-8")
    assert "scripts/build_experience_replay_checklist_candidates.py" in script
    assert "scripts/review_experience_replay_checklist.py" in script
    assert 'log_step "build_experience_replay_checklist_candidates"' in script
    assert 'log_step "review_experience_replay_checklist"' in script
    historical_pos = script.index("scripts/evaluate_experience_replay_historical.py")
    checklist_pos = script.index("scripts/build_experience_replay_checklist_candidates.py")
    review_pos = script.index("scripts/review_experience_replay_checklist.py")
    ab_report_pos = script.index("scripts/build_judgment_asset_ab_report.py")
    assert historical_pos < checklist_pos < review_pos < ab_report_pos
