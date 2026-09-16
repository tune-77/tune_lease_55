from __future__ import annotations

import json
import sys
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


def test_main_warns_and_exits_1_when_final_cases_missing(tmp_path, monkeypatch, capsys):
    """report は読めたのに final.cases が空/欠落＝上位レポートのドリフトを検知する。"""
    report_path = tmp_path / "historical.json"
    report_path.write_text(
        json.dumps({"generated_at": "2026-09-01T00:00:00", "final": {"cases": []}}),
        encoding="utf-8",
    )
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_experience_replay_checklist_candidates.py",
            "--historical-report", str(report_path),
            "--output-json", str(output_json),
            "--output-md", str(output_md),
        ],
    )

    exit_code = checklist.main()

    assert exit_code == 1
    assert "final.cases" in capsys.readouterr().err


def test_main_returns_0_when_all_historical_cases_passed(tmp_path, monkeypatch, capsys):
    """failed_cases=0 は「全件合格」の可能性がある良い意味のゼロなので検知しない。"""
    report_path = tmp_path / "historical.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-09-01T00:00:00",
                "final": {"cases": [{"id": "ok", "query": "通ったケース", "passed": True}]},
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_experience_replay_checklist_candidates.py",
            "--historical-report", str(report_path),
            "--output-json", str(output_json),
            "--output-md", str(output_md),
        ],
    )

    exit_code = checklist.main()

    assert exit_code == 0
    assert capsys.readouterr().err == ""
