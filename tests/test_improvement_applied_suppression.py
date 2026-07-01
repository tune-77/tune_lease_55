from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / ".agents" / "skills" / "auto-improvement-pipeline"
SCRIPTS_DIR = PIPELINE_DIR / "scripts"


def _load_pipeline_runner():
    sys.path.insert(0, str(PIPELINE_DIR))
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "auto_improvement_pipeline_runner_for_test",
        PIPELINE_DIR / "pipeline_runner.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_suppresses_previously_applied_title_even_when_rev_id_changes(tmp_path) -> None:
    runner = _load_pipeline_runner()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "improvement_report_20260619.json").write_text(
        json.dumps(
            {
                "applied": [
                    {
                        "id": "REV-007",
                        "title": "ブルドーザー、ショベル・油圧ショベルのリース期間が「5年」と表示されているが、正しくは「6年」の可能",
                    }
                ],
                "needs_review": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    improvements = [
        {
            "id": "REV-002",
            "title": "ブルドーザー、ショベル・油圧ショベルのリース期間が「5年」と表示されているが、正しくは「6年」の可能",
            "description": "AI Chat 改善ログから再抽出",
        },
        {
            "id": "REV-008",
            "title": "案件ネットワーク画面で案件情報が何も表示されない。",
            "description": "未対応の別件",
        },
    ]

    kept, suppressed = runner.suppress_previously_applied_improvements(improvements, tmp_path)

    assert [item["id"] for item in kept] == ["REV-008"]
    assert suppressed[0]["id"] == "REV-002"
    assert suppressed[0]["matched_applied_id"] == "REV-007"
    assert suppressed[0]["reason"] == "過去レポートで同一タイトルがapplied済み"
