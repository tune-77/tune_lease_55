from pathlib import Path

import pytest

from api import shion_action_ledger as ledger


def test_log_action_appends_and_reads_back(tmp_path: Path):
    path = tmp_path / "shion_action_ledger.jsonl"

    entry = ledger.log_action(
        "improvement_classified",
        summary="改善候補3件を今日やる/後回し/捨てるに分類",
        observed_sources=["/api/improvement-log"],
        risk_level="low",
        requires_user_approval=False,
        result="classified",
        path=path,
    )

    assert entry["agent"] == "shion"
    assert entry["action"] == "improvement_classified"
    assert path.read_text(encoding="utf-8").count("improvement_classified") == 1

    entries = ledger.read_actions(path=path)
    assert len(entries) == 1
    assert entries[0]["summary"] == "改善候補3件を今日やる/後回し/捨てるに分類"


def test_log_action_rejects_invalid_action(tmp_path: Path):
    with pytest.raises(ValueError):
        ledger.log_action("not_a_real_action", summary="x", path=tmp_path / "l.jsonl")


def test_log_action_rejects_invalid_risk_level(tmp_path: Path):
    with pytest.raises(ValueError):
        ledger.log_action(
            "system_watch", summary="x", risk_level="critical", path=tmp_path / "l.jsonl"
        )


def test_read_actions_skips_corrupt_lines(tmp_path: Path):
    path = tmp_path / "shion_action_ledger.jsonl"
    ledger.log_action("daily_report", summary="重大異常なし", path=path)
    with path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    ledger.log_action("daily_report", summary="2件目", path=path)

    entries = ledger.read_actions(path=path)
    assert len(entries) == 2


def test_read_actions_respects_limit(tmp_path: Path):
    path = tmp_path / "shion_action_ledger.jsonl"
    for i in range(5):
        ledger.log_action("daily_report", summary=f"report {i}", path=path)

    entries = ledger.read_actions(path=path, limit=2)
    assert len(entries) == 2
    assert entries[-1]["summary"] == "report 4"
