"""改善台帳バリデータの検査。

ledger-sync.yml は GITHUB_TOKEN で台帳PRを作るため pr-checks.yml が起動しない。
その穴埋めがこのバリデータなので、バリデータ自身が壊れていないことを担保する。
"""
from __future__ import annotations

import json

from scripts.validate_improvement_ledger import (
    KNOWN_STATUSES,
    check_append_only,
    parse_lines,
    validate,
)


def _line(**kw) -> str:
    base = {"key": "misc_0001", "rev_id": "REV-001", "status": "applied", "title": "テスト項目"}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


def test_valid_ledger_has_no_problems():
    text = "\n".join([_line(), _line(rev_id="REV-002", status="rejected")]) + "\n"

    entries, problems = parse_lines(text)

    assert len(entries) == 2
    assert problems == []


def test_broken_json_line_is_reported():
    text = _line() + "\n{壊れた行\n"

    entries, problems = parse_lines(text)

    assert len(entries) == 1
    assert any("JSONとして読めない" in p for p in problems)


def test_unknown_status_is_reported():
    text = _line(status="なにこれ") + "\n"

    _, problems = parse_lines(text)

    assert any("未知の status" in p for p in problems)


def test_existing_ledger_statuses_are_all_known():
    """実際の台帳に入っている値を推測で狭めていないこと。

    初版は done / doing を known から落としており、既存295行が全て NG になった。
    """
    for status in ("applied", "done", "needs_review", "pending", "doing", "rejected"):
        assert status in KNOWN_STATUSES


def test_append_only_accepts_appended_lines():
    baseline = _line() + "\n"
    current = baseline + _line(rev_id="REV-002") + "\n"

    assert check_append_only(current, baseline) == []


def test_append_only_rejects_deleted_lines():
    baseline = _line() + "\n" + _line(rev_id="REV-002") + "\n"
    current = _line() + "\n"

    problems = check_append_only(current, baseline)

    assert any("行数が減っている" in p for p in problems)


def test_append_only_rejects_rewritten_existing_line():
    baseline = _line(status="applied") + "\n"
    current = _line(status="rejected") + "\n"

    problems = check_append_only(current, baseline)

    assert any("既存行が書き換えられている" in p for p in problems)


def test_validate_reports_missing_file(tmp_path):
    problems = validate(tmp_path / "nope.jsonl", base_ref=None)

    assert any("見つからない" in p for p in problems)


def test_validate_reports_empty_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("", encoding="utf-8")

    problems = validate(path, base_ref=None)

    assert any("1件も無い" in p for p in problems)


def test_repository_ledger_is_valid():
    """リポジトリに実在する台帳が、このバリデータを通ること。"""
    from scripts.validate_improvement_ledger import DEFAULT_LEDGER

    assert validate(DEFAULT_LEDGER, base_ref=None) == []
