from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("rev_ledger_utils", ROOT / "scripts" / "rev_ledger_utils.py")
rev_ledger_utils = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rev_ledger_utils)


def test_max_rev_number_empty_ledger():
    assert rev_ledger_utils.max_rev_number([]) == 0


def test_max_rev_number_default_field_is_rev_id():
    entries = [{"rev_id": "REV-003"}, {"rev_id": "REV-048a"}, {"rev_id": "REV-012"}]
    assert rev_ledger_utils.max_rev_number(entries) == 48


def test_max_rev_number_ignores_entries_without_rev_id():
    entries = [{"rev_id": "REV-005"}, {"status": "applied"}, {"rev_id": ""}]
    assert rev_ledger_utils.max_rev_number(entries) == 5


def test_max_rev_number_checks_additional_fields_when_given():
    entries = [{"rev_id": "", "key": "REV-019"}, {"rev_id": "REV-007", "key": "misc_abc"}]
    assert rev_ledger_utils.max_rev_number(entries, fields=("rev_id", "key")) == 19


def test_load_all_rev_sources_merges_both_ledgers(tmp_path):
    """REV-292 の重複発行事故の回帰防止: ledger_rules.json (analyze_*.py系5本)
    と improvement_ledger.jsonl (step1_extract_and_structure.py) は互いを見ずに
    別々に採番していたため、片方にしか無い番号を空き番号と誤認していた。
    load_all_rev_sources() は両方を横断して合算しなければならない。"""
    import json

    rule_engine_dir = tmp_path / "api" / "rule_engine"
    rule_engine_dir.mkdir(parents=True)
    (rule_engine_dir / "ledger_rules.json").write_text(
        json.dumps([{"rev_id": "REV-100"}, {"rev_id": "REV-292"}]),
        encoding="utf-8",
    )

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "improvement_ledger.jsonl").write_text(
        '{"rev_id": "REV-233", "status": "applied"}\n'
        '{"rev_id": "REV-050", "status": "needs_review"}\n',
        encoding="utf-8",
    )

    entries = rev_ledger_utils.load_all_rev_sources(repo_root=tmp_path)
    assert len(entries) == 4
    assert rev_ledger_utils.max_rev_number(entries, fields=("rev_id", "key")) == 292


def test_load_all_rev_sources_missing_files_return_empty(tmp_path):
    assert rev_ledger_utils.load_all_rev_sources(repo_root=tmp_path) == []
