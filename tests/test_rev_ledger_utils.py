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
