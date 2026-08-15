from __future__ import annotations

import json
from pathlib import Path

from scripts import improvement_state_resolver as resolver


def write_ledger(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_deleted_decision_survives_later_needs_review(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger,
        [
            {"key": "same", "canonical_key": "same", "status": "deleted", "title": "削除済み"},
            {"key": "same", "canonical_key": "same", "status": "needs_review", "title": "削除済み"},
        ],
    )

    latest = resolver.load_latest_by_alias(ledger)

    assert latest["same"]["status"] == "deleted"
    assert resolver.load_latest_status_by_title(ledger)[resolver.normalize_title("削除済み")] == "deleted"


def test_apply_failed_overrides_older_applied(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger,
        [
            {"key": "same", "canonical_key": "same", "status": "applied", "title": "未反映訂正"},
            {"key": "same", "canonical_key": "same", "status": "apply_failed", "title": "未反映訂正"},
        ],
    )

    assert resolver.load_latest_by_alias(ledger)["same"]["status"] == "apply_failed"
    assert resolver.status_to_report_bucket("apply_failed") == ""


def test_latest_by_key_does_not_double_count_aliases(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger,
        [
            {
                "key": "runtime-key",
                "canonical_key": "canonical-alias",
                "status": "applied",
                "title": "別名あり",
            },
        ],
    )

    assert set(resolver.load_latest_by_alias(ledger)) == {"runtime-key", "canonical-alias"}
    assert set(resolver.load_latest_by_key(ledger)) == {"runtime-key"}
