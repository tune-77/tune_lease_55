"""cleanup_rag_unrated_rules.py の整理ロジックのテスト。"""

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cleanup_rag_unrated_rules.py"
_spec = importlib.util.spec_from_file_location("cleanup_rag_unrated_rules", _SCRIPT)
cleanup = importlib.util.module_from_spec(_spec)
sys.modules["cleanup_rag_unrated_rules"] = cleanup
_spec.loader.exec_module(cleanup)

NOW = datetime(2026, 7, 10, tzinfo=timezone.utc)


def _rag(rev_id, path, generated_at, pending=True):
    return {
        "rev_id": rev_id,
        "type": "rag_boost_adjust",
        "pending_review": pending,
        "path": path,
        "generated_at": generated_at,
    }


def test_stale_notification_is_parked():
    rules = [
        _rag("RAG-OLD", "note-a", "2026-05-01T00:00:00+00:00"),
        _rag("RAG-NEW", "note-b", "2026-07-01T00:00:00+00:00"),
    ]
    kept, parked = cleanup.plan_cleanup(rules, now=NOW, max_age_days=30)
    assert [r["rev_id"] for r in parked] == ["RAG-OLD"]
    assert [r["rev_id"] for r in kept] == ["RAG-NEW"]
    assert "自動保留" in parked[0]["parked_reason"]
    assert parked[0]["parked_by"] == "cleanup_rag_unrated_rules"


def test_duplicate_path_keeps_only_latest():
    rules = [
        _rag("RAG-DUP-1", "note-a", "2026-07-01T00:00:00+00:00"),
        _rag("RAG-DUP-2", "note-a", "2026-07-05T00:00:00+00:00"),
    ]
    kept, parked = cleanup.plan_cleanup(rules, now=NOW, max_age_days=30)
    assert [r["rev_id"] for r in parked] == ["RAG-DUP-1"]
    assert "集約" in parked[0]["parked_reason"]
    assert [r["rev_id"] for r in kept] == ["RAG-DUP-2"]


def test_other_types_and_reviewed_rules_untouched():
    manual_rule = {"rev_id": "REV-010", "type": "manual", "pending_review": True}
    reviewed_rag = _rag("RAG-DONE", "note-c", "2026-04-01T00:00:00+00:00", pending=False)
    rules = [manual_rule, reviewed_rag]
    kept, parked = cleanup.plan_cleanup(rules, now=NOW, max_age_days=30)
    assert parked == []
    assert kept == rules


def test_fresh_notification_is_kept():
    rules = [_rag("RAG-FRESH", "note-d", "2026-07-04T00:00:00+00:00")]
    kept, parked = cleanup.plan_cleanup(rules, now=NOW, max_age_days=30)
    assert parked == []
    assert [r["rev_id"] for r in kept] == ["RAG-FRESH"]


def test_naive_timestamp_is_treated_as_utc():
    rules = [_rag("RAG-NAIVE", "note-e", "2026-05-01T00:00:00")]
    kept, parked = cleanup.plan_cleanup(rules, now=NOW, max_age_days=30)
    assert [r["rev_id"] for r in parked] == ["RAG-NAIVE"]
