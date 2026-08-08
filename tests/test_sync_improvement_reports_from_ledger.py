import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "improvement-report-sync"
    / "scripts"
    / "sync_improvement_reports.py"
)


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_improvement_reports", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_ledger(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
        encoding="utf-8",
    )


def test_infer_status_ids_from_ledger_uses_group_canonical_key(tmp_path):
    mod = load_sync_module()
    ledger_path = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger_path,
        [
            {
                "key": "misc_done_key",
                "canonical_key": "misc_done_key",
                "status": "applied",
                "title": "修正済みなのに再掲される候補",
                "recorded_at": "2026-07-25T09:00:00",
            }
        ],
    )
    report = {
        "needs_review": [
            {"id": "REV-777", "title": "修正済みなのに再掲される候補", "detail": "旧detail"}
        ],
        "grouped_improvements": [
            {
                "canonical_id": "REV-777",
                "ids": ["REV-777"],
                "canonical_key": "misc_done_key",
                "group_key": "misc_done_key",
            }
        ],
    }

    applied_ids, parked_ids = mod.infer_status_ids_from_ledger(report, {}, ledger_path)

    assert applied_ids == ["REV-777"]
    assert parked_ids == []


def test_from_ledger_moves_applied_and_parked_items(tmp_path):
    mod = load_sync_module()
    ledger_path = tmp_path / "ledger.jsonl"
    write_ledger(
        ledger_path,
        [
            {"key": "REV-001", "status": "applied", "title": "実装済み", "recorded_at": "2026-07-25T09:00:00"},
            {"key": "REV-002", "status": "parked", "title": "監視テーマ", "recorded_at": "2026-07-25T09:00:00"},
        ],
    )
    report = {
        "needs_review": [
            {"id": "REV-001", "title": "実装済み", "file": None, "pr_url": None},
            {"id": "REV-002", "title": "監視テーマ", "file": None, "pr_url": None},
        ],
        "applied": [],
        "summary": {},
    }
    latest = {
        "needs_review": list(report["needs_review"]),
        "applied_improvements": [],
        "items": [
            {"id": "REV-001", "title": "実装済み", "status": "NEEDS_REVIEW"},
            {"id": "REV-002", "title": "監視テーマ", "status": "NEEDS_REVIEW"},
        ],
    }

    applied_ids, parked_ids = mod.infer_status_ids_from_ledger(report, latest, ledger_path)
    updated_report, moved, parked_moved, skipped = mod.sync_report(
        report, applied_ids, [], parked_ids, []
    )
    final_applied_ids = [str(item.get("id") or "") for item in updated_report.get("applied", [])]
    final_parked_ids = [str(item.get("id") or "") for item in updated_report.get("parked", [])]
    updated_latest = mod.sync_latest(latest, final_applied_ids, final_parked_ids)

    assert moved == ["REV-001"]
    assert parked_moved == ["REV-002"]
    assert skipped == []
    assert updated_report["summary"]["needs_review_count"] == 0
    assert [item["id"] for item in updated_report["applied"]] == ["REV-001"]
    assert [item["id"] for item in updated_report["parked"]] == ["REV-002"]
    assert updated_latest["needs_review_count"] == 0
    assert [item["id"] for item in updated_latest["applied_improvements"]] == ["REV-001"]
    assert [item["id"] for item in updated_latest["parked_improvements"]] == ["REV-002"]


def test_report_to_latest_pipeline_does_not_double_bucket_item_matching_both_applied_and_parked(tmp_path):
    """Regression test for the REV-019 (2026-08-08) anomaly: an item whose
    title matches both an applied pattern and a parked_titles pattern (e.g.
    "キャリブレーション") used to be appended to both applied_improvements and
    parked_improvements. sync_latest() no longer resolves title patterns on
    its own — it only projects sync_report()'s already-resolved, mutually
    exclusive applied/parked id sets — so this is now structurally
    impossible rather than merely guarded against.
    """
    mod = load_sync_module()
    report = {
        "needs_review": [
            {
                "id": "REV-019",
                "title": "スコア80-100帯の成約率逆転：モデルキャリブレーション見直し",
                "file": None,
                "pr_url": None,
            }
        ],
        "applied": [],
        "summary": {},
    }
    latest = {
        "needs_review": list(report["needs_review"]),
        "applied_improvements": [],
        "parked_improvements": [],
    }

    updated_report, _moved, _parked_moved, _skipped = mod.sync_report(
        report, ["REV-019"], [], [], ["キャリブレーション"]
    )
    final_applied_ids = [str(item.get("id") or "") for item in updated_report.get("applied", [])]
    final_parked_ids = [str(item.get("id") or "") for item in updated_report.get("parked", [])]
    updated_latest = mod.sync_latest(latest, final_applied_ids, final_parked_ids)

    applied_hit_ids = [item["id"] for item in updated_latest.get("applied_improvements", [])]
    parked_hit_ids = [item["id"] for item in updated_latest.get("parked_improvements", [])]

    assert applied_hit_ids == ["REV-019"]
    assert "REV-019" not in parked_hit_ids
    assert updated_latest["needs_review_count"] == 0
