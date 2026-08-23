import json

import api.shion_memory_decay as decay_module
import api.shion_memory_system_audit as audit_module
from api.shion_memory_system_audit import (
    SHION_MEMORY_SYSTEM_AUDIT_TOOLS,
    audit_memory_freshness_pipeline,
    audit_memory_index_orphans,
    audit_memory_recall_eval_health,
    audit_memory_revision_integrity,
    run_shion_memory_system_audit,
)


def _write_index(path, records):
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def test_audit_memory_index_orphans_reports_missing_source(tmp_path, monkeypatch):
    existing_source = tmp_path / "MEMORY.md"
    existing_source.write_text("# memo", encoding="utf-8")
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(
        index_path,
        [
            {"id": "mem_a", "content": "生きている記憶", "memory_type": "factual_memory", "status": "active", "source_path": "MEMORY.md"},
            {"id": "mem_b", "content": "消えたソース", "memory_type": "technical_memory", "status": "active", "source_path": "memory/9999-99-99.md"},
            {"id": "mem_c", "content": "非推奨記憶", "memory_type": "factual_memory", "status": "deprecated", "source_path": "MEMORY.md"},
        ],
    )
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(audit_module, "_REPO_ROOT", tmp_path)

    result = audit_memory_index_orphans()

    assert result["status"] == "ok"
    assert result["total_records"] == 3
    assert result["orphan_count"] == 1
    assert result["orphan_samples"][0]["id"] == "mem_b"
    assert result["status_breakdown"] == {"active": 2, "deprecated": 1}


def test_audit_memory_index_orphans_missing_index_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_module, "_INDEX_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setattr(audit_module, "_REPO_ROOT", tmp_path)

    result = audit_memory_index_orphans()

    assert result["status"] == "index_not_found"


def test_audit_memory_freshness_pipeline_in_sync(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(
        index_path,
        [
            {"id": "mem_a", "status": "active"},
            {"id": "mem_b", "status": "active"},
            {"id": "mem_c", "status": "deprecated"},
        ],
    )
    freshness_path = tmp_path / "shion_memory_freshness.jsonl"
    freshness_path.write_text(
        json.dumps({"snapshot_at": "2026-08-01T04:00:00", "total": 2, "at_risk_count": 0, "records": []}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(decay_module, "_FRESHNESS_PATH", freshness_path)

    result = audit_memory_freshness_pipeline()

    # decay batch skips deprecated records, so 2 non-deprecated == snapshot total 2
    assert result["index_total"] == 3
    assert result["index_deprecated_count"] == 1
    assert result["snapshot_total"] == 2
    assert result["record_count_drift"] == 0
    assert result["status"] == "in_sync"


def test_audit_memory_freshness_pipeline_detects_drift(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(index_path, [{"id": "mem_a", "status": "active"}, {"id": "mem_b", "status": "active"}])
    freshness_path = tmp_path / "shion_memory_freshness.jsonl"
    # snapshot says only 1 record, but index now has 2 non-deprecated records -> drift
    freshness_path.write_text(
        json.dumps({"snapshot_at": "2026-08-01T04:00:00", "total": 1, "at_risk_count": 0, "records": []}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(decay_module, "_FRESHNESS_PATH", freshness_path)

    result = audit_memory_freshness_pipeline()

    assert result["record_count_drift"] == 1
    assert result["status"] == "drifted"


def test_audit_memory_freshness_pipeline_batch_never_ran(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(index_path, [{"id": "mem_a", "status": "active"}])
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(decay_module, "_FRESHNESS_PATH", tmp_path / "no_snapshot.jsonl")

    result = audit_memory_freshness_pipeline()

    assert result["snapshot_status"] == "no_snapshot_yet"
    assert result["status"] == "batch_never_ran"


def test_audit_memory_revision_integrity_detects_pending(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(
        index_path,
        [{"id": "mem_old", "content": "旧結論", "memory_type": "factual_memory", "status": "active", "source_path": "MEMORY.md"}],
    )
    revisions_path = tmp_path / "shion_memory_revisions.jsonl"
    revisions_path.write_text(
        json.dumps({"ts": "2026-08-01T00:00:00", "old_id": "mem_old", "reason": "制度改定", "new_content": "新結論"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(audit_module, "_REVISIONS_PATH", revisions_path)

    result = audit_memory_revision_integrity()

    assert result["total_revisions"] == 1
    # index was never rebuilt after the revision was declared -> still pending
    assert result["pending_count"] >= 1
    assert result["status"] == "pending_index_rebuild"


def test_audit_memory_revision_integrity_no_revisions_is_ok(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(index_path, [{"id": "mem_a", "status": "active"}])
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(audit_module, "_REVISIONS_PATH", tmp_path / "no_revisions.jsonl")

    result = audit_memory_revision_integrity()

    assert result["status"] == "ok"
    assert result["total_revisions"] == 0
    assert result["pending_count"] == 0


def test_audit_memory_recall_eval_health_reads_real_eval_set():
    result = audit_memory_recall_eval_health()

    assert result["harness_script_found"] is True
    assert result["eval_case_count"] > 0
    assert result["status"] == "ok"


def test_run_shion_memory_system_audit_aggregates_all_checks(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(index_path, [{"id": "mem_a", "status": "active", "source_path": "MEMORY.md"}])
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(audit_module, "_REVISIONS_PATH", tmp_path / "no_revisions.jsonl")
    monkeypatch.setattr(decay_module, "_FRESHNESS_PATH", tmp_path / "no_snapshot.jsonl")
    monkeypatch.setattr(audit_module, "_REPO_ROOT", tmp_path)

    result = run_shion_memory_system_audit()

    assert result["mode"] == "shion_memory_system_audit"
    assert set(result.keys()) >= {
        "index_orphans",
        "freshness_pipeline",
        "revision_integrity",
        "recall_eval_health",
        "issue_count",
    }
    assert isinstance(result["issue_count"], int)


def test_tools_are_plain_functions_importable_without_adk():
    # api/shion_agent_tools.py が google.adk 未導入環境でも import できるための前提条件
    assert len(SHION_MEMORY_SYSTEM_AUDIT_TOOLS) == 5
    assert run_shion_memory_system_audit in SHION_MEMORY_SYSTEM_AUDIT_TOOLS
    for tool in SHION_MEMORY_SYSTEM_AUDIT_TOOLS:
        assert callable(tool)
