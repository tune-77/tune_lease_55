import json

import api.shion_memory_decay as decay_module
import api.shion_memory_system_audit as audit_module
from api.shion_memory_system_audit import (
    SHION_MEMORY_SYSTEM_AUDIT_TOOLS,
    audit_memory_freshness_pipeline,
    audit_memory_index_orphans,
    audit_memory_recall_eval_health,
    audit_memory_revision_integrity,
    run_shion_memory_sentinel,
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


def test_run_shion_memory_sentinel_consolidates_memory_reports(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(
        index_path,
        [
            {
                "id": "lt_1",
                "memory_layer": "long_term",
                "status": "active",
                "domain": "credit",
                "use_when": "信用判断を見るとき",
            },
            {
                "id": "lt_2",
                "memory_layer": "long_term",
                "status": "active",
                "domain": "contract",
                "use_when": "",
            },
        ],
    )
    effect_path = tmp_path / "shion_memory_effect_latest.json"
    effect_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-25T01:00:00",
                "summary": {
                    "usage_events": 10,
                    "used_memory_ids": 3,
                    "likely_helpful_memory_ids": 2,
                    "needs_feedback_memory_ids": 1,
                    "possible_noise_memory_ids": 0,
                },
                "needs_feedback": [
                    {
                        "id": "mem_nf_1",
                        "domain": "credit",
                        "memory_layer": "long_term",
                        "memory_type": "judgment_memory",
                        "used_count": 4,
                        "content": "信用判断の確認記憶",
                    },
                    {
                        "id": "mem_nf_2",
                        "domain": "credit",
                        "memory_layer": "long_term",
                        "memory_type": "judgment_memory",
                        "used_count": 2,
                        "content": "銀行支援の確認記憶",
                    },
                ],
                "needs_feedback_triage": {
                    "record_count": 99,
                    "batch_count": 12,
                    "top_batches": [
                        {
                            "domain": "credit",
                            "memory_layer": "long_term",
                            "memory_type": "judgment_memory",
                            "count": 99,
                            "used_count": 200,
                            "samples": [],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    engineering_path = tmp_path / "memory_engineering_latest.json"
    engineering_path.write_text(
        json.dumps(
            {
                "summary": {
                    "open_human_review_records": 120,
                    "open_human_review_batches": 7,
                    "candidate_to_active_pressure": 0.8,
                    "write_policy_metadata_completion_rate": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    contradictions_path = tmp_path / "shion_memory_contradictions_latest.json"
    contradictions_path.write_text(json.dumps({"candidates": []}), encoding="utf-8")
    persistent_path = tmp_path / "persistent_memory_audit_latest.json"
    persistent_path.write_text(json.dumps({"summary": {"findings": 0, "high": 0}}), encoding="utf-8")
    obsidian_path = tmp_path / "obsidian_memory_effectiveness_latest.json"
    obsidian_path.write_text(json.dumps({"summary": {"used": 4, "validated": 1}}), encoding="utf-8")

    monkeypatch.setattr(audit_module, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(audit_module, "_MEMORY_EFFECT_PATH", effect_path)
    monkeypatch.setattr(audit_module, "_MEMORY_ENGINEERING_PATH", engineering_path)
    monkeypatch.setattr(audit_module, "_MEMORY_CONTRADICTIONS_PATH", contradictions_path)
    monkeypatch.setattr(audit_module, "_PERSISTENT_MEMORY_AUDIT_PATH", persistent_path)
    monkeypatch.setattr(audit_module, "_OBSIDIAN_MEMORY_EFFECTIVENESS_PATH", obsidian_path)

    result = run_shion_memory_sentinel(limit=10)

    assert result["mode"] == "shion_memory_sentinel"
    assert result["status"] == "watch"
    assert result["summary"]["memory_records"] == 2
    assert result["summary"]["long_term_domain_coverage"] == 1.0
    assert result["summary"]["long_term_use_when_coverage"] == 0.5
    assert result["summary"]["needs_feedback_memory_ids"] == 1
    assert result["summary"]["open_human_review_records"] == 120
    assert result["summary"]["open_human_review_batches"] == 7
    assert result["feedback_triage"]["record_count"] == 99
    assert result["feedback_triage"]["batch_count"] == 12
    assert result["feedback_triage"]["top_batches"][0]["domain"] == "credit"
    assert result["feedback_triage"]["top_batches"][0]["used_count"] == 200
    assert {signal["area"] for signal in result["signals"]} >= {
        "memory_metadata",
        "usage_effect",
        "memory_engineering",
    }


def test_run_shion_memory_sentinel_flags_noise_as_action_required(tmp_path, monkeypatch):
    index_path = tmp_path / "shion_memory_index.json"
    _write_index(index_path, [])
    effect_path = tmp_path / "shion_memory_effect_latest.json"
    effect_path.write_text(
        json.dumps({"summary": {"possible_noise_memory_ids": 1}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_module, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit_module, "_INDEX_PATH", index_path)
    monkeypatch.setattr(audit_module, "_MEMORY_EFFECT_PATH", effect_path)
    monkeypatch.setattr(audit_module, "_MEMORY_ENGINEERING_PATH", tmp_path / "missing_engineering.json")
    monkeypatch.setattr(audit_module, "_MEMORY_CONTRADICTIONS_PATH", tmp_path / "missing_contradictions.json")
    monkeypatch.setattr(audit_module, "_PERSISTENT_MEMORY_AUDIT_PATH", tmp_path / "missing_persistent.json")
    monkeypatch.setattr(audit_module, "_OBSIDIAN_MEMORY_EFFECTIVENESS_PATH", tmp_path / "missing_obsidian.json")

    result = run_shion_memory_sentinel()

    assert result["status"] == "action_required"
    assert any(signal["area"] == "usage_effect" and signal["level"] == "action_required" for signal in result["signals"])


def test_tools_are_plain_functions_importable_without_adk():
    # api/shion_agent_tools.py が google.adk 未導入環境でも import できるための前提条件
    assert len(SHION_MEMORY_SYSTEM_AUDIT_TOOLS) == 6
    assert run_shion_memory_system_audit in SHION_MEMORY_SYSTEM_AUDIT_TOOLS
    assert run_shion_memory_sentinel in SHION_MEMORY_SYSTEM_AUDIT_TOOLS
    for tool in SHION_MEMORY_SYSTEM_AUDIT_TOOLS:
        assert callable(tool)
