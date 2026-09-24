import json
import sys
import types

from api import shion_obsidian_curator as curator


def _isolate_proposal() -> dict:
    return {
        "type": "connect_used_isolate",
        "path": "リース知識/工作機械の再販リスク.md",
        "reason": "使用シグナルがあるのにリンクが薄い",
        "suggested_links": ["物件残存価値・処分実務"],
        "requires_human_approval": True,
    }


def test_shadow_log_curation_relatedness_noop_when_mode_off(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_CURATION_MODE", raising=False)
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(curator, "_CURATION_SHADOW_LOG", log_path)

    curator._shadow_log_curation_relatedness([_isolate_proposal()])

    assert not log_path.exists()


def test_shadow_log_curation_relatedness_noop_without_shared_context_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_CURATION_MODE", "shadow")
    monkeypatch.delenv("TYPESAFE_ALLOW_SHARED_CONTEXT", raising=False)
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(curator, "_CURATION_SHADOW_LOG", log_path)

    curator._shadow_log_curation_relatedness([_isolate_proposal()])

    assert not log_path.exists()


def test_shadow_log_curation_relatedness_logs_jev_route(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_CURATION_MODE", "shadow")
    monkeypatch.setenv("TYPESAFE_ALLOW_SHARED_CONTEXT", "1")
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(curator, "_CURATION_SHADOW_LOG", log_path)

    fake_module = types.ModuleType("typesafe_rag_guard")
    fake_module.typesafe_rag_enabled = lambda: True

    def fake_judge_passages(query, hits):
        judged = [{**hit, "typesafe_route": "include"} for hit in hits]
        return judged, {"status": "applied", "model": "jev-test", "usage": {}}

    fake_module.judge_passages = fake_judge_passages
    monkeypatch.setitem(sys.modules, "typesafe_rag_guard", fake_module)

    curator._shadow_log_curation_relatedness([_isolate_proposal()])

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["path"] == "リース知識/工作機械の再販リスク.md"
    assert record["jev_routes"] == ["include"]
    assert record["model"] == "jev-test"


def test_shadow_log_curation_relatedness_skips_non_isolate_proposals(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_CURATION_MODE", "shadow")
    monkeypatch.setenv("TYPESAFE_ALLOW_SHARED_CONTEXT", "1")
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(curator, "_CURATION_SHADOW_LOG", log_path)

    fake_module = types.ModuleType("typesafe_rag_guard")
    fake_module.typesafe_rag_enabled = lambda: True
    fake_module.judge_passages = lambda query, hits: (_ for _ in ()).throw(
        AssertionError("should not be called for non-isolate proposals")
    )
    monkeypatch.setitem(sys.modules, "typesafe_rag_guard", fake_module)

    curator._shadow_log_curation_relatedness([
        {"type": "inspect_dormant_isolates", "directory": "Generated", "degree0_count": 3}
    ])

    assert not log_path.exists()
