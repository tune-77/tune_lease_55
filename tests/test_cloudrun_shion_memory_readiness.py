import json

import scripts.check_cloudrun_demo_readiness as readiness
from scripts.check_cloudrun_demo_readiness import (
    CheckRun,
    check_memory_review_bundle,
    check_obsidian_retrieval_graph,
    check_shion_memory_index,
)


def _run_with_bundle_index(tmp_path, monkeypatch, content) -> CheckRun:
    bundle_index = tmp_path / "shion_memory_index.json"
    if content is not None:
        bundle_index.write_text(content, encoding="utf-8")
    monkeypatch.setattr(readiness, "BUNDLE_SHION_MEMORY_INDEX", bundle_index)
    monkeypatch.setattr(readiness, "LOCAL_SHION_MEMORY_INDEX", tmp_path / "local_index.json")
    checks = CheckRun()
    check_shion_memory_index(checks)
    return checks


def test_fails_when_bundle_index_missing(tmp_path, monkeypatch):
    checks = _run_with_bundle_index(tmp_path, monkeypatch, None)
    assert any("missing" in msg for msg in checks.failures)


def test_fails_when_bundle_index_empty(tmp_path, monkeypatch):
    checks = _run_with_bundle_index(tmp_path, monkeypatch, json.dumps({"records": []}))
    assert any("0 records" in msg for msg in checks.failures)


def test_passes_with_enough_records(tmp_path, monkeypatch):
    records = [{"id": f"mem_{i}", "content": "x"} for i in range(120)]
    checks = _run_with_bundle_index(tmp_path, monkeypatch, json.dumps({"records": records}))
    assert not checks.failures
    assert any("120 records" in msg for msg in checks.infos)


def _run_with_bundle_graph(tmp_path, monkeypatch, content) -> CheckRun:
    graph_path = tmp_path / "obsidian_retrieval_graph.json"
    if content is not None:
        graph_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(readiness, "BUNDLE_OBSIDIAN_RETRIEVAL_GRAPH", graph_path)
    checks = CheckRun()
    check_obsidian_retrieval_graph(checks)
    return checks


def test_fails_when_bundle_obsidian_graph_missing(tmp_path, monkeypatch):
    checks = _run_with_bundle_graph(tmp_path, monkeypatch, None)
    assert any("graph pre-routing will be disabled" in msg for msg in checks.failures)


def test_fails_when_bundle_obsidian_graph_empty(tmp_path, monkeypatch):
    checks = _run_with_bundle_graph(tmp_path, monkeypatch, json.dumps({"nodes": []}))
    assert any("0 nodes" in msg for msg in checks.failures)


def test_passes_with_enough_obsidian_graph_nodes(tmp_path, monkeypatch):
    nodes = [{"path": f"note-{i}.md", "summary": "x"} for i in range(120)]
    checks = _run_with_bundle_graph(tmp_path, monkeypatch, json.dumps({"nodes": nodes}))
    assert not checks.failures
    assert any("120 nodes" in msg for msg in checks.infos)


def test_memory_review_bundle_fails_without_report_and_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "BUNDLE_MEMORY_ENGINEERING_REPORT", tmp_path / "memory_engineering_latest.json")
    monkeypatch.setattr(readiness, "BUNDLE_MEMORY_REVIEW_SOURCES", [tmp_path / "missing.jsonl"])

    checks = CheckRun()
    check_memory_review_bundle(checks)

    assert any("memory_engineering_latest.json is missing" in msg for msg in checks.failures)
    assert any("no Memory Review JSONL sources" in msg for msg in checks.failures)


def test_memory_review_bundle_passes_with_report_and_source_rows(tmp_path, monkeypatch):
    report_path = tmp_path / "memory_engineering_latest.json"
    report_path.write_text(json.dumps({"summary": {"write_path_records": 1}}), encoding="utf-8")
    source_path = tmp_path / "judgment_materials_preview.jsonl"
    source_path.write_text(json.dumps({"id": "a1", "claim": "候補"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(readiness, "BUNDLE_MEMORY_ENGINEERING_REPORT", report_path)
    monkeypatch.setattr(readiness, "BUNDLE_MEMORY_REVIEW_SOURCES", [source_path])

    checks = CheckRun()
    check_memory_review_bundle(checks)

    assert not checks.failures
    assert any("memory_engineering_latest.json is present" in msg for msg in checks.infos)
    assert any("1 rows" in msg for msg in checks.infos)
