import json

import scripts.check_cloudrun_demo_readiness as readiness
from scripts.check_cloudrun_demo_readiness import CheckRun, check_shion_memory_index


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
