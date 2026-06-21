from pathlib import Path

from scripts.validate_okf_subset import build_report, validate_file


def test_okf_subset_sample_pack_is_valid():
    report = build_report([Path("knowledge_base/okf_lease_concepts")])
    assert report["status"] == "ok"
    assert report["checked"] >= 10


def test_okf_subset_requires_type(tmp_path):
    note = tmp_path / "bad.md"
    note.write_text("---\ntitle: Missing type\n---\n\n# Body\n", encoding="utf-8")
    errors = validate_file(note)
    assert "missing required field: type" in errors
