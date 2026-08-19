from pathlib import Path

from scripts.obsidian_taxonomy_audit import build_taxonomy_audit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_detects_case_and_separator_tag_duplicates(tmp_path):
    vault = tmp_path
    _write(vault / "A" / "note1.md", "---\ntags: [AI, 機械学習]\n---\nbody")
    _write(vault / "a" / "note2.md", "---\ntags: [ai, 機械学習]\n---\nbody")
    _write(vault / "note3.md", "---\ntags: [tag-test, tag_test]\n---\nbody")

    report = build_taxonomy_audit(vault)

    tag_labels = {label for cluster in report["tag_duplicate_clusters"] for label in cluster["labels"]}
    assert {"AI", "ai"} <= tag_labels
    assert {"tag-test", "tag_test"} <= tag_labels
    folder_labels = {label for cluster in report["folder_duplicate_clusters"] for label in cluster["labels"]}
    assert {"A", "a"} <= folder_labels
    assert report["naming_rules"]
    assert len(report["naming_rules"]) <= 5


def test_no_duplicates_yields_generic_rule(tmp_path):
    vault = tmp_path
    _write(vault / "note1.md", "---\ntags: [固有タグ]\n---\nbody")

    report = build_taxonomy_audit(vault)

    assert report["tag_duplicate_clusters"] == []
    assert report["folder_duplicate_clusters"] == []
    assert report["naming_rules"] == [
        "既存タグ・フォルダ名の一覧を作成し、新規追加前に重複がないか確認する運用を続ける"
    ]


def test_excludes_operational_folders(tmp_path):
    vault = tmp_path
    _write(vault / "Private Reflection" / "2026-08-19.md", "---\ntags: [内省]\n---\nbody")
    _write(vault / "Dialogue" / "2026-08-19.md", "---\ntags: [対話]\n---\nbody")

    report = build_taxonomy_audit(vault)

    assert report["totals"]["distinct_tags"] == 0
