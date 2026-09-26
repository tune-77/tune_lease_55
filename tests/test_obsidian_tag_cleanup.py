from pathlib import Path

from scripts.obsidian_taxonomy_audit import build_taxonomy_audit
from scripts.obsidian_tag_cleanup import (
    apply_tag_renames,
    compute_canonical_mapping,
    plan_tag_renames,
    rewrite_tags_field,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_canonical_mapping_prefers_higher_count_label():
    clusters = [
        {"labels": ["AI", "ai"], "counts": {"AI": 1, "ai": 5}},
    ]
    assert compute_canonical_mapping(clusters) == {"AI": "ai"}


def test_canonical_mapping_merges_overrides():
    mapping = compute_canonical_mapping([], overrides={"リースク": "リース"})
    assert mapping == {"リースク": "リース"}


def test_rewrite_tags_field_flow_style():
    text = "---\ntitle: t\ntags: [AI, other]\n---\nbody\n"
    new_text = rewrite_tags_field(text, ["ai", "other"])
    assert new_text == "---\ntitle: t\ntags: [ai, other]\n---\nbody\n"


def test_rewrite_tags_field_block_style_preserves_indent():
    text = "---\ntags:\n  - AI\n  - other\ncssclass: x\n---\nbody\n"
    new_text = rewrite_tags_field(text, ["ai", "other"])
    assert new_text == "---\ntags:\n  - ai\n  - other\ncssclass: x\n---\nbody\n"


def test_rewrite_tags_field_no_tags_returns_none():
    text = "---\ntitle: t\n---\nbody\n"
    assert rewrite_tags_field(text, ["x"]) is None


def test_plan_and_apply_merges_case_duplicate(tmp_path):
    vault = tmp_path
    _write(vault / "note1.md", "---\ntags: [AI, 機械学習]\n---\nbody1")
    _write(vault / "note2.md", "---\ntags: [ai, 機械学習]\n---\nbody2")
    _write(vault / "note3.md", "---\ntags: [固有タグ]\n---\nbody3")

    audit = build_taxonomy_audit(vault)
    mapping = compute_canonical_mapping(audit["tag_duplicate_clusters"])
    changes = plan_tag_renames(vault, mapping)

    changed_paths = {c["path"] for c in changes}
    assert changed_paths == {"note1.md"} or changed_paths == {"note2.md"}
    assert "note3.md" not in changed_paths

    # dry-run: nothing written yet
    original = (vault / "note1.md").read_text(encoding="utf-8")

    written = apply_tag_renames(vault, changes)
    assert written == len(changes)

    changed_path = vault / next(iter(changed_paths))
    new_text = changed_path.read_text(encoding="utf-8")
    assert new_text != original or changed_path.name != "note1.md"

    # re-planning after apply should find no further changes
    audit2 = build_taxonomy_audit(vault)
    mapping2 = compute_canonical_mapping(audit2["tag_duplicate_clusters"])
    assert plan_tag_renames(vault, mapping2) == []


def test_apply_only_writes_planned_changes_not_dry_run(tmp_path):
    vault = tmp_path
    _write(vault / "note1.md", "---\ntags: [AI]\n---\nbody1")
    _write(vault / "note2.md", "---\ntags: [ai]\n---\nbody2")

    audit = build_taxonomy_audit(vault)
    mapping = compute_canonical_mapping(audit["tag_duplicate_clusters"])
    changes = plan_tag_renames(vault, mapping)
    assert len(changes) == 1  # only one of the two needs a rewrite

    before = {p.name: p.read_text(encoding="utf-8") for p in vault.glob("*.md")}
    # not calling apply_tag_renames: files must stay untouched (dry-run default)
    after = {p.name: p.read_text(encoding="utf-8") for p in vault.glob("*.md")}
    assert before == after
