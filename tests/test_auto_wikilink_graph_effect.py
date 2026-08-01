import json

from scripts.auto_wikilink import (
    add_graph_effect_links,
    load_graph_effect_link_targets,
    process_file,
)


def _write_report(path, *, hubs, isolated=(), bridges=()):
    report = {
        "top_effective_hubs": [{"path": p} for p in hubs],
        "isolated_but_used": [{"path": p} for p in isolated],
        "bridge_candidates": [{"path": p} for p in bridges],
    }
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


def test_load_graph_effect_link_targets_maps_isolated_and_bridge_to_hubs(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        hubs=["Hub/Wiki.md", "Hub/Second.md", "Hub/Third.md"],
        isolated=["Notes/Orphan.md"],
        bridges=["Notes/Bridge.md"],
    )

    targets = load_graph_effect_link_targets(report_path)

    assert targets["Notes/Orphan.md"] == ["Hub/Wiki", "Hub/Second"]
    assert targets["Notes/Bridge.md"] == ["Hub/Wiki", "Hub/Second"]


def test_load_graph_effect_link_targets_missing_file_is_noop(tmp_path):
    assert load_graph_effect_link_targets(tmp_path / "missing.json") == {}


def test_load_graph_effect_link_targets_without_hubs_is_noop(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, hubs=[], isolated=["Notes/Orphan.md"])
    assert load_graph_effect_link_targets(report_path) == {}


def test_add_graph_effect_links_appends_missing_hub_links():
    body = "# Orphan\n\n本文。\n"
    targets = {"Notes/Orphan.md": ["Hub/Wiki", "Hub/Second"]}

    new_body, added = add_graph_effect_links(body, "Notes/Orphan.md", targets)

    assert added == 2
    assert "[[Hub/Wiki]]" in new_body
    assert "[[Hub/Second]]" in new_body
    assert "## 関連（判断効果分析による自動リンク）" in new_body


def test_add_graph_effect_links_skips_already_linked_hub():
    body = "# Orphan\n\n[[Hub/Wiki]] を既に参照している。\n"
    targets = {"Notes/Orphan.md": ["Hub/Wiki", "Hub/Second"]}

    new_body, added = add_graph_effect_links(body, "Notes/Orphan.md", targets)

    assert added == 1
    assert new_body.count("[[Hub/Wiki]]") == 1
    assert "[[Hub/Second]]" in new_body


def test_add_graph_effect_links_is_idempotent_across_runs():
    body = "# Orphan\n\n本文。\n"
    targets = {"Notes/Orphan.md": ["Hub/Wiki"]}

    first_body, first_added = add_graph_effect_links(body, "Notes/Orphan.md", targets)
    second_body, second_added = add_graph_effect_links(first_body, "Notes/Orphan.md", targets)

    assert first_added == 1
    assert second_added == 0
    assert second_body == first_body


def test_add_graph_effect_links_noop_for_untargeted_note():
    body = "# Ordinary\n\n本文。\n"
    new_body, added = add_graph_effect_links(body, "Notes/Ordinary.md", {})
    assert added == 0
    assert new_body == body


def test_process_file_applies_graph_effect_links(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Hub").mkdir()
    (vault / "Hub" / "Wiki.md").write_text("# Wiki\n", encoding="utf-8")
    notes = vault / "Notes"
    notes.mkdir()
    orphan = notes / "Orphan.md"
    orphan.write_text("# Orphan\n\n本文。\n", encoding="utf-8")

    result = process_file(
        orphan,
        vault,
        title_index={},
        graph_effect_targets={"Notes/Orphan.md": ["Hub/Wiki"]},
    )

    assert result["graph_effect_links_added"] == 1
    assert result["changes"] == 1
    assert "[[Hub/Wiki]]" in orphan.read_text(encoding="utf-8")
