import json

from scripts import build_obsidian_graph_judgment_effect as graph


def test_graph_report_separates_effective_and_unproven_complexity(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Hub.md").write_text("[[A]] [[B]] [[C]] [[D]] [[E]] [[F]]\n", encoding="utf-8")
    for name in ["A", "B", "C", "D", "E", "F"]:
        (vault / f"{name}.md").write_text("[[Hub]]\n", encoding="utf-8")
    (vault / "UnusedHub.md").write_text("[[U1]] [[U2]] [[U3]] [[U4]] [[U5]] [[U6]]\n", encoding="utf-8")
    for name in ["U1", "U2", "U3", "U4", "U5", "U6"]:
        (vault / f"{name}.md").write_text("[[UnusedHub]]\n", encoding="utf-8")

    report = graph.build_report(
        vault=vault,
        memory_index={"records": []},
        memory_usage_rows=[],
        rag_search_rows=[{"results": [{"obsidian_ref": "[[Hub#x]]", "rank": 1}]}],
        rag_feedback_rows=[{"obsidian_ref": "[[Hub#x]]", "rating": "good"}],
    )

    assert report["summary"]["notes"] == 14
    assert report["top_effective_hubs"][0]["key"] == "Hub"
    assert any(item["key"] == "UnusedHub" for item in report["complex_but_unproven"])


def test_memory_usage_maps_memory_source_path_to_note_key(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Rule.md").write_text("# Rule\n", encoding="utf-8")
    report = graph.build_report(
        vault=vault,
        memory_index={"records": [{"id": "mem_1", "source_path": "Projects/tune_lease_55/Rule.md"}]},
        memory_usage_rows=[{"refs": ["mem_1"]}],
        rag_search_rows=[],
        rag_feedback_rows=[],
    )

    node = next(item for item in report["isolated_but_used"] if item["key"] == "Rule")
    assert node["signals"]["memory_usage_count"] == 1


def test_cli_dry_run_outputs_json(tmp_path, capsys):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "A.md").write_text("# A\n", encoding="utf-8")
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps({"records": []}), encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    assert graph.main([
        "--vault", str(vault),
        "--memory-index", str(idx),
        "--memory-usage", str(empty),
        "--rag-search", str(empty),
        "--rag-feedback", str(empty),
        "--dry-run",
    ]) == 0
    out = capsys.readouterr().out
    assert "notes" in out
