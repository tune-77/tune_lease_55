import json
from pathlib import Path

from scripts import compare_llamaindex_rag


def test_compare_llamaindex_rag_skips_without_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_llamaindex_rag, "_load_llamaindex", lambda: None)

    payload = compare_llamaindex_rag.compare(
        knowledge_dir=Path("knowledge_base/okf_lease_concepts"),
        eval_set=Path("api/knowledge/okf_rag_eval_set.json"),
        top_k=5,
        output_prefix=tmp_path / "llamaindex_rag_comparison_latest",
    )

    assert payload["status"] == "skipped"
    assert "not installed" in payload["reason"]
    assert payload["method"]["guardrail"] == "sidecar_only_no_rag_rank_change_no_prompt_change_no_scoring_no_obsidian_write"
    assert (tmp_path / "llamaindex_rag_comparison_latest.json").exists()
    assert (tmp_path / "llamaindex_rag_comparison_latest.md").exists()

    saved = json.loads((tmp_path / "llamaindex_rag_comparison_latest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "skipped"
