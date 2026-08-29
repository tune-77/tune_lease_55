import os

import pytest

from scripts.eval_obsidian_graph_routing import load_eval_cases, run_eval


def test_obsidian_graph_routing_eval_set_shape():
    cases = load_eval_cases()
    assert len(cases) >= 10
    ids = [case.get("id") for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert str(case.get("query") or "").strip()
        assert case.get("expected_top_path_any")
        assert isinstance(case.get("forbidden_top_path_any"), list)


def test_obsidian_graph_routing_eval_set_passes():
    if os.environ.get("RUN_OBSIDIAN_GRAPH_EVAL") != "1":
        pytest.skip("実Vaultを使う変動性のある統合評価は RUN_OBSIDIAN_GRAPH_EVAL=1 で実行")
    results = run_eval()
    failures = [
        f"{result.case_id}: {result.detail} top={result.top_path} source={result.source}"
        for result in results
        if not result.passed
    ]
    assert not failures, "Obsidian graph routing eval failures:\n" + "\n".join(failures)
