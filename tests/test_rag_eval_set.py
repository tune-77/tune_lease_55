import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = REPO_ROOT / "api" / "knowledge" / "rag_eval_set.json"


def test_rag_eval_set_has_broad_unique_cases():
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    assert len(cases) >= 25
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    assert len({case["category"] for case in cases}) >= 10


def test_rag_eval_set_cases_have_required_fields():
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    for case in cases:
        assert case["id"].strip()
        assert case["category"].strip()
        assert case["query"].strip()
        assert case["expected_path_any"]
        assert isinstance(case["forbidden_path_any"], list)
