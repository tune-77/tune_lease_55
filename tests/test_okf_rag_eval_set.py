import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = REPO_ROOT / "api" / "knowledge" / "okf_rag_eval_set.json"


def test_okf_rag_eval_set_covers_initial_pack():
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["query"].strip()
        assert case["expected_path_any"]
        assert case["expected_path_any"][0].startswith("knowledge_base/okf_lease_concepts/")

