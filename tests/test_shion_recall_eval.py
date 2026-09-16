"""紫苑記憶想起の評価セットが全件パスすることをCIゲートにする。

評価セット: api/knowledge/shion_recall_eval_set.json
ハーネス:   scripts/eval_shion_memory_recall.py

索引はテスト実行時にリポジトリ内ソース（MEMORY.md / memory/*.md /
knowledge_base/）から組み立てるため、data/ 配下の生成物には依存しない。
"""
from pathlib import Path

from scripts.eval_shion_memory_recall import evaluate_case, load_eval_cases, run_eval


def test_eval_set_is_not_empty():
    cases = load_eval_cases()
    assert len(cases) >= 10
    ids = [c.get("id") for c in cases]
    assert len(ids) == len(set(ids)), "評価ケースIDが重複している"


def test_recall_eval_all_cases_pass():
    results = run_eval()
    failures = [f"{r.case_id}: {r.detail} (recalled={r.recalled_paths})" for r in results if not r.passed]
    assert not failures, "想起評価セットに失敗ケースあり:\n" + "\n".join(failures)


def test_evaluate_case_pins_outcome_signals_against_live_feedback_drift(monkeypatch, tmp_path):
    """本番の judgment_asset_usage_feedback.jsonl 更新に評価結果が左右されないことを固定する（REV-304a/REV-392a）。"""
    import api.shion_memory_recall as recall_mod

    captured = {}

    def fake_recall_memories(query, *, limit=5, index_path=None, outcome_signals=None, **kwargs):
        captured["outcome_signals"] = outcome_signals
        return {"route": "case_screening", "memories": []}

    monkeypatch.setattr(recall_mod, "recall_memories", fake_recall_memories)

    case = {"id": "x", "query": "テスト", "expected_route": "case_screening"}
    evaluate_case(case, index_path=Path(tmp_path / "index.json"))

    assert captured["outcome_signals"] == {}
