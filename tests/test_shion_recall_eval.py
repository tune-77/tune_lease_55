"""紫苑記憶想起の評価セットが全件パスすることをCIゲートにする。

評価セット: api/knowledge/shion_recall_eval_set.json
ハーネス:   scripts/eval_shion_memory_recall.py

索引はテスト実行時にリポジトリ内ソース（MEMORY.md / memory/*.md /
knowledge_base/）から組み立てるため、data/ 配下の生成物には依存しない。
"""
from scripts.eval_shion_memory_recall import load_eval_cases, run_eval


def test_eval_set_is_not_empty():
    cases = load_eval_cases()
    assert len(cases) >= 10
    ids = [c.get("id") for c in cases]
    assert len(ids) == len(set(ids)), "評価ケースIDが重複している"


def test_recall_eval_all_cases_pass():
    results = run_eval()
    failures = [f"{r.case_id}: {r.detail} (recalled={r.recalled_paths})" for r in results if not r.passed]
    assert not failures, "想起評価セットに失敗ケースあり:\n" + "\n".join(failures)
