from __future__ import annotations

import importlib.util
from pathlib import Path

from scripts import improvement_consolidator


ROOT = Path(__file__).resolve().parents[1]


def _load_extract_module():
    spec = importlib.util.spec_from_file_location(
        "extract_obsidian_improvements_semantic_test",
        ROOT / "scripts" / "extract_obsidian_improvements.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_small_candidate_set_skips_gemini(monkeypatch) -> None:
    items = [{"title": f"改善{i}", "reason": "理由"} for i in range(15)]
    monkeypatch.setattr(
        improvement_consolidator,
        "_get_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("Gemini must not be called")),
    )

    assert improvement_consolidator.consolidate_with_ai(items) is items


def test_semantic_dedup_merges_only_high_confidence_pair() -> None:
    module = _load_extract_module()
    items = [
        {"tag": "改善", "title": "EDINET連携 Phase2", "reason": "機能を強化", "duplicate_count": 2},
        {"tag": "改善", "title": "EDINET連携の高度化", "reason": "情報を拡充", "duplicate_count": 1},
        {"tag": "改善", "title": "表示ラベル修正", "reason": "誤字", "duplicate_count": 1},
    ]

    merged, meta = module.semantic_deduplicate_improvements(
        items,
        request_fn=lambda _payload: {
            "model": "jev-test",
            "answers": {"pair0_same_issue": {"type": "noul", "noul": 0.88}},
            "usage": {"input_tokens": 50},
        },
    )

    assert len(merged) == 2
    assert merged[0]["duplicate_count"] == 3
    assert merged[0]["semantic_duplicate_titles"] == ["EDINET連携の高度化"]
    assert meta["auto_merged_pairs"] == 1


def test_semantic_dedup_fails_open() -> None:
    module = _load_extract_module()
    items = [
        {"tag": "改善", "title": "EDINET連携 Phase2", "reason": "機能を強化"},
        {"tag": "改善", "title": "EDINET連携の高度化", "reason": "情報を拡充"},
    ]

    result, meta = module.semantic_deduplicate_improvements(
        items,
        request_fn=lambda _payload: (_ for _ in ()).throw(RuntimeError("timeout")),
    )

    assert result == items
    assert meta["status"] == "fallback"


def test_semantic_dedup_does_not_merge_inconsistent_chain(monkeypatch) -> None:
    module = _load_extract_module()
    items = [
        {"tag": "改善", "title": "ABCDEF", "reason": "A"},
        {"tag": "改善", "title": "ABCDEG", "reason": "B"},
        {"tag": "改善", "title": "ABCDGH", "reason": "C"},
    ]
    monkeypatch.setattr(
        module,
        "_jaccard_similarity",
        lambda _a, _b: 0.40,
    )

    result, meta = module.semantic_deduplicate_improvements(
        items,
        request_fn=lambda _payload: {
            "answers": {
                "pair0_same_issue": {"type": "noul", "noul": 0.9},
                "pair1_same_issue": {"type": "noul", "noul": 0.1},
                "pair2_same_issue": {"type": "noul", "noul": 0.9},
            }
        },
    )

    assert [row["title"] for row in result] == [row["title"] for row in items]
    assert all(row["duplicate_count"] == 1 for row in result)
    assert meta["auto_merged_pairs"] == 0
    assert meta["inconsistent_components"] == 1
