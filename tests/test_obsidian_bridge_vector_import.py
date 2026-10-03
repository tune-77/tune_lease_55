"""API と同じ読み方（`mobile_app.obsidian_bridge`）でもベクトル検索の部品を import できること。

2026-05-30〜10-04、`mobile_app/` が sys.path に無いため import に失敗し、キーワード検索だけに落ちていた。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import morning_rag_review_v2 as review

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_enhancements_importable_as_package_module() -> None:
    code = (
        "import sys, pathlib\n"
        "assert str(pathlib.Path('mobile_app').resolve()) not in sys.path\n"
        "import mobile_app.obsidian_bridge as ob\n"
        "enh = ob._enhancements()\n"
        "assert callable(enh.get_vector_store_with_retry) and callable(enh.filter_by_industry)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "SILENT_FAILURE_LOG_PATH": "off"},
    )
    assert result.returncode == 0, result.stderr[-2000:]


def _run(monkeypatch, tmp_path, quality):
    recorded = []
    monkeypatch.setattr(review, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(review, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(review, "DISPATCH_QUEUE", tmp_path / "none.jsonl")
    monkeypatch.setattr(review.RagReviewPhase, "rebuild_index", staticmethod(lambda: True))
    monkeypatch.setattr(review.RagReviewPhase, "test_search_quality", staticmethod(lambda: quality))
    monkeypatch.setattr(review.RagReviewPhase, "analyze_metadata_coverage", staticmethod(lambda: {}))
    monkeypatch.setattr(review.RagReviewPhase, "detect_hot_topics", staticmethod(lambda: []))
    monkeypatch.setattr(review.ImprovementValidator, "verify_improvements", staticmethod(lambda: {"verified": []}))
    monkeypatch.setattr(review, "record_silent_failure", lambda *a, **k: recorded.append(a))
    return review.run_morning_rag_review(), recorded


def test_empty_search_quality_is_failure(monkeypatch, tmp_path) -> None:
    results, recorded = _run(monkeypatch, tmp_path, {})
    assert results["status"] == "failed"
    assert recorded and recorded[0][0] == "answer.morning_rag_review.search_quality"


def test_keyword_only_search_is_failure(monkeypatch, tmp_path) -> None:
    results, recorded = _run(monkeypatch, tmp_path, {"Q-Risk": {"hits": 3, "vector_hits": 0}})
    assert results["status"] == "failed" and recorded


def test_vector_search_used_is_ok(monkeypatch, tmp_path) -> None:
    results, recorded = _run(monkeypatch, tmp_path, {"Q-Risk": {"hits": 3, "vector_hits": 2}})
    assert results["status"] == "ok" and not recorded
