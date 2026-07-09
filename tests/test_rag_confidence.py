"""RAG 信頼度スコア（REV-179）の単体テスト。

confidence_for_hit は rank_score（関連度・ソース優先度込み）70% と
ノート新鮮度（mtime）30% を合成して 0-1 の信頼度とレベルを返す。
"""
import time

import pytest

from api.knowledge.vector_store import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    confidence_for_hit,
)


NOW = 1_800_000_000.0


def _days_ago(days: float) -> float:
    return NOW - days * 86_400


def test_fresh_high_rank_hit_is_high_confidence():
    hit = {"rank_score": 0.95, "metadata": {"mtime": _days_ago(1)}}
    confidence, level = confidence_for_hit(hit, now=NOW)
    assert confidence == pytest.approx(0.7 * 0.95 + 0.3 * 1.0, abs=1e-3)
    assert confidence >= CONFIDENCE_HIGH_THRESHOLD
    assert level == "high"


def test_stale_low_rank_hit_is_low_confidence():
    hit = {"rank_score": 0.2, "metadata": {"mtime": _days_ago(700)}}
    confidence, level = confidence_for_hit(hit, now=NOW)
    assert confidence == pytest.approx(0.7 * 0.2 + 0.3 * 0.2, abs=1e-3)
    assert confidence < CONFIDENCE_MEDIUM_THRESHOLD
    assert level == "low"


def test_mid_range_hit_is_medium_confidence():
    hit = {"rank_score": 0.6, "metadata": {"mtime": _days_ago(120)}}
    confidence, level = confidence_for_hit(hit, now=NOW)
    assert confidence == pytest.approx(0.7 * 0.6 + 0.3 * 0.6, abs=1e-3)
    assert CONFIDENCE_MEDIUM_THRESHOLD <= confidence < CONFIDENCE_HIGH_THRESHOLD
    assert level == "medium"


def test_rank_score_is_clamped_to_unit_range():
    confidence_high, _ = confidence_for_hit(
        {"rank_score": 1.7, "metadata": {"mtime": _days_ago(1)}}, now=NOW
    )
    confidence_low, _ = confidence_for_hit(
        {"rank_score": -0.4, "metadata": {"mtime": _days_ago(1)}}, now=NOW
    )
    assert confidence_high == pytest.approx(1.0, abs=1e-3)
    assert confidence_low == pytest.approx(0.3, abs=1e-3)


def test_falls_back_to_distance_when_rank_score_missing():
    hit = {"distance": 0.25, "metadata": {"mtime": _days_ago(1)}}
    confidence, _ = confidence_for_hit(hit, now=NOW)
    assert confidence == pytest.approx(0.7 * 0.75 + 0.3 * 1.0, abs=1e-3)


def test_falls_back_to_keyword_score_when_no_rank_or_distance():
    hit = {"score": 10, "metadata": {"mtime": _days_ago(1)}}
    confidence, _ = confidence_for_hit(hit, now=NOW)
    assert confidence == pytest.approx(0.7 * 0.5 + 0.3 * 1.0, abs=1e-3)


def test_unknown_mtime_uses_neutral_recency():
    confidence, _ = confidence_for_hit({"rank_score": 0.8}, now=NOW)
    assert confidence == pytest.approx(0.7 * 0.8 + 0.3 * 0.5, abs=1e-3)


def test_top_level_mtime_and_string_mtime_are_accepted():
    top_level, _ = confidence_for_hit(
        {"rank_score": 0.8, "mtime": _days_ago(1)}, now=NOW
    )
    stringified, _ = confidence_for_hit(
        {"rank_score": 0.8, "metadata": {"mtime": str(_days_ago(1))}}, now=NOW
    )
    assert top_level == stringified == pytest.approx(0.7 * 0.8 + 0.3 * 1.0, abs=1e-3)


def test_garbage_mtime_does_not_raise():
    confidence, level = confidence_for_hit(
        {"rank_score": 0.8, "metadata": {"mtime": "not-a-number"}}, now=NOW
    )
    assert confidence == pytest.approx(0.7 * 0.8 + 0.3 * 0.5, abs=1e-3)
    assert level in ("high", "medium", "low")


def test_default_now_uses_current_time():
    hit = {"rank_score": 0.9, "metadata": {"mtime": time.time()}}
    confidence, level = confidence_for_hit(hit)
    assert confidence == pytest.approx(0.7 * 0.9 + 0.3 * 1.0, abs=1e-3)
    assert level == "high"
