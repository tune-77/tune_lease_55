"""Tests for the optional TypeSafe/Jev asset-score estimation.

Every test injects a fake ``request_fn`` or leaves the feature disabled, so the
suite never contacts TypeSafe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import typesafe_asset_guard as guard  # noqa: E402


def _asset(**overrides: Any) -> dict[str, Any]:
    base = {
        "category": "車両",
        "asset_name": "大型トラック",
        "asset_detail": "大型トラック 2台",
        "lease_term_months": 60,
    }
    base.update(overrides)
    return base


def _answers(score: float, confidence: float = 0.8) -> dict[str, Any]:
    """Build a response answering every dimension of the 車両 category."""
    items = guard.resolve_items("車両")
    return {
        "model": "jev-1.13.0",
        "answers": {
            f"item_{item['id']}": {
                "type": "score",
                "score": score,
                "confidence": confidence,
            }
            for item in items
        },
        "usage": {"tokens": 1},
    }


# --- request construction -------------------------------------------------


def test_resolve_items_rejects_unknown_category():
    with pytest.raises(guard.TypeSafeAssetError):
        guard.resolve_items("宇宙船")


def test_request_covers_every_dimension_once():
    items = guard.resolve_items("産業機械")
    payload = guard.build_asset_request(_asset(category="産業機械"), items)

    assert set(payload["questions"]) == {f"item_{item['id']}" for item in items}
    for question in payload["questions"].values():
        assert question["type"] == "score"
        assert len(question["criteria"]) == len(guard.LEVEL_LADDER)


def test_payload_carries_only_item_identification():
    """Company names and financial figures must not leave the machine."""
    asset = _asset(
        company_name="株式会社サンプル",
        applicant_id="A-0001",
        sales=120000,
        bank_credit=45000,
        memo="担当者メモ",
    )
    payload = guard.build_asset_request(asset, guard.resolve_items("車両"))

    assert set(payload["state"]["asset"]) <= {
        "category",
        "name",
        "detail",
        "maker",
        "lease_term_months",
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    for leaked in ("サンプル", "A-0001", "120000", "45000", "担当者メモ"):
        assert leaked not in serialized


def test_long_free_text_is_truncated():
    asset = _asset(asset_detail="あ" * 5000, asset_name="い" * 5000)
    payload = guard.build_asset_request(asset, guard.resolve_items("車両"))

    assert len(payload["state"]["asset"]["detail"]) == guard.MAX_DETAIL_CHARS
    assert len(payload["state"]["asset"]["name"]) == guard.MAX_NAME_CHARS


# --- rescaling ------------------------------------------------------------


def test_rescale_spans_the_full_range():
    levels = len(guard.LEVEL_LADDER)
    assert guard._rescale(0, levels) == 0.0
    assert guard._rescale(levels - 1, levels) == 100.0


def test_top_level_maps_to_100():
    scores, meta = guard.judge_asset(
        _asset(), request_fn=lambda _: _answers(len(guard.LEVEL_LADDER) - 1)
    )

    assert meta["status"] == "applied"
    assert set(scores) == {item["id"] for item in guard.resolve_items("車両")}
    assert all(value == 100.0 for value in scores.values())


# --- response validation --------------------------------------------------


def test_score_outside_the_ladder_is_rejected():
    with pytest.raises(guard.TypeSafeAssetError):
        guard.judge_asset(_asset(), request_fn=lambda _: _answers(99.0))


def test_missing_answer_is_rejected():
    def _request(_payload: dict[str, Any]) -> dict[str, Any]:
        body = _answers(2.0)
        body["answers"].popitem()
        return body

    with pytest.raises(guard.TypeSafeAssetError):
        guard.judge_asset(_asset(), request_fn=_request)


def test_low_confidence_is_reported_not_suppressed():
    scores, meta = guard.judge_asset(
        _asset(), request_fn=lambda _: _answers(2.0, confidence=0.05)
    )

    assert scores, "a low-confidence answer is still returned"
    assert set(meta["low_confidence_items"]) == set(scores)


# --- fail-open ------------------------------------------------------------


def test_disabled_returns_empty_so_calc_asset_score_uses_50(monkeypatch):
    monkeypatch.delenv("TYPESAFE_ASSET_ENABLED", raising=False)
    scores, meta = guard.judge_asset_if_enabled(_asset())

    assert scores == {}
    assert meta["status"] == "disabled"


def test_transport_failure_falls_back_to_empty():
    def _boom(_payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("connection reset")

    scores, meta = guard.judge_asset_if_enabled(_asset(), request_fn=_boom)

    assert scores == {}
    assert meta == {"status": "fallback", "error_type": "RuntimeError"}


def test_fallback_reproduces_current_default_behavior():
    """An empty mapping must give exactly today's all-50 result."""
    from asset_scorer import calc_asset_score

    scores, _ = guard.judge_asset_if_enabled(
        _asset(), request_fn=lambda _: (_ for _ in ()).throw(ValueError("bad"))
    )
    fallback = calc_asset_score("車両", scores)
    current = calc_asset_score("車両", {})

    assert fallback["total_score"] == current["total_score"]
    assert fallback["grade"] == current["grade"]


def test_enabled_requires_both_flag_and_key():
    assert not guard.typesafe_asset_enabled({"TYPESAFE_ASSET_ENABLED": "1"})
    assert not guard.typesafe_asset_enabled({"TYPESAFE_API_KEY": "sk-test"})
    assert guard.typesafe_asset_enabled(
        {"TYPESAFE_ASSET_ENABLED": "1", "TYPESAFE_API_KEY": "sk-test"}
    )
