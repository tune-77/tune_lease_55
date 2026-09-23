"""Tests for the optional TypeSafe/Jev judgment-asset quality guard.

Every test injects a fake ``request_fn`` or leaves the feature disabled, so the
suite never contacts TypeSafe.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import typesafe_judgment_quality_guard as guard

ROOT = Path(__file__).resolve().parents[1]


def _load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "build_autoresearch_candidates_for_quality_test",
        ROOT / "scripts" / "build_autoresearch_judgment_asset_candidates.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _claims():
    return [
        {"candidate_type": "caution", "claim": "残価だけで承認可否を決めつけない。"},
        {"candidate_type": "condition_signal", "claim": "補助金の交付決定前に検収が前倒しされている場合は保全を見直す。"},
    ]


def _answer(choice: str, confidence: float, probabilities=None):
    payload = {"type": "choice", "choice": choice, "confidence": confidence}
    if probabilities is not None:
        payload["probabilities"] = probabilities
    return payload


# --- privacy screen -------------------------------------------------------

def test_privacy_screen_rejects_case_and_pii_text():
    assert not guard.is_safe_public_claim({"claim": "株式会社サンプルの与信を見直す。"})
    assert not guard.is_safe_public_claim({"claim": "案件番号 A-12345 を確認する。"})
    assert not guard.is_safe_public_claim({"claim": "保証人の資力を確認する。"})


def test_privacy_screen_keeps_published_thresholds():
    """Money figures are the substance of policy claims and must not be dropped."""
    assert guard.is_safe_public_claim({"claim": "補助金上限は1,000万円に引き上げられた。"})


def test_filter_safe_claims_reports_skipped_count():
    claims = _claims() + [{"candidate_type": "caution", "claim": "株式会社サンプルの件。"}]
    safe, skipped = guard.filter_safe_claims(claims)
    assert safe == [0, 1]
    assert skipped == 1


# --- request construction -------------------------------------------------

def test_request_refuses_an_incomplete_class_catalog(monkeypatch):
    monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, "actionable", "")
    with pytest.raises(guard.TypeSafeJudgmentQualityError):
        guard.build_quality_request(_claims())


def test_request_sends_one_question_per_claim(monkeypatch):
    for name in guard.JUDGMENT_QUALITY_CLASSES:
        monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, name, f"{name} の定義")
    payload = guard.build_quality_request(_claims(), model="jev-test")

    assert payload["model"] == "jev-test"
    assert set(payload["questions"]) == {"claim0_quality", "claim1_quality"}
    assert set(payload["questions"]["claim0_quality"]["criteria"]) == set(
        guard.JUDGMENT_QUALITY_CLASSES
    )
    # Only the claim text and its type may leave the machine.
    assert set(payload["state"]["claims"][0]) == {"candidate_type", "claim"}


# --- routing policy -------------------------------------------------------

def test_route_requires_both_confidence_and_margin():
    assert guard.route_quality("actionable", 0.95, 0.80) == "auto"
    assert guard.route_quality("actionable", 0.70, 0.80) == "review"
    assert guard.route_quality("actionable", 0.95, 0.10) == "review"


def test_route_auto_allows_missing_distribution():
    """No probabilities means the margin rule cannot fire; confidence still gates."""
    assert guard.route_quality("actionable", 0.95, None) == "auto"


def test_needs_context_always_goes_to_review():
    assert guard.route_quality("needs_context", 0.99, 0.99) == "review"


# --- judging --------------------------------------------------------------

def test_judge_claims_returns_indices_and_routes(monkeypatch):
    for name in guard.JUDGMENT_QUALITY_CLASSES:
        monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, name, f"{name} の定義")
    judged, meta = guard.judge_claims(
        _claims(),
        request_fn=lambda _payload: {
            "model": "jev-test",
            "answers": {
                "claim0_quality": _answer(
                    "textbook_general", 0.91, {"textbook_general": 0.91, "actionable": 0.06, "needs_context": 0.03}
                ),
                "claim1_quality": _answer(
                    "actionable", 0.62, {"actionable": 0.62, "needs_context": 0.30, "textbook_general": 0.08}
                ),
            },
            "usage": {"input_tokens": 40, "output_tokens": 8},
        },
    )

    assert [item["index"] for item in judged] == [0, 1]
    assert judged[0]["decision"] == "auto"
    assert judged[1]["decision"] == "review"
    assert meta["status"] == "applied"
    assert meta["auto_count"] == 1
    assert meta["review_count"] == 1


def test_judge_claims_rejects_an_unknown_class(monkeypatch):
    for name in guard.JUDGMENT_QUALITY_CLASSES:
        monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, name, f"{name} の定義")
    with pytest.raises(guard.TypeSafeJudgmentQualityError):
        guard.judge_claims(
            _claims()[:1],
            request_fn=lambda _payload: {
                "answers": {"claim0_quality": _answer("something_else", 0.99)}
            },
        )


def test_judge_claims_if_enabled_sends_nothing_when_disabled(monkeypatch):
    monkeypatch.delenv("TYPESAFE_JUDGMENT_QUALITY_ENABLED", raising=False)
    judged, meta = guard.judge_claims_if_enabled(_claims())
    assert judged == []
    assert meta == {"status": "disabled"}


def test_judge_claims_if_enabled_falls_back_on_error(monkeypatch):
    for name in guard.JUDGMENT_QUALITY_CLASSES:
        monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, name, f"{name} の定義")

    def _boom(_payload):
        raise RuntimeError("network down")

    judged, meta = guard.judge_claims_if_enabled(_claims(), request_fn=_boom)
    assert judged == []
    assert meta["status"] == "fallback"
    assert meta["error_type"] == "RuntimeError"


def test_an_unwritten_class_catalog_fails_open_and_never_sends(monkeypatch):
    """The catalog guard fires before any request, so an unfinished catalog is safe."""
    monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, "needs_context", "")
    sent = []

    judged, meta = guard.judge_claims_if_enabled(
        _claims(), request_fn=lambda payload: sent.append(payload) or {}
    )
    assert sent == []
    assert judged == []
    assert meta == {
        "status": "fallback",
        "error_type": "TypeSafeJudgmentQualityError",
    }


# --- integration with the candidate builder -------------------------------

def _candidate(claim: str, *, asset_quality: str, promotion_status: str):
    return {
        "id": claim,
        "candidate_type": "caution",
        "claim": claim,
        "asset_quality": asset_quality,
        "quality_reasons": [],
        "quality_source": "rule",
        "quality_decision": "rule_only",
        "needs_review": False,
        "promotion_status": promotion_status,
        "verified_status": "unverified",
        "use_count": 0,
        "useful_count": 0,
        "rejected_count": 0,
        "neutral_count": 0,
        "edit_count": 0,
    }


def test_disabled_pass_leaves_every_rule_verdict_untouched():
    module = _load_builder_module()
    candidates = [_candidate("残価を過信しない。", asset_quality="actionable", promotion_status="not_promoted")]
    meta = module.apply_typesafe_quality(candidates)

    assert meta["status"] == "disabled"
    assert candidates[0]["asset_quality"] == "actionable"
    assert candidates[0]["quality_source"] == "rule"
    assert candidates[0]["needs_review"] is False


def test_review_rescues_a_candidate_the_rule_suppressed(monkeypatch):
    """The whole point: an unresolved claim reaches a human instead of vanishing."""
    module = _load_builder_module()
    for name in guard.JUDGMENT_QUALITY_CLASSES:
        monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, name, f"{name} の定義")
    candidates = [
        _candidate(
            "補助金の交付決定前に検収が前倒しされていないか見る。",
            asset_quality="textbook_general",
            promotion_status="not_promoted_textbook_general",
        )
    ]
    meta = module.apply_typesafe_quality(
        candidates,
        request_fn=lambda _payload: {
            "answers": {"claim0_quality": _answer("actionable", 0.60, {"actionable": 0.60, "needs_context": 0.35, "textbook_general": 0.05})}
        },
    )

    assert meta["needs_review"] == 1
    assert candidates[0]["needs_review"] is True
    assert candidates[0]["promotion_status"] == "needs_review_quality"
    # A review judgment must not rewrite the verdict itself.
    assert candidates[0]["asset_quality"] == "textbook_general"


def test_auto_judgment_rewrites_the_verdict_and_status(monkeypatch):
    module = _load_builder_module()
    for name in guard.JUDGMENT_QUALITY_CLASSES:
        monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, name, f"{name} の定義")
    candidates = [
        _candidate(
            "補助金の交付決定前に検収が前倒しされていないか見る。",
            asset_quality="textbook_general",
            promotion_status="not_promoted_textbook_general",
        )
    ]
    meta = module.apply_typesafe_quality(
        candidates,
        request_fn=lambda _payload: {
            "answers": {"claim0_quality": _answer("actionable", 0.96, {"actionable": 0.96, "needs_context": 0.03, "textbook_general": 0.01})}
        },
    )

    assert meta["quality_changed"] == 1
    assert candidates[0]["asset_quality"] == "actionable"
    assert candidates[0]["needs_review"] is False
    assert candidates[0]["promotion_status"] == "not_promoted"


def test_human_decided_statuses_are_never_rewritten(monkeypatch):
    module = _load_builder_module()
    for name in guard.JUDGMENT_QUALITY_CLASSES:
        monkeypatch.setitem(guard.JUDGMENT_QUALITY_CLASSES, name, f"{name} の定義")
    candidates = [
        _candidate("残価を過信しない。", asset_quality="actionable", promotion_status="ready_for_promotion"),
        _candidate("与信は総合判断である。", asset_quality="actionable", promotion_status="promoted"),
    ]
    module.apply_typesafe_quality(
        candidates,
        request_fn=lambda _payload: {
            "answers": {
                "claim0_quality": _answer("textbook_general", 0.99, {"textbook_general": 0.99, "actionable": 0.005, "needs_context": 0.005}),
                "claim1_quality": _answer("needs_context", 0.55, {"needs_context": 0.55, "actionable": 0.40, "textbook_general": 0.05}),
            }
        },
    )

    assert candidates[0]["promotion_status"] == "ready_for_promotion"
    assert candidates[1]["promotion_status"] == "promoted"


def test_the_class_catalog_travels_once_not_once_per_claim():
    """Shared state carries the definitions; questions carry only the gloss."""
    payload = guard.build_quality_request(_claims() * 8)

    assert payload["state"]["judgment_quality_classes"] == guard.JUDGMENT_QUALITY_CLASSES
    for question in payload["questions"].values():
        assert question["criteria"] == guard.CLASS_GLOSS
        assert set(question["criteria"]) == set(guard.JUDGMENT_QUALITY_CLASSES)
    blob = json.dumps(payload, ensure_ascii=False)
    assert blob.count(guard.JUDGMENT_QUALITY_CLASSES["needs_context"]) == 1


def test_inline_criteria_restores_the_repeated_payload_shape():
    """The A/B switch reproduces the pre-change form for a Phase 1 comparison."""
    payload = guard.build_quality_request(_claims(), inline_criteria=True)

    assert "judgment_quality_classes" not in payload["state"]
    for question in payload["questions"].values():
        assert question["criteria"] == guard.JUDGMENT_QUALITY_CLASSES


def test_gloss_and_catalog_must_cover_the_same_classes(monkeypatch):
    """A class added to one table and not the other must not reach the wire."""
    monkeypatch.setitem(guard.CLASS_GLOSS, "unmapped_class", "説明")

    with pytest.raises(guard.TypeSafeJudgmentQualityError):
        guard.build_quality_request(_claims())
