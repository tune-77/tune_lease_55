"""Tests for the optional TypeSafe/Jev dedup guard.

Every test injects a fake ``request_fn`` or leaves the feature disabled, so the
suite never contacts TypeSafe.
"""

from __future__ import annotations

import importlib.util
import concurrent.futures
from pathlib import Path

import pytest

import typesafe_dedup_guard as guard

ROOT = Path(__file__).resolve().parents[1]


def _load_extract_module():
    spec = importlib.util.spec_from_file_location(
        "extract_obsidian_improvements_for_dedup_test",
        ROOT / "scripts" / "extract_obsidian_improvements.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _candidates() -> list[dict[str, str]]:
    return [
        {"title": "PRタイトルにREV番号が入らない", "reason": "台帳が更新されない"},
        {"title": "cleanup_improvement_reviews が空振りする", "reason": "台帳が更新されない"},
        {"title": "業種別成約率の表示を追加", "reason": "傾向が見えない"},
    ]


class _FixedSimilarity:
    """Stand-in for the bigram metric with explicit, readable scores."""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self._scores = scores

    def __call__(self, a: str, b: str) -> float:
        return self._scores.get((a, b), self._scores.get((b, a), 0.0))


def test_gray_high_matches_the_existing_jaccard_threshold() -> None:
    """The band's upper edge must track the rule it defers to."""
    module = _load_extract_module()
    assert guard.GRAY_HIGH == module._JACCARD_THRESHOLD


def test_select_gray_pairs_excludes_confident_bands() -> None:
    candidates = _candidates()
    similarity = _FixedSimilarity(
        {
            (candidates[0]["title"], candidates[1]["title"]): 0.42,  # gray
            (candidates[0]["title"], candidates[2]["title"]): 0.05,  # clearly distinct
            (candidates[1]["title"], candidates[2]["title"]): 0.80,  # already duplicate
        }
    )

    assert guard.select_gray_pairs(candidates, similarity) == [(0, 1)]


def test_select_gray_pairs_treats_the_upper_edge_as_already_decided() -> None:
    candidates = _candidates()[:2]
    at_threshold = _FixedSimilarity(
        {(candidates[0]["title"], candidates[1]["title"]): guard.GRAY_HIGH}
    )
    just_below = _FixedSimilarity(
        {(candidates[0]["title"], candidates[1]["title"]): guard.GRAY_HIGH - 0.01}
    )

    assert guard.select_gray_pairs(candidates, at_threshold) == []
    assert guard.select_gray_pairs(candidates, just_below) == [(0, 1)]


def test_build_pair_request_batches_one_request_and_truncates() -> None:
    candidates = [
        {"title": "あ" * 500, "reason": "い" * 900},
        {"title": "う" * 10, "reason": "え" * 10},
        {"title": "お" * 10, "reason": "か" * 10},
    ]

    payload = guard.build_pair_request(candidates, [(0, 1), (0, 2)])

    assert set(payload["questions"]) == {"pair0_same_issue", "pair1_same_issue"}
    assert len(payload["state"]["pairs"]) == 2
    first = payload["state"]["pairs"][0]
    assert len(first["a_title"]) == guard.MAX_TITLE_CHARS
    assert set(first) == {"a_title", "b_title"}
    assert payload["questions"]["pair0_same_issue"]["type"] == "noul"


def test_build_pair_request_omits_local_metadata() -> None:
    candidates = [
        {"title": "A", "reason": "r", "tag": "[改善]", "source_path": "/Users/me/Vault/x.md"},
        {"title": "B", "reason": "r", "tag": "[TODO]", "source_path": "/Users/me/Vault/y.md"},
    ]

    payload = guard.build_pair_request(candidates, [(0, 1)])

    sent = payload["state"]["pairs"][0]
    assert set(sent) == {"a_title", "b_title"}
    assert "/Users/me" not in str(payload)
    assert "reason" not in str(payload)


@pytest.mark.parametrize(
    "candidate",
    [
        {"title": "A社の案件を修正", "reason": ""},
        {"title": "表示修正", "reason": "顧客番号: 12345"},
        {"title": "通知改善", "reason": "user@example.comへ送る"},
        {"title": "入力改善", "reason": "03-1234-5678"},
        {"title": "金額表示", "reason": "1,000万円"},
        {"title": "山田商店の自己資本を確認", "reason": ""},
        {"title": "鈴木太郎は年収500万", "reason": "評価する"},
    ],
)
def test_privacy_screen_rejects_case_and_personal_data(candidate) -> None:
    assert guard.is_safe_public_candidate(candidate) is False


def test_filter_safe_pairs_keeps_only_public_improvements() -> None:
    candidates = [
        {"title": "表示ラベル修正", "reason": "誤字"},
        {"title": "UI文言改善", "reason": "説明不足"},
        {"title": "A社の案件修正", "reason": "個別案件"},
    ]

    safe, skipped = guard.filter_safe_pairs(candidates, [(0, 1), (0, 2)])

    assert safe == [(0, 1)]
    assert skipped == 1


def test_judge_pairs_routes_by_probability() -> None:
    candidates = _candidates()
    captured: dict[str, object] = {}

    def fake_request(payload):
        captured["payload"] = payload
        return {
            "answers": {
                "pair0_same_issue": {"type": "noul", "noul": 0.87},
                "pair1_same_issue": {"type": "noul", "noul": 0.12},
            },
            "model": "jev-latest",
            "usage": {"input_tokens": 120},
        }

    judged, meta = guard.judge_pairs(
        candidates, [(0, 1), (0, 2)], request_fn=fake_request
    )

    assert [item["route"] for item in judged] == ["duplicate", "distinct"]
    assert judged[0] == {"a": 0, "b": 1, "same_issue": 0.87, "route": "duplicate"}
    assert meta["status"] == "applied"
    assert meta["pair_count"] == 2
    assert meta["duplicate_count"] == 1
    assert meta["usage"] == {"input_tokens": 120}
    assert len(captured["payload"]["questions"]) == 2  # one batched request


def test_judge_pairs_honors_an_explicit_threshold() -> None:
    def fake_request(_payload):
        return {"answers": {"pair0_same_issue": {"type": "noul", "noul": 0.55}}}

    strict, _ = guard.judge_pairs(
        _candidates(), [(0, 1)], request_fn=fake_request, threshold=0.70
    )
    lenient, _ = guard.judge_pairs(
        _candidates(), [(0, 1)], request_fn=fake_request, threshold=0.50
    )

    assert strict[0]["route"] == "distinct"
    assert lenient[0]["route"] == "duplicate"


def test_judge_pairs_skips_when_no_pair_is_ambiguous() -> None:
    def fail_request(_payload):  # pragma: no cover - must never run
        raise AssertionError("no request should be sent")

    judged, meta = guard.judge_pairs(_candidates(), [], request_fn=fail_request)

    assert judged == []
    assert meta == {"status": "skipped", "reason": "no_pairs"}


def test_invalid_probability_is_rejected() -> None:
    def fake_request(_payload):
        return {"answers": {"pair0_same_issue": {"type": "noul", "noul": 1.4}}}

    with pytest.raises(guard.TypeSafeDedupError):
        guard.judge_pairs(_candidates(), [(0, 1)], request_fn=fake_request)


def test_malformed_response_raises_instead_of_reporting_no_duplicates() -> None:
    """A broken response must not look like a successful 'nothing matched'."""

    def fake_request(_payload):
        return {"model": "jev-latest"}  # answers missing entirely

    with pytest.raises(guard.TypeSafeDedupError):
        guard.judge_pairs(_candidates(), [(0, 1)], request_fn=fake_request)


def test_dedup_is_disabled_without_explicit_configuration(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_DEDUP_ENABLED", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEYCHAIN_SERVICE", raising=False)

    assert guard.typesafe_dedup_enabled({}) is False
    assert guard.typesafe_dedup_enabled({"TYPESAFE_DEDUP_ENABLED": "1"}) is False
    assert guard.typesafe_dedup_enabled({"TYPESAFE_API_KEY": "k"}) is False
    assert (
        guard.typesafe_dedup_enabled(
            {"TYPESAFE_DEDUP_ENABLED": "1", "TYPESAFE_API_KEY": "k"}
        )
        is True
    )


def test_dedup_flag_is_independent_of_the_rag_flag() -> None:
    """Either surface must be switchable without disturbing the other."""
    env = {"TYPESAFE_RAG_ENABLED": "1", "TYPESAFE_API_KEY": "k"}

    assert guard.typesafe_dedup_enabled(env) is False


def test_dedup_can_be_enabled_with_macos_keychain(monkeypatch) -> None:
    monkeypatch.setattr(guard.sys, "platform", "darwin")

    class _Result:
        stdout = "secret-from-keychain\n"

    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: _Result())

    assert (
        guard.typesafe_dedup_enabled(
            {"TYPESAFE_DEDUP_ENABLED": "1", "TYPESAFE_API_KEYCHAIN_SERVICE": "svc"}
        )
        is True
    )


def test_keychain_lookup_failure_keeps_dedup_disabled(monkeypatch) -> None:
    monkeypatch.setattr(guard.sys, "platform", "darwin")

    def _raise(*_args, **_kwargs):
        raise OSError("security binary unavailable")

    monkeypatch.setattr(guard.subprocess, "run", _raise)

    assert (
        guard.typesafe_dedup_enabled(
            {"TYPESAFE_DEDUP_ENABLED": "1", "TYPESAFE_API_KEYCHAIN_SERVICE": "svc"}
        )
        is False
    )


def test_judge_pairs_if_enabled_falls_back_to_distinct() -> None:
    """Failure must reproduce today's behavior: the gray band stays separate."""

    def broken_request(_payload):
        raise RuntimeError("timeout")

    judged, meta = guard.judge_pairs_if_enabled(
        _candidates(), [(0, 1), (0, 2)], request_fn=broken_request
    )

    assert [item["route"] for item in judged] == ["distinct", "distinct"]
    assert [item["same_issue"] for item in judged] == [None, None]
    assert meta["status"] == "fallback"
    assert meta["error_type"] == "RuntimeError"


def test_judge_pairs_if_enabled_sends_nothing_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_DEDUP_ENABLED", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEYCHAIN_SERVICE", raising=False)

    judged, meta = guard.judge_pairs_if_enabled(_candidates(), [(0, 1)])

    assert meta == {"status": "disabled"}
    assert judged == [{"a": 0, "b": 1, "same_issue": None, "route": "distinct"}]


def test_default_request_has_hard_wall_clock_deadline(monkeypatch) -> None:
    class TimedOutFuture:
        def result(self, **_kwargs):
            raise concurrent.futures.TimeoutError

    class Executor:
        def submit(self, *_args, **_kwargs):
            return TimedOutFuture()

    monkeypatch.setattr(guard, "_resolve_api_key", lambda: "secret")
    monkeypatch.setattr(guard, "_REQUEST_EXECUTOR", Executor())
    monkeypatch.setenv("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "0.01")

    with pytest.raises(guard.TypeSafeDedupError, match="exceeded"):
        guard._default_request({"state": {}, "questions": {}})
