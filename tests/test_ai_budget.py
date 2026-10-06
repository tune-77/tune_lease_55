from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import ai_budget
from ai_runtime_client import tracked_ai_call

JST = timezone(timedelta(hours=9))


@pytest.fixture
def budget_env(tmp_path, monkeypatch):
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("AI_USAGE_LOG_PATH", str(log_path))
    monkeypatch.setenv("AI_BUDGET_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("AI_BUDGET_GUARD", "force")
    monkeypatch.setenv("AI_DAILY_BUDGET_YEN", "50")
    monkeypatch.setenv("AI_USD_JPY", "150")
    monkeypatch.setenv("AI_COST_CALIBRATION", "1")
    for name in ("AI_CALL_CLASS", "AI_LIVE_VERIFY", "XPC_SERVICE_NAME"):
        monkeypatch.delenv(name, raising=False)
    ai_budget._cache.clear()
    return tmp_path


def _spend(log_path, yen: float, *, call_class: str = "essential", when: datetime | None = None) -> None:
    # flash-lite 入力 $0.25/M・¥150 → 1円 = 26,667 トークン
    tokens = int(yen / 150 / 0.25 * 1_000_000)
    entry = {
        "timestamp": (when or datetime.now(timezone.utc)).isoformat(),
        "provider": "google",
        "feature": "chat_memory_tools",
        "model": "gemini-3.1-flash-lite",
        "ok": True,
        "input_tokens": tokens,
        "output_tokens": 0,
        "total_tokens": tokens,
        "call_class": call_class,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def test_call_class_precedence(monkeypatch):
    monkeypatch.delenv("AI_CALL_CLASS", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    assert ai_budget.call_class("shinsa_gunshi", {"source": "__main__.py"}) == "essential"
    assert ai_budget.call_class("lease_intelligence_reflection", {"source": "__main__.py"}) == "memory"
    assert ai_budget.call_class("novelist_daily_image", {"source": "__main__.py"}) == "proactive"
    assert ai_budget.call_class("chat_memory_tools", {"source": "-"}) == "verification"
    assert ai_budget.call_class("chat_memory_tools", {"source": "x.py", "worktree": "tl55-x"}) == "verification"
    assert ai_budget.call_class("ai_chat", {"source": "evaluate_answer_quality.py"}) == "verification"
    # launchd の夜間ジョブは、検証名のスクリプトでも夜間扱い（明示フラグなしで止めない）
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.tunelease.improvement-pipeline")
    assert ai_budget.call_class("ai_chat", {"source": "evaluate_okf_rag.py"}) == "nightly"
    # 夜間でも記憶・内省は memory（止めない）
    assert ai_budget.call_class("ai_chat", {"source": "build_shion_memory_promotion_queue.py"}) == "memory"
    assert ai_budget.call_class("loop_engineering", {"source": "x.py"}) == "memory"
    monkeypatch.setenv("XPC_SERVICE_NAME", "com.tunelease.next")
    assert ai_budget.call_class("shinsa_gunshi", {"source": "__main__.py"}) == "essential"
    monkeypatch.setenv("AI_CALL_CLASS", "proactive")
    assert ai_budget.call_class("chat_memory_tools", {"source": "-"}) == "proactive"


def test_entry_cost_counts_thinking_and_cache():
    base = {"provider": "google", "model": "gemini-3.1-flash-lite"}
    plain = ai_budget.entry_cost_usd({**base, "input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000})
    assert plain == pytest.approx(0.25)
    cached = ai_budget.entry_cost_usd({**base, "input_tokens": 1_000_000, "cached_tokens": 1_000_000, "total_tokens": 1_000_000})
    assert cached == pytest.approx(0.025)
    thinking = ai_budget.entry_cost_usd({**base, "input_tokens": 0, "output_tokens": 10, "total_tokens": 1_000_000})
    assert thinking == pytest.approx(1.5)
    assert ai_budget.entry_cost_usd({"provider": "anthropic", "input_tokens": 10**6}) == 0


def test_verification_needs_live_flag_and_cap(budget_env, monkeypatch):
    monkeypatch.setenv("AI_CALL_CLASS", "verification")
    with pytest.raises(ai_budget.AIBudgetBlocked, match="AI_LIVE_VERIFY"):
        tracked_ai_call(lambda: {}, provider="google", model="gemini-3.1-flash-lite", feature="chat_memory_tools")
    monkeypatch.setenv("AI_LIVE_VERIFY", "1")
    monkeypatch.setenv("AI_VERIFY_MAX_CALLS", "2")
    for _ in range(2):
        tracked_ai_call(lambda: {}, provider="google", model="gemini-3.1-flash-lite", feature="chat_memory_tools")
    with pytest.raises(ai_budget.AIBudgetBlocked, match="verify_daily_cap"):
        tracked_ai_call(lambda: {}, provider="google", model="gemini-3.1-flash-lite", feature="chat_memory_tools")
    events = [json.loads(line) for line in (budget_env / "events.jsonl").read_text().splitlines()]
    assert [e["reason"] for e in events] == ["live_verify_flag_required", "verify_daily_cap"]


def test_stop_order_keeps_essential(budget_env, monkeypatch):
    log_path = budget_env / "usage.jsonl"
    _spend(log_path, 35)  # 70%: 検証は止まり、夜間・自発は通る
    monkeypatch.setenv("AI_LIVE_VERIFY", "1")
    with pytest.raises(ai_budget.AIBudgetBlocked, match="daily_budget"):
        ai_budget.check("x", "verification")
    ai_budget.check("x", "proactive")
    ai_budget.check("x", "nightly")
    _spend(log_path, 20)  # 110%: 自発・夜間も止まり、チャットは通る
    ai_budget._cache.clear()
    with pytest.raises(ai_budget.AIBudgetBlocked):
        ai_budget.check("novelist_daily_image", "proactive")
    assert ai_budget.deferred("proactive") is True
    ai_budget.check("lease_intelligence_reflection", "memory")  # 記憶・内省は上限超過でも止めない
    assert ai_budget.deferred("memory") is False
    called = []
    monkeypatch.setenv("AI_CALL_CLASS", "essential")  # worktree 内の実行でも本番チャット相当として扱う
    tracked_ai_call(lambda: called.append(1) or {}, provider="google", model="m", feature="chat_memory_tools")
    assert called == [1]
    events = [json.loads(line) for line in (budget_env / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "limit_reached" for e in events)


def test_guard_ignores_other_providers_and_off_switch(budget_env, monkeypatch):
    monkeypatch.setenv("AI_CALL_CLASS", "verification")
    tracked_ai_call(lambda: {}, provider="anthropic", model="claude", feature="x")
    monkeypatch.setenv("AI_BUDGET_GUARD", "off")
    tracked_ai_call(lambda: {}, provider="google", model="m", feature="x")


def test_morning_report_lines(budget_env):
    log_path = budget_env / "usage.jsonl"
    now = datetime(2026, 10, 8, 6, 0, tzinfo=JST)
    yesterday = datetime(2026, 10, 7, 12, 0, tzinfo=JST)
    _spend(log_path, 60, when=yesterday)
    _spend(log_path, 5, call_class="proactive", when=yesterday)
    (budget_env / "events.jsonl").write_text(
        json.dumps({"date": "2026-10-07", "event": "blocked", "call_class": "verification", "reason": "daily_budget"}) + "\n",
        encoding="utf-8",
    )
    lines = ai_budget.morning_report_lines(today=now)
    assert lines[0].startswith("> [!warning] Gemini 1日予算に到達")
    assert "verification:daily_budget 1件" in lines[1]
    assert "前日 ¥65" in lines[2] and "essential ¥60" in lines[2]
