"""collect_lease_news_to_obsidian 側のモード別配線。ここが本番の安全性と課金を決める。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRIPT_PATH = _ROOT / "scripts" / "collect_lease_news_to_obsidian.py"
_SPEC = importlib.util.spec_from_file_location("collect_lease_news_to_obsidian", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
collector = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = collector
_SPEC.loader.exec_module(collector)

import typesafe_news_guard as guard  # noqa: E402


def _article(title: str) -> "collector.Article":
    return collector.Article(
        title=title,
        link=f"https://example.com/{title}",
        source="Example",
        published=None,
        summary="要約",
        query="リース",
    )


def test_off_mode_never_calls_the_guard(monkeypatch):
    """既定offでは1リクエストも発生させない。課金と障害面を増やさないため。"""
    monkeypatch.delenv("TYPESAFE_NEWS_MODE", raising=False)
    monkeypatch.setattr(
        guard,
        "screen_articles",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("off で呼んだ")),
    )

    assert collector._news_guard_actions([_article("a"), _article("b")]) == ["send", "send"]


def test_shadow_mode_measures_but_changes_nothing(monkeypatch):
    monkeypatch.setenv("TYPESAFE_NEWS_MODE", "shadow")
    monkeypatch.setattr(guard, "typesafe_available", lambda *a, **k: True)
    monkeypatch.setattr(
        guard,
        "screen_articles",
        lambda articles: {
            "status": "applied",
            "actions": ["skip", "quarantine"],
            "judgments": [
                {"index": 0, "action": "skip", "repayment": 0.1, "injection": 0.0},
                {"index": 1, "action": "quarantine", "repayment": 0.9, "injection": 0.9},
            ],
            "counts": {"send": 0, "skip": 1, "quarantine": 1},
        },
    )

    actions = collector._news_guard_actions([_article("a"), _article("b")])

    assert actions == ["send", "send"], "shadow は測るだけで送信内容を変えない"


def test_shadow_mode_warns_that_injection_was_not_excluded(monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_NEWS_MODE", "shadow")
    monkeypatch.setattr(guard, "typesafe_available", lambda *a, **k: True)
    monkeypatch.setattr(
        guard,
        "screen_articles",
        lambda articles: {
            "status": "applied",
            "actions": ["quarantine"],
            "judgments": [{"index": 0, "action": "quarantine", "repayment": 0.9, "injection": 0.95}],
            "counts": {"send": 0, "skip": 0, "quarantine": 1},
        },
    )

    collector._news_guard_actions([_article("a")])

    assert "NOT excluded (shadow mode)" in capsys.readouterr().err, (
        "検知したのに素通りした事実がログに出ないと、enforceへ上げる判断材料が残らない"
    )


def test_enforce_mode_applies_skip_and_quarantine(monkeypatch):
    monkeypatch.setenv("TYPESAFE_NEWS_MODE", "enforce")
    monkeypatch.setattr(guard, "typesafe_available", lambda *a, **k: True)
    monkeypatch.setattr(
        guard,
        "screen_articles",
        lambda articles: {
            "status": "applied",
            "actions": ["send", "skip", "quarantine"],
            "judgments": [{"index": 2, "action": "quarantine", "repayment": 0.9, "injection": 0.9}],
            "counts": {"send": 1, "skip": 1, "quarantine": 1},
        },
    )

    actions = collector._news_guard_actions([_article("a"), _article("b"), _article("c")])

    assert actions == ["send", "skip", "quarantine"]


def test_enforce_mode_sends_everything_without_credential(monkeypatch):
    """鍵未設定でニュース収集の品質を落とさない。"""
    monkeypatch.setenv("TYPESAFE_NEWS_MODE", "enforce")
    monkeypatch.setattr(guard, "typesafe_available", lambda *a, **k: False)
    monkeypatch.setattr(
        guard,
        "screen_articles",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("鍵なしで呼んだ")),
    )

    assert collector._news_guard_actions([_article("a")]) == ["send"]
