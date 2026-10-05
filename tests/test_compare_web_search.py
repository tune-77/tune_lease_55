import datetime as dt
import json

import pytest

from scripts import compare_web_search as cws


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode()

    def json(self):
        return self._payload


def test_score_result_counts_expected_words_dates_and_primary_sources():
    case = cws.Case("x", "news", "q", ("使用権資産", "存在しない語"))
    sources = [
        {"title": "企業会計基準委員会", "url": "https://www.asb-j.jp/a"},
        {"title": "blog", "url": "https://example.com/b"},
    ]
    text = "2026年9月12日更新。使用権資産を計上。2099年1月は未来日付なので無視。"
    scored = cws.score_result(case, text, sources, dt.date(2026, 10, 5))
    assert scored["expect_hit"] == "1/2"
    assert scored["latest_date"] == "2026-09-12"
    assert scored["primary_sources"] == 1
    assert scored["unique_domains"] == 2
    assert scored["ja_ratio"] > 0.5


def test_cloudflare_search_posts_documented_body_and_parses_items():
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return _Response(
            200,
            {"items": [{"url": "https://www.meti.go.jp/x", "title": "補助金", "description": "公募 締切 2026年10月"}],
             "metadata": {"latencyMs": 612}},
        )

    result = cws.cloudflare_search("クエリ", "ceramic", token="t", account_id="acc", limit=50, post=fake_post)
    assert seen["url"].endswith("/accounts/acc/ai/websearch/")
    assert seen["headers"]["Authorization"] == "Bearer t"
    assert seen["body"]["provider"] == "ceramic"
    assert seen["body"]["limit"] == 10  # API 上限に丸める
    assert seen["body"]["options"]["gateway"]["id"]
    assert result["sources"] == [{"title": "補助金", "url": "https://www.meti.go.jp/x"}]
    assert "公募" in result["text"]
    assert result["server_latency_ms"] == 612


def test_cloudflare_search_raises_on_payment_required():
    def fake_post(url, headers, json, timeout):
        return _Response(402, {"ok": False, "error": {"code": "web_search_payment_required"}})

    with pytest.raises(RuntimeError, match="payment_required"):
        cws.cloudflare_search("q", "linkup", token="t", account_id="a", post=fake_post)


def test_payment_required_stops_further_calls_for_that_backend(monkeypatch):
    calls = []

    def fake_search(query, provider, **kwargs):
        calls.append(provider)
        raise RuntimeError("Cloudflare web search 402: web_search_payment_required")

    monkeypatch.setattr(cws, "cloudflare_credentials", lambda: ("t", "a"))
    monkeypatch.setattr(cws, "cloudflare_search", fake_search)
    rows = cws.run(["ceramic"], cws.CASES[:3], limit=3)
    assert calls == ["ceramic"]
    assert len(rows) == 1 and "payment_required" in rows[0]["error"]


def test_budget_guard_refuses_expensive_plan(capsys):
    assert cws.main(["--backends", "exa", "--max-cost-usd", "0.01"]) == 2
    assert cws.estimate_cost(["ceramic"], 8) == pytest.approx(0.002)
