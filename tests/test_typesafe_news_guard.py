from __future__ import annotations

import typesafe_news_guard as guard


def _noul(probability: float) -> dict[str, object]:
    # 実応答の形（2026-09-25 に本物のレスポンスで確認）。"probability" と書くと
    # テストは通るのに本番で毎回パース失敗する。外部契約のフィクスチャは
    # 推測で書かず実物に合わせる。
    return {"type": "noul", "noul": probability}


def _articles(count: int) -> list[dict[str, str]]:
    return [
        {
            "title": f"記事{index}",
            "summary": "設備投資と資金繰りに関する報道",
            "source": "日経",
            "query": "リース 与信",
        }
        for index in range(count)
    ]


def test_mode_defaults_to_off_and_rejects_unknown_values():
    assert guard.news_guard_mode({}) == "off"
    assert guard.news_guard_mode({"TYPESAFE_NEWS_MODE": "bogus"}) == "off"
    assert guard.news_guard_mode({"TYPESAFE_NEWS_MODE": "ENFORCE"}) == "enforce"


def test_injection_is_checked_before_relevance():
    """返済力に効く記事でも、指示文を含むなら隔離される。"""
    action = guard.decide_article_action(
        {"repayment": 0.99, "injection": 0.95},
        relevant_threshold=0.35,
        injection_threshold=0.70,
    )
    assert action == "quarantine"


def test_ambiguous_article_is_sent_not_dropped():
    """0.5付近は落とさない。審査に効く記事を落とす誤りの方が高くつく。"""
    assert (
        guard.decide_article_action(
            {"repayment": 0.50, "injection": 0.01},
            relevant_threshold=guard.DEFAULT_RELEVANT_MIN,
            injection_threshold=0.70,
        )
        == "send"
    )
    assert (
        guard.decide_article_action(
            {"repayment": 0.10, "injection": 0.01},
            relevant_threshold=guard.DEFAULT_RELEVANT_MIN,
            injection_threshold=0.70,
        )
        == "skip"
    )


def test_screen_articles_batches_every_judgment_into_one_request():
    captured: list[dict] = []

    def fake_request(payload):
        captured.append(payload)
        answers = {}
        for index in range(len(payload["state"]["articles"])):
            answers[f"a{index}_repayment"] = _noul(0.9 if index == 0 else 0.1)
            answers[f"a{index}_injection"] = _noul(0.02)
        return {"answers": answers, "model": "jev-latest", "usage": {"requests": 1}}

    result = guard.screen_articles(_articles(3), request_fn=fake_request, environ={})

    assert len(captured) == 1, "記事ごとに呼ぶと削減分を呼び出し回数で食い潰す"
    assert len(captured[0]["questions"]) == 6
    assert result["actions"] == ["send", "skip", "skip"]
    assert result["counts"] == {"send": 1, "skip": 2, "quarantine": 0}


def test_articles_with_internal_identifiers_are_never_sent():
    articles = [{"title": "案件 ABC-1234 の稟議", "summary": "", "source": "", "query": ""}]

    def fail_request(payload):  # pragma: no cover - 呼ばれてはいけない
        raise AssertionError("内部識別子を含む記事を外部送信した")

    result = guard.screen_articles(articles, request_fn=fail_request, environ={})
    assert result["status"] == "skipped"
    assert result["actions"] == ["send"], "送らない判断でも分類自体は従来どおり続ける"


def test_missing_answer_raises_instead_of_silently_dropping():
    def partial_request(payload):
        return {"answers": {"a0_repayment": _noul(0.9)}, "model": "jev-latest"}

    try:
        guard.screen_articles(_articles(1), request_fn=partial_request, environ={})
    except guard.TypeSafeRagError:
        return
    raise AssertionError("回答欠落を検出できていない")
