from api.knowledge.shion_hyde import build_combined_search_query, build_shion_hyde_query


def test_shion_hyde_converts_ambiguous_worry_to_screening_terms():
    result = build_shion_hyde_query("この案件、なんか不安。どこを見ればいい？")

    assert result.should_search is True
    assert "general_screening" in result.intent_tags
    assert "返済原資" in result.hyde_query
    assert "競合・成約リスク" in result.hyde_query
    assert "物件換金性" in result.hyde_query
    assert "最大3つ" in result.hyde_query


def test_shion_hyde_detects_competition_and_asset_liquidity():
    result = build_shion_hyde_query("競合がありそう。この設備は中古で売れるかな")

    assert "competition_risk" in result.intent_tags
    assert "asset_liquidity" in result.intent_tags
    assert "相見積" in result.search_terms
    assert "中古市場" in result.search_terms


def test_shion_hyde_skips_too_short_query():
    result = build_shion_hyde_query("？")

    assert result.should_search is False
    assert result.hyde_query == ""
    assert result.reason == "empty_or_too_short"


def test_combined_search_query_contains_original_and_hyde_terms():
    result = build_shion_hyde_query("新規先で返済原資が気になる")
    combined = build_combined_search_query(result)

    assert "新規先で返済原資が気になる" in combined
    assert "new_customer" in combined
    assert "repayment_source" in combined
    assert "仮想審査メモ" in combined
