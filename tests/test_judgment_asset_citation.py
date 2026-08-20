from judgment_asset_citation import (
    classify_citations,
    extract_cited_rule_id_prefixes,
    resolve_rule_ids_from_citations,
)

ACTIVE_IDS = ["b259411afb954d6d", "cf61a9701fc8cc42"]


def test_extract_cited_rule_id_prefixes_dedupes_and_preserves_order():
    text = (
        "判断資産出典: 正規 JA-cr-b2594 / business_plan\n"
        "判断資産出典: 正規 JA-cr-cf61a / asset_life\n"
        "判断資産出典: 正規 JA-cr-b2594 / repeated"
    )
    assert extract_cited_rule_id_prefixes(text) == ["b2594", "cf61a"]


def test_extract_cited_rule_id_prefixes_ignores_non_citation_text():
    assert extract_cited_rule_id_prefixes("特に出典のない普通の文章です。") == []
    assert extract_cited_rule_id_prefixes("") == []
    assert extract_cited_rule_id_prefixes(None) == []


def test_resolve_rule_ids_from_citations_matches_unique_prefix():
    text = "判断資産出典: 正規 JA-cr-b2594 / business_plan"
    assert resolve_rule_ids_from_citations(text, ACTIVE_IDS) == ["b259411afb954d6d"]


def test_resolve_rule_ids_from_citations_skips_ambiguous_prefix():
    ids_with_collision = ACTIVE_IDS + ["b259400000000000"]
    text = "判断資産出典: 正規 JA-cr-b2594"
    assert resolve_rule_ids_from_citations(text, ids_with_collision) == []


def test_resolve_rule_ids_from_citations_skips_unresolved_prefix():
    text = "判断資産出典: 正規 JA-cr-ffffff（失効ルール想定）"
    assert resolve_rule_ids_from_citations(text, ACTIVE_IDS) == []


def test_classify_citations_reports_all_three_buckets():
    ids_with_collision = ACTIVE_IDS + ["b259400000000000"]
    text = (
        "判断資産出典: 正規 JA-cr-b2594\n"
        "判断資産出典: 正規 JA-cr-cf61a\n"
        "判断資産出典: 正規 JA-cr-ffffff"
    )
    result = classify_citations(text, ids_with_collision)
    assert result == {
        "resolved": ["cf61a9701fc8cc42"],
        "ambiguous": ["b2594"],
        "unresolved": ["ffffff"],
    }
