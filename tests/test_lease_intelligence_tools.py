from lease_intelligence_tools import (
    execute_tool,
    inspect_scoring_policy,
    search_obsidian,
)


def test_inspect_scoring_policy_reports_route_split():
    result = inspect_scoring_policy("物件スコアと借手スコアの統合")

    assert result["status"] == "current_implementation_route_split"
    routes = result["facts"]["routes"]
    assert routes["quick_batch_scoring_core"]["asset_score_affects_final_score"] is False
    assert routes["quick_batch_scoring_core"]["base_score_source"] == "score_borrower"
    assert routes["next_full_api"]["asset_score_affects_final_score"] is True
    assert routes["next_full_api"]["endpoint"] == "/api/score/full"
    assert "経路で異なる" in result["explanation"]


def test_search_obsidian_uses_shared_context_route(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "obsidian_ai_context.collect_obsidian_ai_context",
        lambda *args, **kwargs: {
            "hits": [
                {
                    "path": "Policies/scoring.md",
                    "snippet": "物件評価は警告として扱う。",
                    "source": "keyword",
                    "score": 12.0,
                }
            ]
        },
    )

    result = search_obsidian("asset_score 重み付け", tmp_path)

    assert result["count"] == 1
    assert result["results"][0]["file"] == "Policies/scoring.md"
    assert result["search_route"].startswith("obsidian_query")


def test_execute_tool_dispatches_scoring_policy():
    result = execute_tool(
        "inspect_scoring_policy",
        {"topic": "承認理由"},
    )

    assert result["facts"]["requires_route_identification"] is True


def test_tool_declarations_include_senior_reasoner_contract():
    from lease_intelligence_tools import TOOL_DECLARATIONS

    declaration = next(
        item for item in TOOL_DECLARATIONS
        if item["name"] == "consult_senior_reasoner"
    )
    required = declaration["parameters"]["required"]
    assert required == [
        "question",
        "shion_hypothesis",
        "confidence",
        "evidence_summary",
    ]


def test_obsidian_query_expands_scoring_identifiers_to_business_terms():
    from mobile_app.obsidian_bridge import _expand_query_terms

    terms = _expand_query_terms(
        "scoring_core asset_score score_borrower 統合 重み付け"
    )

    assert "物件スコア" in terms
    assert "借手スコア" in terms
    assert "最終スコア" in terms
    assert "担保価値" in terms
    assert "配点" in terms
