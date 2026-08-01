from api import vertex_agent_search as vertex


def test_vertex_agent_search_disabled_skips_without_auth(monkeypatch):
    monkeypatch.setenv("VERTEX_AGENT_SEARCH_ENABLED", "false")

    result = vertex.search_vertex_agent("補助金前提の工作機械リース")

    assert result["used"] is False
    assert result["status"] == "disabled"
    assert result["refs"] == []


def test_parse_results_deduplicates_same_autoresearch_topic():
    response = {
        "results": [
            {
                "document": {
                    "id": "a",
                    "structData": {
                        "title": "返済余力と資金繰りの異常兆候 - リース判断Auto Research",
                        "source_path": "Projects/tune_lease_55/Research/Auto Research/2026-07-06_cash-flow.md",
                    },
                    "derivedStructData": {"snippets": [{"snippet": "資金繰りを確認"}]},
                }
            },
            {
                "document": {
                    "id": "b",
                    "structData": {
                        "title": "返済余力と資金繰りの異常兆候 - リース判断Auto Research",
                        "source_path": "Projects/tune_lease_55/Research/Auto Research/2026-07-26_cash-flow.md",
                    },
                    "derivedStructData": {"snippets": [{"snippet": "返済余力を確認"}]},
                }
            },
            {
                "document": {
                    "id": "c",
                    "structData": {
                        "title": "補助金 総合レポート",
                        "source_path": "Projects/tune_lease_55/Research/補助金 総合レポート.md",
                    },
                    "derivedStructData": {"snippets": [{"snippet": "<b>補助金</b>の対象経費"}]},
                }
            },
        ]
    }

    parsed = vertex._parse_results(response, 5)

    assert [item["doc_id"] for item in parsed] == ["a", "c"]
    assert parsed[1]["excerpt"] == "補助金の対象経費"


def test_build_answer_request_body_clamps_rephrase_steps_and_includes_controls():
    body = vertex.build_answer_request_body(
        "補助金前提の工作機械リース",
        page_size=20,
        filter_expression='source_path: ANY("Research")',
        boost_spec={"conditionBoostSpecs": [{"condition": "title: ANY(\"補助金\")", "boost": 0.5}]},
        grounding_filtering_level="FILTERING_LEVEL_LOW",
        max_rephrase_steps=9,
    )

    assert body["query"] == {"text": "補助金前提の工作機械リース"}
    assert body["answerGenerationSpec"]["includeCitations"] is True
    assert body["answerGenerationSpec"]["answerLanguageCode"] == "ja"
    assert body["queryUnderstandingSpec"]["queryRephraserSpec"]["maxRephraseSteps"] == 5
    assert body["groundingSpec"]["includeGroundingSupports"] is True
    assert body["groundingSpec"]["filteringLevel"] == "FILTERING_LEVEL_LOW"
    params = body["searchSpec"]["searchParams"]
    assert params["maxReturnResults"] == 10
    assert params["filter"] == 'source_path: ANY("Research")'
    assert params["boostSpec"]["conditionBoostSpecs"][0]["boost"] == 0.5


def test_local_controls_promote_domain_hits_and_bury_chat_noise():
    results = [
        {"title": "AI Chat", "source_path": "AI Chat/補助金メモ.md", "excerpt": "補助金"},
        {"title": "補助金 総合レポート", "source_path": "Projects/tune_lease_55/Research/補助金 総合レポート.md", "excerpt": "対象要件"},
    ]

    ranked, controls = vertex._apply_local_controls(results, "補助金前提の工作機械リース")

    assert controls["mode"] == "local_rerank"
    assert ranked[0]["title"] == "補助金 総合レポート"


def test_collect_answer_search_results_dedupes_uris():
    response = {
        "answer": {
            "steps": [
                {
                    "actions": [
                        {
                            "observation": {
                                "searchResults": [
                                    {"title": "A", "uri": "obsidian://a", "snippetInfo": {"snippet": "<b>残価</b>"}},
                                    {"title": "A again", "uri": "obsidian://a", "snippet": "duplicate"},
                                    {"title": "B", "uri": "obsidian://b", "snippet": "耐用年数"},
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    }

    parsed = vertex._collect_answer_search_results(response, 5)

    assert [item["uri"] for item in parsed] == ["obsidian://a", "obsidian://b"]
    assert parsed[0]["snippet"] == "残価"


def test_extract_answer_grounding_supports_and_average_score():
    answer = {
        "groundingSupports": [
            {"claimText": "補助金の対象経費は公募要領で確認する。", "supportScore": 0.9, "citationIndices": [0]},
            {"claimText": "低稼働なら保守状況も見る。", "supportScore": 0.4, "citationIndices": [1]},
        ]
    }

    parsed = vertex._extract_answer_grounding(answer)

    assert parsed["grounding_score"] == 0.65
    assert parsed["support_count"] == 2
    assert parsed["low_support_claim_count"] == 1
    assert parsed["grounding_score_source"] == "official"
    assert parsed["grounding_supports"][0]["claim_text"] == "補助金の対象経費は公募要領で確認する。"


def test_citation_coverage_score_counts_supported_answer_spans():
    score = vertex._citation_coverage_score(
        "abcdefghij",
        [
            {"startIndex": "0", "endIndex": "4"},
            {"startIndex": "6", "endIndex": "9"},
        ],
    )

    assert score == 0.7
