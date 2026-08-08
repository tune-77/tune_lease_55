from api.vertex_knowledge_workflows import (
    build_workflow_query,
    normalize_workflow_mode,
    run_vertex_knowledge_workflow,
)


def test_build_workflow_query_adds_mode_specific_terms():
    query = build_workflow_query("補助金前提の工作機械リース", "judgment_candidates")

    assert "補助金前提の工作機械リース" in query
    assert "判断資産" in query
    assert "確認質問" in query


def test_unknown_workflow_mode_falls_back_to_evidence_support():
    assert normalize_workflow_mode("unknown") == "evidence_support"


def test_run_workflow_uses_injected_search_and_answer_functions():
    calls = {}

    def fake_search(query, **kwargs):
        calls["search_query"] = query
        calls["search_kwargs"] = kwargs
        return {
            "used": True,
            "status": "ok",
            "refs": ["Projects/tune_lease_55/Research/補助金.md"],
            "results": [
                {
                    "title": "補助金確認",
                    "source_path": "Projects/tune_lease_55/Research/補助金.md",
                    "excerpt": "交付決定と対象経費を確認する。",
                }
            ],
        }

    def fake_answer(query, **kwargs):
        calls["answer_query"] = query
        calls["answer_kwargs"] = kwargs
        return {
            "used": True,
            "status": "ok",
            "answer_text": "補助金は入金遅延時の返済余力を確認する。",
            "refs": ["obsidian://補助金"],
            "grounding_score": 0.8,
        }

    result = run_vertex_knowledge_workflow(
        "補助金前提の工作機械リース",
        mode="evidence_support",
        page_size=3,
        search_fn=fake_search,
        answer_fn=fake_answer,
    )

    assert result["mode"] == "evidence_support"
    assert result["label"] == "審査コメント根拠補強"
    assert result["guardrail"].startswith("Vertexは補助索引")
    assert "審査 根拠" in result["query"]
    assert calls["search_kwargs"]["page_size"] == 3
    assert calls["search_kwargs"]["apply_controls"] is True
    assert calls["answer_kwargs"]["include_grounding_supports"] is True
    assert result["high_signal_hits"][0]["title"] == "補助金確認"
    assert result["refs"][0] == "Projects/tune_lease_55/Research/補助金.md"
    assert "審査コメントに使う前" in result["next_actions"][0]


def test_judgment_candidate_workflow_never_auto_promotes():
    def fake_search(query, **kwargs):
        return {"used": True, "status": "ok", "refs": ["Judgment Assets/返済余力.md"], "results": []}

    def fake_answer(query, **kwargs):
        return {"used": True, "status": "ok", "answer_text": "確認質問候補", "refs": []}

    result = run_vertex_knowledge_workflow(
        "返済余力",
        mode="judgment_candidates",
        search_fn=fake_search,
        answer_fn=fake_answer,
    )

    assert result["mode"] == "judgment_candidates"
    assert any("自動昇格せず" in action for action in result["next_actions"])


def test_market_outlook_workflow_flags_reference_use_over_conclusion():
    def fake_search(query, **kwargs):
        return {
            "used": True,
            "status": "ok",
            "refs": ["knowledge_base/lease_review_knowledge/04_残価リスク評価と市場動向.md"],
            "results": [],
        }

    def fake_answer(query, **kwargs):
        return {"used": True, "status": "ok", "answer_text": "金利は緩やかな上昇傾向。", "refs": []}

    result = run_vertex_knowledge_workflow(
        "基準金利 今後の推移",
        mode="market_outlook",
        search_fn=fake_search,
        answer_fn=fake_answer,
    )

    assert result["mode"] == "market_outlook"
    assert result["label"] == "金利推移見通し"
    assert "金利 推移 予測" in result["query"]
    assert any("参考所見" in action for action in result["next_actions"])


def test_empty_workflow_result_suggests_adding_curated_notes():
    def fake_search(query, **kwargs):
        return {"used": False, "status": "ok", "refs": [], "results": []}

    def fake_answer(query, **kwargs):
        return {"used": False, "status": "ok", "answer_text": "", "refs": []}

    result = run_vertex_knowledge_workflow(
        "ニッチなテーマ",
        mode="knowledge_audit",
        search_fn=fake_search,
        answer_fn=fake_answer,
    )

    assert result["refs"] == []
    assert "整理済みノートを追加" in result["next_actions"][1]
