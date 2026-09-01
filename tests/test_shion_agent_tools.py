"""紫苑 ADK エージェントのツール構成テスト。

ツール選定は google.adk 非依存の api/shion_agent_tools.py に集約しているため、
ADK 未導入の CI でもツール構成（読み取り専用・想定ツールが揃っている）を検証できる。
エージェント本体の配線は google.adk がある環境でのみ検証する（無い環境では skip）。
"""

from __future__ import annotations

import pytest


EXPECTED_DB_TOOLS = {
    "search_cases",
    "get_score_detail",
    "score_full_case",
    "get_portfolio_stats",
    "get_weekly_trend",
    "get_system_overview",
    "get_recent_errors",
    "get_pipeline_item_details",
    "recall_judgment_memory",
    "build_judgment_preview",
    "search_obsidian_context",
    "audit_ledger_consistency",
    "get_lease_system_gaps",
    "get_active_rules",
    "review_obsidian_vault_health",
    "suggest_obsidian_curation_actions",
    "get_obsidian_syntax_guide",
    "structure_judgment_asset_candidate",
    "validate_lease_source_summary",
    "convert_research_to_screening_insights",
    "build_screening_decision_flow",
    "write_scqa_report",
    "inspect_agentic_skill_flow",
    "propose_agentic_skill_next_actions",
    "search_shion_system_context",
    "propose_shion_system_improvement_focus",
    "run_shion_memory_system_audit",
    "run_shion_memory_sentinel",
    "audit_memory_index_orphans",
    "audit_memory_freshness_pipeline",
    "audit_memory_revision_integrity",
    "audit_memory_recall_eval_health",
    "audit_cloudrun_data_safety",
}

EXPECTED_VERTEX_TOOLS = {
    "search_lease_knowledge_vertex",
    "answer_lease_question_vertex",
}


def test_read_only_tools_are_wired_and_callable():
    from api.shion_agent_tools import READ_ONLY_DB_TOOLS

    names = {fn.__name__ for fn in READ_ONLY_DB_TOOLS}
    assert names == EXPECTED_DB_TOOLS
    # 全ツールが呼び出し可能な関数であること（配線先のリネーム検知）。
    for fn in READ_ONLY_DB_TOOLS:
        assert callable(fn)
    # 重複登録がないこと。
    assert len(READ_ONLY_DB_TOOLS) == len(EXPECTED_DB_TOOLS)


def test_agent_registers_all_tools_when_adk_available():
    pytest.importorskip("google.adk")
    import api.shion_agent as sa

    names = {fn.__name__ for fn in sa._AGENT_TOOL_FUNCS}
    # 既存2ツール + 読み取り専用DB/Obsidian Curatorツール（Vertex は既定で無効）。
    assert names == {"get_industry_benchmark", "assess_risk_level"} | EXPECTED_DB_TOOLS
    assert len(sa._AGENT_TOOL_FUNCS) == len(EXPECTED_DB_TOOLS) + 2


# ---------- Vertex ツール（課金される外部 API のため opt-in） ----------

def test_vertex_tools_are_disabled_by_default(monkeypatch):
    """既定でエージェント構成を変えないこと（追加課金ゼロを守る）。"""
    monkeypatch.delenv("SHION_ENABLE_VERTEX_TOOLS", raising=False)
    from api import shion_vertex_tools as svt

    assert svt.vertex_tools_enabled() is False
    assert svt.build_vertex_agent_tools() == []


def test_vertex_tools_opt_in_via_env(monkeypatch):
    monkeypatch.setenv("SHION_ENABLE_VERTEX_TOOLS", "1")
    from api import shion_vertex_tools as svt

    assert svt.vertex_tools_enabled() is True
    tools = svt.build_vertex_agent_tools()
    assert {fn.__name__ for fn in tools} == EXPECTED_VERTEX_TOOLS
    for fn in tools:
        assert callable(fn)


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("on", True),
                                          ("0", False), ("", False), ("no", False)])
def test_vertex_opt_in_accepts_common_truthy_values(monkeypatch, raw, expected):
    monkeypatch.setenv("SHION_ENABLE_VERTEX_TOOLS", raw)
    from api import shion_vertex_tools as svt

    assert svt.vertex_tools_enabled() is expected


def test_vertex_tools_are_not_in_the_read_only_list():
    """READ_ONLY_DB_TOOLS は「ローカル読み取りのみ」を保つ。

    課金される外部 API ツールがここに混ざると、追加課金ゼロという前提が黙って壊れる。
    """
    from api.shion_agent_tools import READ_ONLY_DB_TOOLS

    names = {fn.__name__ for fn in READ_ONLY_DB_TOOLS}
    assert names & EXPECTED_VERTEX_TOOLS == set()


def test_obsidian_curator_tools_are_read_only_and_bounded():
    from api.shion_obsidian_curator import (
        review_obsidian_vault_health,
        suggest_obsidian_curation_actions,
    )

    health = review_obsidian_vault_health(limit=3)
    assert health["mode"] == "read_only_obsidian_curator"
    assert "no_vault_write" in health["guardrail"]
    assert len(health["proposals"]) <= 3

    themed = suggest_obsidian_curation_actions("補助金 未採択", limit=2)
    assert themed["mode"] == "read_only_obsidian_curator"
    assert "no_chroma_reindex" in themed["guardrail"]
    assert len(themed.get("suggestions", [])) <= 2


def test_obsidian_syntax_guide_tool_reads_vendored_reference():
    from api.shion_obsidian_curator import get_obsidian_syntax_guide

    markdown = get_obsidian_syntax_guide("callouts")
    assert markdown["status"] == "ok"
    assert markdown["topic"] == "callouts"
    assert "no_vault_write" in markdown["guardrail"]
    assert markdown["content"]

    fallback = get_obsidian_syntax_guide("not_a_real_topic")
    assert fallback["topic"] == "markdown"
    assert fallback["status"] == "ok"


def test_agentic_skill_tools_are_side_effect_free_and_bounded():
    from api.shion_agentic_skills import (
        build_screening_decision_flow,
        convert_research_to_screening_insights,
        structure_judgment_asset_candidate,
        validate_lease_source_summary,
        write_scqa_report,
        inspect_agentic_skill_flow,
        propose_agentic_skill_next_actions,
    )

    candidate = structure_judgment_asset_candidate("競合見積がある時は成約リスクを信用リスクと混同しない")
    assert candidate["status"] == "candidate"
    assert "no_memory_promotion" in candidate["guardrail"]

    source = validate_lease_source_summary(
        "業界ニュース",
        publisher="業界紙",
        published_date="2026-08-22",
        evidence_summary="調査結果あり",
        lease_relevance="設備投資需要に影響",
    )
    assert source["assessment"]["use_level"] in {"usable_as_fact", "usable_as_signal", "background_only"}
    assert "no_auto_memory_promotion" in source["guardrail"]

    insights = convert_research_to_screening_insights("金利上昇で資金繰りが悪化しやすい", max_checks=99)
    assert len(insights["top_checks"]) <= 3
    assert "no_score_change" in insights["guardrail"]

    flow = build_screening_decision_flow("赤字だが銀行支援あり。追加確認して条件付き承認。")
    assert flow["nodes"][0]["type"] == "start"
    assert "automatic_approval" in flow["guardrail"]

    scqa = write_scqa_report("現状は情報が散らばっている。判断に使いにくい。型に整理する。")
    assert set(scqa["scqa"]) == {"situation", "complication", "question", "answer"}
    assert "writing_support_only" in scqa["guardrail"]

    flow = inspect_agentic_skill_flow(limit="loose")
    assert flow["mode"] == "agentic_skill_flow_inspection"
    assert "read_only" in flow["guardrail"]

    proposals = propose_agentic_skill_next_actions(limit="loose")
    assert proposals["mode"] == "agentic_skill_next_action_proposals"
    assert len(proposals["proposals"]) <= 3
    assert "proposal_only" in proposals["guardrail"]


def test_agentic_skill_usage_log_and_review_inbox(tmp_path):
    from api.agentic_skill_usage import (
        load_agentic_skill_review_inbox,
        public_usage_notice,
        record_agentic_skill_call,
        record_agentic_skill_result,
        check_agentic_skill_flow,
        review_agentic_skill_inbox_item,
    )

    usage_log = tmp_path / "usage.jsonl"
    review_inbox = tmp_path / "inbox.jsonl"
    review_log = tmp_path / "reviews.jsonl"

    call = record_agentic_skill_call(
        tool_name="structure_judgment_asset_candidate",
        args={"note": "競合見積がある時は成約リスクとして分ける"},
        session_id="s1",
        case_context={"company_name": "demo"},
        usage_log_path=usage_log,
    )
    assert call and call["event_type"] == "agentic_skill_call"

    result = record_agentic_skill_result(
        tool_name="structure_judgment_asset_candidate",
        result={
            "result": {
                "candidate": {
                    "core_judgment": "競合見積がある時は信用リスクと成約リスクを分ける",
                    "recommended_action": "追加確認する",
                }
            }
        },
        session_id="s1",
        case_context={"company_name": "demo"},
        usage_log_path=usage_log,
        review_inbox_path=review_inbox,
    )
    assert result and result["review_inbox_id"]
    notice = public_usage_notice("structure_judgment_asset_candidate", result)
    assert notice and notice["type"] == "agentic_skill_usage"
    usage_rows = usage_log.read_text(encoding="utf-8").splitlines()
    assert '"review_inbox_id"' in usage_rows[-1]

    inbox = load_agentic_skill_review_inbox(
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )
    assert len(inbox) == 1
    assert inbox[0]["status"] == "candidate"

    review = review_agentic_skill_inbox_item(
        inbox_id=inbox[0]["id"],
        decision="held",
        note="実案件で確認してから",
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )
    assert review["status"] == "recorded"

    reviewed = load_agentic_skill_review_inbox(
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )
    assert reviewed[0]["status"] == "held"
    assert reviewed[0]["review_note"] == "実案件で確認してから"

    flow = check_agentic_skill_flow(
        usage_log_path=usage_log,
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )
    assert flow["status"] == "ok"
    assert flow["summary"]["linked_reviewable_results"] == 1
    assert flow["summary"]["review_decisions"] == 1


def test_agentic_skill_result_flags_similar_prior_rejection(tmp_path):
    """一度rejected/heldされた論点と似た候補は、次回の候補作成時にprior_review_contextへ
    自動で反映される（WikiSkill論文のWiki層＝知見の不変蓄積、に相当）。人間の採否判断は変えない。"""
    from api.agentic_skill_usage import (
        record_agentic_skill_result,
        review_agentic_skill_inbox_item,
        load_agentic_skill_review_inbox,
    )
    from api.shion_agentic_skills import propose_agentic_skill_next_actions

    usage_log = tmp_path / "usage.jsonl"
    review_inbox = tmp_path / "inbox.jsonl"
    review_log = tmp_path / "reviews.jsonl"

    first = record_agentic_skill_result(
        tool_name="structure_judgment_asset_candidate",
        result={"result": {"candidate": {
            "core_judgment": "競合見積がある時は信用リスクとして減点する",
            "recommended_action": "減点する",
        }}},
        session_id="s1",
        usage_log_path=usage_log,
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )
    review_agentic_skill_inbox_item(
        inbox_id=first["review_inbox_id"],
        decision="rejected",
        note="競合見積だけでは減点根拠として弱い",
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )

    record_agentic_skill_result(
        tool_name="structure_judgment_asset_candidate",
        result={"result": {"candidate": {
            "core_judgment": "競合見積がある案件は信用リスクとして減点すべき",
            "recommended_action": "減点する",
        }}},
        session_id="s2",
        usage_log_path=usage_log,
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )

    inbox = load_agentic_skill_review_inbox(
        review_inbox_path=review_inbox,
        review_log_path=review_log,
    )
    new_candidate = [item for item in inbox if item["status"] == "candidate"][0]
    prior = new_candidate.get("prior_review_context")
    assert prior and prior[0]["decision"] == "rejected"
    assert "減点根拠として弱い" in prior[0]["note"]


def test_propose_agentic_skill_next_actions_flags_repeated_pattern(monkeypatch):
    from api import shion_agentic_skills as sas

    monkeypatch.setattr(sas, "check_agentic_skill_flow", lambda: {
        "status": "ok",
        "summary": {},
        "checks": [],
    })
    monkeypatch.setattr(sas, "load_agentic_skill_review_inbox", lambda limit=20, status="candidate": [
        {
            "id": "new1",
            "tool_name": "structure_judgment_asset_candidate",
            "candidate_type": "judgment_asset_candidate",
            "claim": "競合見積がある案件は信用リスクとして減点すべき",
            "prior_review_context": [
                {"decision": "rejected", "note": "競合見積だけでは減点根拠として弱い"},
            ],
        }
    ])

    proposals = sas.propose_agentic_skill_next_actions(limit=3)
    types = [p["type"] for p in proposals["proposals"]]
    assert "flag_repeated_pattern" in types
    flagged = next(p for p in proposals["proposals"] if p["type"] == "flag_repeated_pattern")
    assert flagged["priority"] == "high"
    assert flagged["target"]["prior_review_context"][0]["decision"] == "rejected"


def test_agentic_skill_tools_handle_loose_llm_arguments():
    from api.shion_agentic_skills import (
        convert_research_to_screening_insights,
        validate_lease_source_summary,
    )

    insights = convert_research_to_screening_insights("補助金終了で設備投資が鈍る", max_checks="many")
    assert len(insights["top_checks"]) == 3

    source = validate_lease_source_summary(
        "PR記事",
        publisher="PR TIMES",
        published_date="2026-08-22",
        evidence_summary="スポンサー記事",
        lease_relevance="設備投資需要",
    )
    assert source["assessment"]["reliability"] == "weak"
    assert source["assessment"]["use_level"] == "do_not_use"


def test_shion_system_self_inspection_is_read_only_and_bounded():
    from api.shion_system_self_inspection import (
        propose_shion_system_improvement_focus,
        search_shion_system_context,
    )

    results = search_shion_system_context("紫苑 ADK レビュー箱", limit=50)
    assert results["mode"] == "shion_system_context_search"
    assert results["count"] <= 20
    assert "no_secrets" in results["guardrail"]
    for item in results["results"]:
        assert item["path"] != "MEMORY.md"
        assert ".streamlit" not in item["path"]
        assert "secret" not in item["path"].lower()

    proposals = propose_shion_system_improvement_focus("紫苑 レビュー箱", limit=99)
    assert proposals["mode"] == "shion_system_improvement_focus"
    assert len(proposals["proposals"]) <= 3
    assert proposals["scoring_policy"]["recommendation"] == "today | next | observe"
    for proposal in proposals["proposals"]:
        assert set(proposal["score"]) == {"impact", "risk", "effort", "evidence", "recommendation"}
        assert proposal["score"]["recommendation"] in {"today", "next", "observe"}
    assert "read_only" in proposals["guardrail"]


def test_vertex_search_tool_reports_failure_without_raising(monkeypatch):
    """外部 API が落ちてもエージェントを止めない（例外を返り値に畳む）。"""
    from api import shion_vertex_tools as svt
    from api import vertex_agent_search as vas

    def _boom(*args, **kwargs):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(vas, "search_vertex_agent", _boom)
    result = svt.search_lease_knowledge_vertex("残価", limit=3)

    assert result["used"] is False
    assert "RuntimeError" in result["status"]
    assert result["refs"] == []


def test_vertex_search_tool_clamps_limit(monkeypatch):
    from api import shion_vertex_tools as svt
    from api import vertex_agent_search as vas

    seen: dict = {}

    def _fake(query, *, page_size=None, **kwargs):
        seen["page_size"] = page_size
        return {"used": True, "status": "ok", "refs": []}

    monkeypatch.setattr(vas, "search_vertex_agent", _fake)

    svt.search_lease_knowledge_vertex("q", limit=99)
    assert seen["page_size"] == 10
    svt.search_lease_knowledge_vertex("q", limit=0)
    assert seen["page_size"] == 1
