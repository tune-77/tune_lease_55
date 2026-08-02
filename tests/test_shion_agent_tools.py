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
    # 既存2ツール + 読み取り専用DBツール11個 = 13個（Vertex は既定で無効）。
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
