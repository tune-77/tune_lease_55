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
    "get_portfolio_stats",
    "get_weekly_trend",
    "get_system_overview",
    "get_recent_errors",
    "get_pipeline_item_details",
    "recall_judgment_memory",
    "build_judgment_preview",
    "search_obsidian_context",
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
    # 既存2ツール + 読み取り専用DBツール10個 = 12個。
    assert names == {"get_industry_benchmark", "assess_risk_level"} | EXPECTED_DB_TOOLS
    assert len(sa._AGENT_TOOL_FUNCS) == 12
