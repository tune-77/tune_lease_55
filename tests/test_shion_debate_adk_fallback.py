"""ADK マルチエージェント討論フォールバックのテスト。

本番討論（multi_agent_screening）が失敗したときの独立フォールバック。要点:
  - google.adk が無い/失敗しても run_debate_adk_fallback は **常に妥当な結果**を返す
    （多層フォールバックで審査を止めない）。
  - 判定はスコア由来で scoring_core を単一ソースにする。
  - ADK 構造（ParallelAgent[3] → arbiter）は google.adk がある環境でのみ検証（無ければ skip）。
"""

from __future__ import annotations

import pytest

import api.shion_debate_adk as adk


def test_fallback_always_returns_valid_result_without_adk():
    # このCI/セッションには google.adk が無いため、必ず劣化経路を通る。
    out = adk.run_debate_adk_fallback({"score": 80, "company_name": "テスト社"})
    assert out["final"] in {"承認", "条件付承認", "否決"}
    assert "summary" in out and "conditions" in out
    assert out.get("_fallback")  # フォールバック印が付く


def test_parse_final_priority():
    # 否決 > 条件付承認 > 承認 の優先で拾う。
    assert adk._parse_final("総合すると判定：否決") == "否決"
    assert adk._parse_final("判定：条件付承認 とする") == "条件付承認"
    assert adk._parse_final("判定：承認") == "承認"
    assert adk._parse_final("判定不能") is None


def test_score_derived_minimal_shape():
    out = adk._score_derived_minimal({"score": 30})
    assert set(["final", "score", "mode", "summary", "conditions", "_fallback"]).issubset(out)
    assert out["final"] in {"承認", "条件付承認", "否決"}


def test_final_from_score_uses_scoring_core_thresholds():
    # numpy 未導入の環境では scoring_core を import できず安全側（条件付承認）へ。
    # numpy がある CI では scoring_core.APPROVAL_LINE/CONDITIONAL_LINE でスコア判定される。
    pytest.importorskip("numpy")
    from scoring_core import APPROVAL_LINE, CONDITIONAL_LINE

    assert adk._final_from_score({"score": APPROVAL_LINE}) == "承認"
    assert adk._final_from_score({"score": CONDITIONAL_LINE}) == "条件付承認"
    assert adk._final_from_score({"score": CONDITIONAL_LINE - 1}) == "否決"


def test_adk_structure_when_available():
    pytest.importorskip("google.adk")
    agent = adk.build_adk_debate_agent()
    # SequentialAgent[ ParallelAgent[skeptic, optimist, innovator], arbiter ]
    assert len(agent.sub_agents) == 2
    panel = agent.sub_agents[0]
    assert len(panel.sub_agents) == 3
    names = {a.name for a in panel.sub_agents}
    assert names == {"skeptic", "optimist", "innovator"}
