# -*- coding: utf-8 -*-
"""
tests/test_chat_wizard_steps.py
================================
chat_wizard.py のステップリアクション・ユーモアコメントのロジックテスト。
Streamlit に依存する部分は monkeypatch で差し替える。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def mock_streamlit():
    """streamlit をモックしてインポートエラーを回避する。"""
    st_mock = MagicMock()
    st_mock.session_state = {}
    with patch.dict("sys.modules", {"streamlit": st_mock}):
        yield st_mock


def test_humor_comments_not_empty():
    import components.chat_wizard as cw
    assert len(cw._HUMOR_COMMENTS) > 0
    for comment in cw._HUMOR_COMMENTS:
        assert isinstance(comment, str)
        assert len(comment) > 0


def test_step_reaction_rieki_negative():
    import components.chat_wizard as cw

    reaction = cw._get_step_reaction("pl", {"rieki": -100})
    assert reaction != ""
    assert "赤字" in reaction


def test_step_reaction_rieki_zero():
    import components.chat_wizard as cw

    reaction = cw._get_step_reaction("pl", {"rieki": 0})
    assert reaction != ""
    assert "トントン" in reaction or "損益" in reaction


def test_step_reaction_rieki_positive():
    import components.chat_wizard as cw

    reaction = cw._get_step_reaction("pl", {"rieki": 5000})
    assert reaction == ""


def test_step_reaction_net_assets_negative():
    import components.chat_wizard as cw

    reaction = cw._get_step_reaction("assets_main", {"net_assets": -500})
    assert "マイナス" in reaction or "純資産" in reaction


def test_step_reaction_credit_low_grade():
    import components.chat_wizard as cw

    for grade in ["C", "D", "E"]:
        reaction = cw._get_step_reaction("credit", {"grade": grade})
        assert reaction != "", f"grade={grade} で反応がない"


def test_step_reaction_credit_good_grade():
    import components.chat_wizard as cw

    reaction = cw._get_step_reaction("credit", {"grade": "A"})
    assert reaction == ""


def test_step_reaction_deal_with_competitor():
    import components.chat_wizard as cw

    reaction = cw._get_step_reaction("deal", {"competitor": "有"})
    assert "競合" in reaction


def test_step_reaction_unknown_step():
    import components.chat_wizard as cw

    reaction = cw._get_step_reaction("unknown_step", {})
    assert reaction == ""
