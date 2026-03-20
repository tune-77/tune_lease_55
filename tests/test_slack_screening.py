# -*- coding: utf-8 -*-
"""
tests/test_slack_screening.py
==============================
slack_screening.py の Block Kit ブロック生成と基本フロー テスト。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def mock_slack_sdk():
    """slack_sdk が無い環境でもインポートできるようにする。"""
    with patch.dict("sys.modules", {
        "slack_sdk": MagicMock(),
        "slack_sdk.errors": MagicMock(),
    }):
        yield


def test_build_result_blocks_approval():
    from slack_screening import _build_result_blocks

    blocks = _build_result_blocks(
        company="テスト株式会社",
        industry="製造業",
        asset_name="旋盤",
        lease_amount_man=500.0,
        lease_term=36,
        score=78.0,
        decision="承認",
        hybrid=0.22,
        ai_prob=0.18,
        legacy=0.26,
        top5=["equity_ratio: 0.35", "roa: 0.04"],
        equity_ratio=0.35,
        op_ratio=0.06,
        roa=0.04,
        intuition=4,
    )
    assert isinstance(blocks, list)
    assert len(blocks) > 0
    # 最外層は attachment（color 付き）
    assert "color" in blocks[0]
    assert "blocks" in blocks[0]


def test_build_result_blocks_rejection():
    from slack_screening import _build_result_blocks

    blocks = _build_result_blocks(
        company="赤字商事",
        industry="小売業",
        asset_name="冷蔵庫",
        lease_amount_man=200.0,
        lease_term=60,
        score=32.0,
        decision="否決",
        hybrid=0.72,
        ai_prob=0.70,
        legacy=0.74,
        top5=[],
        equity_ratio=-0.05,
        op_ratio=-0.02,
        roa=-0.01,
        intuition=2,
    )
    assert isinstance(blocks, list)
    # 赤色 attachment
    assert blocks[0]["color"] == "#e01e5a"


def test_build_result_blocks_no_financials():
    """財務指標が None でもクラッシュしない。"""
    from slack_screening import _build_result_blocks

    blocks = _build_result_blocks(
        company="未入力商会",
        industry="その他",
        asset_name="—",
        lease_amount_man=0.0,
        lease_term=36,
        score=55.0,
        decision="保留",
        hybrid=0.45,
        ai_prob=0.45,
        legacy=0.45,
        top5=[],
        equity_ratio=None,
        op_ratio=None,
        roa=None,
        intuition=3,
    )
    assert isinstance(blocks, list)


def test_is_screening_active_false(tmp_path, monkeypatch):
    """セッションが存在しない場合 False を返す。"""
    import slack_screening

    store_path = tmp_path / "screening_store.json"
    monkeypatch.setattr(slack_screening, "_SESSION_FILE", str(store_path))

    result = slack_screening.is_screening_active("C_TEST_CHANNEL")
    assert result is False
