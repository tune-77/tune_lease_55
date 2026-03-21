# -*- coding: utf-8 -*-
"""
tests/test_explainer.py
=======================
scoring/explainer.py の自然言語変換テスト。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scoring.explainer import explain_top_reasons, _explain_one


class TestExplainOne:
    def test_equity_ratio_negative(self):
        result = _explain_one("equity_ratio", -0.05)
        assert "マイナス" in result or "債務超過" in result

    def test_equity_ratio_low(self):
        result = _explain_one("equity_ratio", 0.05)
        assert "低" in result

    def test_equity_ratio_high(self):
        result = _explain_one("equity_ratio", 0.60)
        assert "安定" in result or "高" in result

    def test_roa_negative(self):
        result = _explain_one("roa", -0.01)
        assert "マイナス" in result

    def test_lease_coverage_below_one(self):
        result = _explain_one("lease_coverage_ratio", 0.8)
        assert "不足" in result or "下回" in result

    def test_negative_equity_flag_true(self):
        result = _explain_one("negative_equity_flag", 1.0)
        assert "債務超過" in result

    def test_negative_equity_flag_false(self):
        result = _explain_one("negative_equity_flag", 0.0)
        # フラグOFFは閾値に引っかからないのでデフォルトテンプレート
        assert isinstance(result, str)

    def test_unknown_feature_returns_string(self):
        result = _explain_one("unknown_feature_xyz", 0.5)
        assert isinstance(result, str)
        assert len(result) > 0


class TestExplainTopReasons:
    def test_basic(self):
        reasons = ["equity_ratio: 0.25", "roa: -0.02"]
        results = explain_top_reasons(reasons)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_empty_list(self):
        assert explain_top_reasons([]) == []

    def test_malformed_item_passthrough(self):
        reasons = ["something without colon"]
        results = explain_top_reasons(reasons)
        assert results == ["something without colon"]

    def test_non_numeric_value_passthrough(self):
        reasons = ["equity_ratio: not_a_number"]
        results = explain_top_reasons(reasons)
        assert results == ["equity_ratio: not_a_number"]

    def test_with_industry(self):
        reasons = ["equity_ratio: 0.4"]
        results = explain_top_reasons(reasons, industry="製造業")
        assert len(results) == 1

    def test_five_reasons(self):
        reasons = [
            "equity_ratio: 0.3",
            "roa: 0.02",
            "operating_margin: 0.05",
            "lease_coverage_ratio: 1.5",
            "debt_ratio: 0.6",
        ]
        results = explain_top_reasons(reasons)
        assert len(results) == 5
