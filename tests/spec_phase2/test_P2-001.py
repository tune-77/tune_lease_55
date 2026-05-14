"""
Tests for P2-001: aurion/q_risk.py — 財務矛盾検知モジュール
AC-401〜AC-415 に対応するテスト。
"""
import time
import sys
import os

import pytest

SPEC_ID = "P2-001"
PHASE = 2

_MOBILE_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "mobile_app")
sys.path.insert(0, os.path.abspath(_MOBILE_APP_DIR))

from aurion.q_risk import detect_q_risk


class TestAC401:
    """AC-401: 粗利率異常（100%超）で FIN-CONTRADICT-001 が検知される
    Note: score=35(high×1) → level="caution"。AC-413 と整合する正しい動作。
    SPEC の "high_risk が返る" という表題は誤記（AC-413 のスコア計算式と矛盾）。
    """

    def test_ac_401(self):
        result = detect_q_risk(gross_profit=120.0, nenshu=100.0)
        assert "FIN-CONTRADICT-001" in result["patterns"]
        assert result["level"] != "ok"


class TestAC402:
    """AC-402: 粗利率異常（-50%未満）で FIN-CONTRADICT-001 が返る"""

    def test_ac_402(self):
        result = detect_q_risk(gross_profit=-60.0, nenshu=100.0)
        assert "FIN-CONTRADICT-001" in result["patterns"]


class TestAC403:
    """AC-403: 正常な粗利率で FIN-CONTRADICT-001 が出ない"""

    def test_ac_403(self):
        result = detect_q_risk(gross_profit=30.0, nenshu=100.0)
        assert "FIN-CONTRADICT-001" not in result["patterns"]


class TestAC404:
    """AC-404: 売上ゼロ・費用正の矛盾を検知する"""

    def test_ac_404(self):
        result = detect_q_risk(nenshu=0.0, dep_expense=5.0)
        assert "FIN-CONTRADICT-002" in result["patterns"]


class TestAC405:
    """AC-405: 営業利益が粗利を超える矛盾を検知する"""

    def test_ac_405(self):
        result = detect_q_risk(gross_profit=50.0, op_profit=60.0)
        assert "FIN-CONTRADICT-003" in result["patterns"]
        detail = next(d for d in result["pattern_details"] if d["code"] == "FIN-CONTRADICT-003")
        assert detail["severity"] == "high"


class TestAC406:
    """AC-406: 許容差1百万円以内は FIN-CONTRADICT-003 が出ない"""

    def test_ac_406(self):
        result = detect_q_risk(gross_profit=50.0, op_profit=51.0)
        assert "FIN-CONTRADICT-003" not in result["patterns"]


class TestAC407:
    """AC-407: リース残高/年商超過を検知する"""

    def test_ac_407(self):
        result = detect_q_risk(nenshu=100.0, lease_credit=60.0)
        assert "FIN-CONTRADICT-004" in result["patterns"]


class TestAC408:
    """AC-408: 総債務/年商超過を検知する"""

    def test_ac_408(self):
        result = detect_q_risk(nenshu=100.0, bank_credit=80.0, lease_credit=40.0)
        assert "FIN-CONTRADICT-005" in result["patterns"]


class TestAC409:
    """AC-409: 取得額/年商超過を検知する"""

    def test_ac_409(self):
        result = detect_q_risk(nenshu=100.0, acquisition_cost=35.0)
        assert "FIN-CONTRADICT-006" in result["patterns"]


class TestAC410:
    """AC-410: 機械設備あり・減価償却ゼロを検知する"""

    def test_ac_410(self):
        result = detect_q_risk(machines=10.0, depreciation=0.0)
        assert "FIN-CONTRADICT-007" in result["patterns"]
        detail = next(d for d in result["pattern_details"] if d["code"] == "FIN-CONTRADICT-007")
        assert detail["severity"] == "low"


class TestAC411:
    """AC-411: 純利益・営業利益の大幅乖離を検知する"""

    def test_ac_411(self):
        result = detect_q_risk(op_profit=10.0, net_income=-25.0)
        assert "FIN-CONTRADICT-008" in result["patterns"]


class TestAC412:
    """AC-412: 全入力ゼロで ok が返る"""

    def test_ac_412(self):
        result = detect_q_risk()
        assert result["score"] == 0
        assert result["level"] == "ok"
        assert result["patterns"] == []


class TestAC413:
    """AC-413: スコア計算（high × 1 件）→ score=35, caution"""

    def test_ac_413(self):
        result = detect_q_risk(gross_profit=120.0, nenshu=100.0)
        assert result["score"] == 35
        assert result["level"] == "caution"


class TestAC414:
    """AC-414: スコア計算（high × 2 件）→ score=70, high_risk"""

    def test_ac_414(self):
        # FIN-CONTRADICT-001 (high): gross_profit/nenshu=120%
        # FIN-CONTRADICT-003 (high): op_profit=130 > gross_profit=120 + 1
        result = detect_q_risk(gross_profit=120.0, nenshu=100.0, op_profit=130.0)
        assert "FIN-CONTRADICT-001" in result["patterns"]
        assert "FIN-CONTRADICT-003" in result["patterns"]
        assert result["score"] == 70
        assert result["level"] == "high_risk"


class TestAC415:
    """AC-415: パフォーマンス要件（100回連続で5000ms以内）"""

    def test_ac_415(self):
        start = time.time()
        for _ in range(100):
            detect_q_risk(
                gross_profit=50.0, op_profit=30.0, net_income=10.0,
                nenshu=100.0, dep_expense=5.0, depreciation=3.0,
                machines=20.0, bank_credit=40.0, lease_credit=20.0,
                acquisition_cost=10.0,
            )
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5000, f"100回の処理時間が {elapsed_ms:.1f}ms で 5000ms を超えました"
