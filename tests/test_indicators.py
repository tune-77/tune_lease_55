"""
indicators.py のユニットテスト

テスト対象:
  - calculate_pd : 財務指標から簡易倒産確率を計算
"""
import pytest
from indicators import calculate_pd


class TestCalculatePd:
    """簡易倒産確率（calculate_pd）のテスト"""

    # ── 正常系：財務が良好な場合 ──────────────────────────────
    def test_healthy_company_low_risk(self):
        """自己資本30%超・流動比率150%超・営業利益率5%超 → リスク0"""
        pd_val = calculate_pd(equity=35.0, current=160.0, profit=6.0)
        assert pd_val == pytest.approx(0.0)

    def test_result_is_percentage(self):
        """戻り値は0〜100の範囲内"""
        for eq, cur, pr in [(5, 80, -5), (50, 200, 10), (20, 110, 1)]:
            val = calculate_pd(equity=eq, current=cur, profit=pr)
            assert 0.0 <= val <= 100.0

    # ── 自己資本比率のリスク加算 ─────────────────────────────
    def test_very_low_equity_adds_25(self):
        """自己資本 < 10% → +25"""
        val = calculate_pd(equity=5.0, current=200.0, profit=10.0)
        assert val == pytest.approx(25.0)

    def test_low_equity_adds_12(self):
        """自己資本 10〜20% → +12"""
        val = calculate_pd(equity=15.0, current=200.0, profit=10.0)
        assert val == pytest.approx(12.0)

    def test_moderate_equity_adds_5(self):
        """自己資本 20〜30% → +5"""
        val = calculate_pd(equity=25.0, current=200.0, profit=10.0)
        assert val == pytest.approx(5.0)

    def test_good_equity_adds_0(self):
        """自己資本 >= 30% → +0"""
        val = calculate_pd(equity=30.0, current=200.0, profit=10.0)
        assert val == pytest.approx(0.0)

    # ── 流動比率のリスク加算 ──────────────────────────────────
    def test_very_low_current_adds_20(self):
        """流動比率 < 100% → +20"""
        val = calculate_pd(equity=50.0, current=90.0, profit=10.0)
        assert val == pytest.approx(20.0)

    def test_low_current_adds_8(self):
        """流動比率 100〜120% → +8"""
        val = calculate_pd(equity=50.0, current=110.0, profit=10.0)
        assert val == pytest.approx(8.0)

    def test_moderate_current_adds_3(self):
        """流動比率 120〜150% → +3"""
        val = calculate_pd(equity=50.0, current=130.0, profit=10.0)
        assert val == pytest.approx(3.0)

    def test_good_current_adds_0(self):
        """流動比率 >= 150% → +0"""
        val = calculate_pd(equity=50.0, current=150.0, profit=10.0)
        assert val == pytest.approx(0.0)

    # ── 営業利益率のリスク加算 ────────────────────────────────
    def test_negative_profit_adds_30(self):
        """営業利益率 < 0 → +30"""
        val = calculate_pd(equity=50.0, current=200.0, profit=-1.0)
        assert val == pytest.approx(30.0)

    def test_very_low_profit_adds_10(self):
        """営業利益率 0〜2% → +10"""
        val = calculate_pd(equity=50.0, current=200.0, profit=1.0)
        assert val == pytest.approx(10.0)

    def test_low_profit_adds_4(self):
        """営業利益率 2〜5% → +4"""
        val = calculate_pd(equity=50.0, current=200.0, profit=3.0)
        assert val == pytest.approx(4.0)

    def test_good_profit_adds_0(self):
        """営業利益率 >= 5% → +0"""
        val = calculate_pd(equity=50.0, current=200.0, profit=5.0)
        assert val == pytest.approx(0.0)

    def test_profit_none_skips_profit_factor(self):
        """profit=None のとき営業利益率リスクは加算しない"""
        val_none = calculate_pd(equity=50.0, current=200.0, profit=None)
        val_good = calculate_pd(equity=50.0, current=200.0, profit=10.0)
        assert val_none == pytest.approx(val_good)

    # ── 最悪ケース・上限キャップ ──────────────────────────────
    def test_worst_case_capped_at_100(self):
        """全指標が最悪 → 合計75だが100を超えない"""
        val = calculate_pd(equity=1.0, current=50.0, profit=-10.0)
        assert val == pytest.approx(75.0)

    def test_combined_medium_risk(self):
        """中程度リスクの複合ケース: 自己資本15% + 流動比率110% = 20"""
        val = calculate_pd(equity=15.0, current=110.0, profit=None)
        assert val == pytest.approx(20.0)
