"""
scoring_core.py のユニットテスト

テスト対象:
  - _safe_sigmoid  : シグモイド関数（オーバーフロー対策）
  - _calculate_z   : 係数セットからロジット値を計算
"""
import pytest
import math
from scoring_core import _safe_sigmoid, _calculate_z


# ============================================================
# _safe_sigmoid
# ============================================================
class TestSafeSigmoid:
    """シグモイド関数のテスト"""

    def test_zero_returns_half(self):
        assert _safe_sigmoid(0) == pytest.approx(0.5)

    def test_positive_large_returns_near_one(self):
        assert _safe_sigmoid(100) == pytest.approx(1.0)

    def test_negative_large_returns_near_zero(self):
        assert _safe_sigmoid(-100) == pytest.approx(0.0)

    def test_overflow_positive_capped(self):
        """700超でオーバーフローせず1.0を返す"""
        assert _safe_sigmoid(1000) == pytest.approx(1.0)

    def test_overflow_negative_capped(self):
        """-700未満でオーバーフローせず0.0を返す"""
        assert _safe_sigmoid(-1000) == pytest.approx(0.0)

    def test_result_between_0_and_1(self):
        for x in [-500, -10, -1, 0, 1, 10, 500]:
            val = _safe_sigmoid(x)
            assert 0.0 <= val <= 1.0

    def test_monotone_increasing(self):
        """シグモイドは単調増加"""
        xs = [-5, -1, 0, 1, 5]
        vals = [_safe_sigmoid(x) for x in xs]
        assert vals == sorted(vals)

    def test_symmetry(self):
        """sigmoid(-x) + sigmoid(x) == 1"""
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert _safe_sigmoid(x) + _safe_sigmoid(-x) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("x, expected", [
        (0,   0.5),
        (1,   1 / (1 + math.exp(-1))),
        (-1,  1 / (1 + math.exp(1))),
        (2,   1 / (1 + math.exp(-2))),
    ])
    def test_known_values(self, x, expected):
        assert _safe_sigmoid(x) == pytest.approx(expected, abs=1e-9)


# ============================================================
# _calculate_z
# ============================================================
class TestCalculateZ:
    """ロジット値計算のテスト"""

    BASE_COEFF = {
        "intercept":          0.0,
        "sales_log":          0.1,
        "bank_credit_log":    0.05,
        "lease_credit_log":   0.03,
        "op_profit":          0.0001,
        "ord_profit":         0.0001,
        "net_income":         0.0001,
        "machines":           0.0,
        "other_assets":       0.0,
        "rent":               0.0,
        "depreciation":       0.0,
        "dep_expense":        0.0,
        "rent_expense":       0.0,
        "contracts":          0.0,
        "lease_term":         0.0,
        "acquisition_cost":   0.0,
        "lease_asset_score":  0.0,
        "ind_medical":        0.5,
        "ind_transport":      0.3,
        "ind_construction":   0.1,
        "ind_manufacturing": -0.2,
        "ind_service":        0.0,
    }

    def _data(self, **kwargs):
        base = {
            "industry_major": "D 建設業",
            "nenshu": 10_000,
            "bank_credit": 5_000,
            "lease_credit": 1_000,
            "op_profit": 500,
            "ord_profit": 400,
            "net_income": 300,
            "machines": 0,
            "other_assets": 0,
            "rent": 0,
            "depreciation": 0,
            "dep_expense": 0,
            "rent_expense": 0,
            "contracts": 0,
            "lease_term": 0,
            "acquisition_cost": 0,
            "lease_asset_score": 0,
        }
        base.update(kwargs)
        return base

    def test_returns_float(self):
        z = _calculate_z(self._data(), self.BASE_COEFF)
        assert isinstance(z, float)

    def test_intercept_only(self):
        """全係数0・切片1 → z=1"""
        zero_coeff = {k: 0.0 for k in self.BASE_COEFF}
        zero_coeff["intercept"] = 1.0
        data = self._data(nenshu=0, bank_credit=0, lease_credit=0,
                          op_profit=0, ord_profit=0, net_income=0)
        z = _calculate_z(data, zero_coeff)
        assert z == pytest.approx(1.0)

    def test_industry_construction_bonus(self):
        """建設業は ind_construction 係数が加算される"""
        coeff = dict(self.BASE_COEFF)
        coeff["ind_construction"] = 2.0
        z = _calculate_z(self._data(industry_major="D 建設業",
                                     nenshu=0, bank_credit=0, lease_credit=0,
                                     op_profit=0, ord_profit=0, net_income=0),
                         coeff)
        assert z == pytest.approx(2.0)

    def test_industry_medical_bonus(self):
        """医療業は ind_medical 係数が加算される"""
        coeff = {k: 0.0 for k in self.BASE_COEFF}
        coeff["ind_medical"] = 3.0
        z = _calculate_z(self._data(industry_major="P 医療業",
                                     nenshu=0, bank_credit=0, lease_credit=0,
                                     op_profit=0, ord_profit=0, net_income=0),
                         coeff)
        assert z == pytest.approx(3.0)

    def test_sales_log_effect(self):
        """年商が大きいほどzが増加（sales_log > 0 の場合）"""
        coeff = {k: 0.0 for k in self.BASE_COEFF}
        coeff["sales_log"] = 1.0
        z_small = _calculate_z(self._data(nenshu=1_000, bank_credit=0, lease_credit=0,
                                           op_profit=0, ord_profit=0, net_income=0,
                                           industry_major=""),
                                coeff)
        z_large = _calculate_z(self._data(nenshu=100_000, bank_credit=0, lease_credit=0,
                                           op_profit=0, ord_profit=0, net_income=0,
                                           industry_major=""),
                                coeff)
        assert z_large > z_small

    def test_zero_nenshu_skips_log(self):
        """nenshu=0 のとき sales_log が加算されない（ゼロ除算なし）"""
        coeff = {k: 0.0 for k in self.BASE_COEFF}
        coeff["sales_log"] = 999.9
        z = _calculate_z(self._data(nenshu=0, bank_credit=0, lease_credit=0,
                                     op_profit=0, ord_profit=0, net_income=0,
                                     industry_major=""),
                          coeff)
        assert z == pytest.approx(0.0)

    def test_op_profit_negative_decreases_z(self):
        """営業利益がマイナスだとzが下がる（op_profit係数が正の場合）"""
        coeff = {k: 0.0 for k in self.BASE_COEFF}
        coeff["op_profit"] = 0.001
        z_pos = _calculate_z(self._data(nenshu=0, bank_credit=0, lease_credit=0,
                                         op_profit=1000, ord_profit=0, net_income=0,
                                         industry_major=""),
                              coeff)
        z_neg = _calculate_z(self._data(nenshu=0, bank_credit=0, lease_credit=0,
                                         op_profit=-1000, ord_profit=0, net_income=0,
                                         industry_major=""),
                              coeff)
        assert z_pos > z_neg
