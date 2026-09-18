"""
scoring_core.py のユニットテスト

テスト対象:
  - _safe_sigmoid  : シグモイド関数（オーバーフロー対策）
  - _calculate_z   : 係数セットからロジット値を計算
  - _safe_float / _safe_int : 安全な型変換
  - _normalize_competitor_count_value / _normalize_deal_occurrence_value /
    _normalize_competitor_rate_value : 定性項目の正規化
  - compute_dscr_approx / compute_interest_coverage : 財務指標近似値
  - _apply_demo_food_service_score_cap : 公開デモ用スコア固定
"""
import pytest
import math
from scoring_core import (
    _safe_sigmoid,
    _calculate_z,
    _safe_float,
    _safe_int,
    _normalize_competitor_count_value,
    _normalize_deal_occurrence_value,
    _normalize_competitor_rate_value,
    compute_dscr_approx,
    compute_interest_coverage,
    _apply_demo_food_service_score_cap,
    DEMO_FOOD_SERVICE_COMPANY_NO,
    DEMO_FOOD_SERVICE_COMPANY_NAME,
    DEMO_FOOD_SERVICE_TARGET_SCORE,
)


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


# ============================================================
# _safe_float / _safe_int
# ============================================================
class TestSafeFloat:
    """安全なfloat変換のテスト"""

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=1.5) == 1.5

    def test_valid_numeric_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_valid_number(self):
        assert _safe_float(42) == pytest.approx(42.0)

    def test_invalid_string_returns_default(self):
        assert _safe_float("not-a-number") == 0.0
        assert _safe_float("abc", default=-1.0) == -1.0

    def test_invalid_type_returns_default(self):
        assert _safe_float([1, 2, 3]) == 0.0


class TestSafeInt:
    """安全なint変換のテスト"""

    def test_none_returns_default(self):
        assert _safe_int(None) == 0
        assert _safe_int(None, default=7) == 7

    def test_valid_numeric_string(self):
        assert _safe_int("42") == 42

    def test_valid_float_truncates(self):
        assert _safe_int(3.9) == 3

    def test_invalid_string_returns_default(self):
        assert _safe_int("not-a-number") == 0
        assert _safe_int("abc", default=-1) == -1

    def test_invalid_type_returns_default(self):
        assert _safe_int([1, 2, 3]) == 0


# ============================================================
# _normalize_competitor_count_value
# ============================================================
class TestNormalizeCompetitorCountValue:
    """競合社数正規化のテスト"""

    def test_none_returns_zero(self):
        assert _normalize_competitor_count_value(None) == 0.0

    def test_negative_number_returns_zero(self):
        assert _normalize_competitor_count_value(-5) == 0.0

    def test_number_capped_at_three(self):
        assert _normalize_competitor_count_value(10) == 3.0

    def test_number_passthrough_below_cap(self):
        assert _normalize_competitor_count_value(2) == 2.0

    @pytest.mark.parametrize("s, expected", [
        ("3社", 3.0),
        ("2社", 2.0),
        ("1社", 1.0),
        ("0社", 0.0),
        ("指名", 0.0),
        ("不明", 0.0),
    ])
    def test_string_values(self, s, expected):
        assert _normalize_competitor_count_value(s) == expected


# ============================================================
# _normalize_deal_occurrence_value
# ============================================================
class TestNormalizeDealOccurrenceValue:
    """発生経緯正規化のテスト"""

    def test_none_returns_zero(self):
        assert _normalize_deal_occurrence_value(None) == 0.0

    @pytest.mark.parametrize("s, expected", [
        ("相見積もり", 2.0),
        ("競争入札", 2.0),
        ("指名案件", 1.0),
        ("その他", 0.0),
        ("", 0.0),
    ])
    def test_string_values(self, s, expected):
        assert _normalize_deal_occurrence_value(s) == expected


# ============================================================
# _normalize_competitor_rate_value
# ============================================================
class TestNormalizeCompetitorRateValue:
    """競合提示金利正規化のテスト"""

    def test_invalid_returns_zero(self):
        assert _normalize_competitor_rate_value(None) == 0.0
        assert _normalize_competitor_rate_value("abc") == 0.0

    def test_zero_or_negative_returns_zero(self):
        assert _normalize_competitor_rate_value(0) == 0.0
        assert _normalize_competitor_rate_value(-1) == 0.0

    def test_fraction_scaled_to_percent(self):
        """1.0以下は% 表記への変換（×100）後に正規化される"""
        assert _normalize_competitor_rate_value(0.15) == pytest.approx(0.5, abs=1e-9)

    def test_permille_scaled_down(self):
        """1000超は千分率とみなして ÷1000 する"""
        assert _normalize_competitor_rate_value(1500) == pytest.approx(0.05, abs=1e-9)

    def test_result_clamped_to_unit_interval(self):
        val = _normalize_competitor_rate_value(100)
        assert 0.0 <= val <= 1.0
        assert val == pytest.approx(1.0)


# ============================================================
# compute_dscr_approx
# ============================================================
class TestComputeDscrApprox:
    """DSCR近似値計算のテスト"""

    def test_zero_denominator_returns_neutral(self):
        assert compute_dscr_approx({"op_profit": 500, "dep_expense": 0, "rent_expense": 0}) == 1.0

    def test_basic_calculation(self):
        result = compute_dscr_approx({"op_profit": 1500, "dep_expense": 500, "rent_expense": 500})
        assert result == pytest.approx(1.5)

    def test_falls_back_to_legacy_keys(self):
        """rieki/depreciation/rent の旧キーにフォールバックする"""
        result = compute_dscr_approx({"rieki": 1000, "depreciation": 500, "rent": 500})
        assert result == pytest.approx(1.0)

    def test_negative_op_profit(self):
        result = compute_dscr_approx({"op_profit": -500, "dep_expense": 500, "rent_expense": 0})
        assert result == pytest.approx(-1.0)


# ============================================================
# compute_interest_coverage
# ============================================================
class TestComputeInterestCoverage:
    """インタレスト・カバレッジ・レシオ計算のテスト"""

    def test_no_interest_returns_high_safety_value(self):
        assert compute_interest_coverage({"op_profit": 1000, "interest_expense": 0}) == 10.0

    def test_basic_calculation(self):
        result = compute_interest_coverage({"op_profit": 1000, "interest_expense": 200})
        assert result == pytest.approx(5.0)

    def test_negative_interest_treated_as_no_debt(self):
        result = compute_interest_coverage({"op_profit": 1000, "interest_expense": -100})
        assert result == 10.0


# ============================================================
# _apply_demo_food_service_score_cap
# ============================================================
class TestApplyDemoFoodServiceScoreCap:
    """公開デモ用スコア固定のテスト"""

    def test_unrelated_company_unaffected(self):
        score, adj = _apply_demo_food_service_score_cap({"company_no": "123456"}, 80.0)
        assert score == 80.0
        assert adj == 0.0

    def test_demo_company_no_below_target_unaffected(self):
        """既にターゲット以下なら調整しない"""
        inputs = {"company_no": DEMO_FOOD_SERVICE_COMPANY_NO}
        score, adj = _apply_demo_food_service_score_cap(inputs, 20.0)
        assert score == 20.0
        assert adj == 0.0

    def test_demo_company_no_above_target_capped(self):
        inputs = {"company_no": DEMO_FOOD_SERVICE_COMPANY_NO}
        score, adj = _apply_demo_food_service_score_cap(inputs, 80.0)
        assert score == DEMO_FOOD_SERVICE_TARGET_SCORE
        assert adj == pytest.approx(DEMO_FOOD_SERVICE_TARGET_SCORE - 80.0)

    def test_demo_company_name_matches_too(self):
        inputs = {"company_name": DEMO_FOOD_SERVICE_COMPANY_NAME}
        score, adj = _apply_demo_food_service_score_cap(inputs, 90.0)
        assert score == DEMO_FOOD_SERVICE_TARGET_SCORE
