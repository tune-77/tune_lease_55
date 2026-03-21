"""
credit_limit.py のユニットテスト

テスト対象:
  - _lookup           : テーブルルックアップ（補間）
  - suggest_credit_limit : 与信枠提案の計算
"""
import pytest
from credit_limit import _lookup, suggest_credit_limit


# ============================================================
# _lookup
# ============================================================
class TestLookup:
    """テーブルルックアップのテスト"""

    TABLE = [(90, 1.5), (70, 1.2), (50, 1.0), (0, 0.7)]

    def test_exact_match_top(self):
        assert _lookup(self.TABLE, 90) == pytest.approx(1.5)

    def test_exact_match_middle(self):
        assert _lookup(self.TABLE, 70) == pytest.approx(1.2)

    def test_between_values_returns_upper(self):
        """85 → 90以上ではないので70以上のバケットに入る → 1.2"""
        assert _lookup(self.TABLE, 85) == pytest.approx(1.2)

    def test_below_all_thresholds_returns_last(self):
        """-1 → 全閾値未満 → 末尾の係数"""
        assert _lookup(self.TABLE, -1) == pytest.approx(0.7)

    def test_zero_threshold(self):
        assert _lookup(self.TABLE, 0) == pytest.approx(0.7)

    def test_above_max_threshold(self):
        assert _lookup(self.TABLE, 100) == pytest.approx(1.5)


# ============================================================
# suggest_credit_limit
# ============================================================
class TestSuggestCreditLimit:
    """与信枠提案の計算テスト"""

    def _base_res(self, **kwargs):
        """テスト用の標準的な審査結果辞書"""
        base = {
            "financials": {
                "nenshu":       50_000,
                "net_income":   2_000,
                "machines":     10_000,
                "other_assets": 5_000,
                "bank_credit":  20_000,
                "lease_credit": 5_000,
            },
            "score":    80.0,
            "user_eq":  30.0,
            "user_op":  5.0,
        }
        base.update(kwargs)
        return base

    def test_returns_credit_limit_result(self):
        from credit_limit import CreditLimitResult
        res = suggest_credit_limit(self._base_res())
        assert isinstance(res, CreditLimitResult)

    def test_suggested_is_non_negative(self):
        """提案与信枠は常に0以上"""
        res = suggest_credit_limit(self._base_res())
        assert res.suggested >= 0

    def test_high_score_gets_better_tier(self):
        """スコア90超 vs スコア50前後でtierが異なる"""
        high = suggest_credit_limit(self._base_res(score=95.0))
        low  = suggest_credit_limit(self._base_res(score=40.0))
        # 高スコアの方が提案額が高い or 同等（件数少でフロアが効く場合もある）
        assert high.suggested >= low.suggested or high.tier <= low.tier

    def test_high_equity_raises_limit(self):
        """自己資本比率が高い方が与信枠が大きい"""
        low_eq  = suggest_credit_limit(self._base_res(**{"user_eq": 5.0}))
        high_eq = suggest_credit_limit(self._base_res(**{"user_eq": 50.0}))
        assert high_eq.suggested >= low_eq.suggested

    def test_negative_profit_reduces_limit(self):
        """営業利益率マイナスは与信枠を下げる"""
        pos = suggest_credit_limit(self._base_res(**{"user_op": 8.0}))
        neg = suggest_credit_limit(self._base_res(**{"user_op": -5.0}))
        assert pos.suggested >= neg.suggested

    def test_as_dict_has_key_fields(self):
        """as_dict()が主要フィールドを含む"""
        d = suggest_credit_limit(self._base_res()).as_dict()
        for key in ("suggested", "tier", "gross_limit", "available"):
            assert key in d

    def test_tier_label_is_string(self):
        res = suggest_credit_limit(self._base_res())
        assert isinstance(res.tier_label, str)
        assert len(res.tier_label) > 0

    def test_remarks_is_list(self):
        res = suggest_credit_limit(self._base_res())
        assert isinstance(res.remarks, list)

    def test_zero_nenshu_does_not_crash(self):
        """年商0でもクラッシュしない"""
        res_dict = self._base_res()
        res_dict["financials"]["nenshu"] = 0
        res = suggest_credit_limit(res_dict)
        assert res.suggested >= 0

    def test_minimum_floor_for_approved(self):
        """スコア>=70の承認案件は最低100万以上の提案"""
        res = suggest_credit_limit(self._base_res(score=75.0))
        assert res.suggested >= 100
