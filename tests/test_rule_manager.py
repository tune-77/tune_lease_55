"""
rule_manager.py のユニットテスト

テスト対象:
  - evaluate_condition   : 数値条件判定
  - evaluate_custom_rules: カスタムルール適用
"""
import pytest
from rule_manager import evaluate_condition, evaluate_custom_rules


# ============================================================
# evaluate_condition
# ============================================================
class TestEvaluateCondition:
    """条件演算子ごとの真偽判定テスト"""

    @pytest.mark.parametrize("value, op, threshold, expected", [
        # 大なり・以上
        (100,   ">",  50,  True),
        (50,    ">",  50,  False),
        (51,    ">=", 50,  True),
        (50,    ">=", 50,  True),
        (49,    ">=", 50,  False),
        # 小なり・以下
        (30,    "<",  50,  True),
        (50,    "<",  50,  False),
        (49,    "<=", 50,  True),
        (50,    "<=", 50,  True),
        (51,    "<=", 50,  False),
        # 等号
        (100,   "=",  100, True),
        (100,   "==", 100, True),
        (99,    "=",  100, False),
        # 境界値（0・負数）
        (0,     ">",  -1,  True),
        (-5,    "<",  0,   True),
        (-5,    ">=", -5,  True),
    ])
    def test_numeric(self, value, op, threshold, expected):
        assert evaluate_condition(value, op, threshold) == expected

    def test_none_value_returns_false(self):
        assert evaluate_condition(None, ">", 10) is False

    def test_none_threshold_returns_false(self):
        assert evaluate_condition(10, ">", None) is False

    def test_both_none_returns_false(self):
        assert evaluate_condition(None, ">", None) is False

    def test_string_numeric_coercion(self):
        """文字列数値も数値として比較できる"""
        assert evaluate_condition("100", ">", 50) is True

    def test_unknown_op_returns_false(self):
        """未定義の演算子はFalse"""
        assert evaluate_condition(10, "!=", 5) is False


# ============================================================
# evaluate_custom_rules
# ============================================================
class TestEvaluateCustomRules:
    """カスタムルールの適用テスト"""

    def _make_rule(self, field, op, threshold, delta=0, status=None, label="テストルール"):
        """
        テスト用ルール辞書を生成するヘルパー。
        実装の実際のフォーマット:
          conditions[].target / .op / .value
          action_type: "deduct_score" | "force_status"
          action_value: 減点数 or ステータス文字列
        """
        if status:
            action_type  = "force_status"
            action_value = status   # "review" → 実装が "要審議" にマップ
        else:
            action_type  = "deduct_score"
            action_value = abs(delta)
        return {
            "label": label,
            "industry": "ALL",
            "conditions": [{"target": field, "op": op, "value": threshold}],
            "logic": "AND",
            "action_type":  action_type,
            "action_value": action_value,
        }

    def test_empty_rules_returns_zero(self):
        result = evaluate_custom_rules([], {"op_profit": -1000})
        assert result["score_delta"] == 0
        assert result["forced_status"] is None
        assert result["applied_reasons"] == []

    def test_matching_rule_applies_delta(self):
        rule = self._make_rule("op_profit", "<", 0, delta=-10)
        ctx = {"op_profit": -5000, "industry": "ALL"}
        result = evaluate_custom_rules([rule], ctx)
        assert result["score_delta"] == pytest.approx(-10)

    def test_non_matching_rule_no_effect(self):
        rule = self._make_rule("op_profit", "<", 0, delta=-10)
        ctx = {"op_profit": 5000, "industry": "ALL"}
        result = evaluate_custom_rules([rule], ctx)
        assert result["score_delta"] == pytest.approx(0)

    def test_forced_status_review(self):
        """"review" は実装内で "要審議" に正規化される"""
        rule = self._make_rule("op_profit", "<", 0, status="review")
        ctx = {"op_profit": -1, "industry": "ALL"}
        result = evaluate_custom_rules([rule], ctx)
        assert result["forced_status"] == "要審議"

    def test_multiple_rules_accumulate_delta(self):
        rules = [
            self._make_rule("op_profit", "<", 0, delta=-10, label="赤字ペナルティ"),
            self._make_rule("nenshu", "<", 10000, delta=-5, label="小規模ペナルティ"),
        ]
        ctx = {"op_profit": -100, "nenshu": 5000, "industry": "ALL"}
        result = evaluate_custom_rules(rules, ctx)
        assert result["score_delta"] == pytest.approx(-15)
        assert len(result["applied_reasons"]) == 2

    def test_industry_filter_excludes_other_industry(self):
        rule = self._make_rule("op_profit", "<", 0, delta=-10)
        rule["industry"] = "E 製造業"
        ctx = {"op_profit": -1000, "industry": "D 建設業"}
        result = evaluate_custom_rules([rule], ctx)
        # 業種が一致しないのでルール非適用
        assert result["score_delta"] == pytest.approx(0)

    def test_deduct_score_always_decrements(self):
        """
        deduct_score アクションは常に減点（abs値を引く）。
        加点ボーナスの仕組みは現実装では未対応。
        """
        rule = self._make_rule("nenshu", ">=", 100000, delta=5, label="deduct例")
        ctx = {"nenshu": 200000, "industry": "ALL"}
        result = evaluate_custom_rules([rule], ctx)
        # abs(5)=5 を引くので score_delta は -5
        assert result["score_delta"] == pytest.approx(-5)
