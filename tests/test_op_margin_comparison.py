from ai_chat import _op_margin_judgement
from components.score_calculation import _format_op_margin_comparison
from indicators import format_indicator_comparison, is_indicator_favorable


def test_negative_op_margin_is_not_described_as_high_even_if_better_than_negative_benchmark():
    assert _format_op_margin_comparison(-1.0, -3.0) == "赤字（業界平均より赤字幅は小さいが、強み扱いせず要注意）"
    assert "強み扱いは禁止" in _op_margin_judgement(-1.0, -3.0)


def test_negative_op_margin_below_positive_benchmark_is_red_flag():
    assert _format_op_margin_comparison(-1.0, 3.0) == "赤字（業界平均を下回り要注意）"
    assert _op_margin_judgement(-1.0, 3.0).startswith("赤字（業界平均を下回る")


def test_positive_op_margin_keeps_normal_benchmark_comparison():
    assert _format_op_margin_comparison(5.0, 3.0) == "平均より高い"
    assert _format_op_margin_comparison(2.0, 3.0) == "平均より低い"


def test_negative_equity_and_roa_are_not_favorable_even_if_benchmark_is_more_negative():
    assert not is_indicator_favorable("自己資本比率", -5.0, -10.0)
    assert not is_indicator_favorable("ROA", -1.0, -3.0)
    assert "債務超過" in format_indicator_comparison("自己資本比率", -5.0, -10.0)
    assert "強み扱いせず要注意" in format_indicator_comparison("ROA", -1.0, -3.0)
