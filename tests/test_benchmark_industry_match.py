"""REV-482: 業種の業界目安照合（コード体系ずれ対応・照合不可時は不明扱い）"""
from scoring_core import _industry_name, _resolve_benchmark, run_quick_scoring

BENCH = {
    "21 金属製品製造業": {"op_margin": 4.04, "equity_ratio": 45.0, "lease_cost_ratio": 1.0},
    "24 生産用機械器具製造業": {"op_margin": 7.84, "equity_ratio": 50.0},
    "_last_updated": "2026-06-07",
}


def test_industry_name_strips_leading_code_only():
    assert _industry_name("24 金属製品製造業") == "金属製品製造業"
    assert _industry_name("50-55 各種卸売業") == "各種卸売業"
    assert _industry_name("R サービス業(他に分類されないもの)") == "R サービス業(他に分類されないもの)"
    assert _industry_name("サービス業全般") == "サービス業全般"


def test_exact_match_first():
    assert _resolve_benchmark(BENCH, "21 金属製品製造業")["op_margin"] == 4.04


def test_name_match_across_code_systems():
    # 案件側は JSIC の 24=金属製品。ベンチマーク側の 24（生産用機械）に誤照合しないこと
    assert _resolve_benchmark(BENCH, "24 金属製品製造業")["op_margin"] == 4.04
    assert _resolve_benchmark(BENCH, "26 生産用機械器具製造業")["op_margin"] == 7.84


def test_unmatched_returns_none():
    assert _resolve_benchmark(BENCH, "サービス業全般") is None
    assert _resolve_benchmark(BENCH, "") is None
    assert _resolve_benchmark(BENCH, None) is None
    assert _resolve_benchmark(BENCH, "_last_updated") is None


def _inputs(industry_sub):
    return {
        "nenshu": 500000, "op_profit": 25000, "ord_profit": 20000, "net_income": 12000,
        "net_assets": 150000, "total_assets": 400000, "industry_major": "E 製造業",
        "industry_sub": industry_sub, "grade": "②4-6 (標準)", "customer_type": "既存先",
        "asset_score": 60,
    }


def test_run_quick_scoring_unmatched_is_unknown():
    r = run_quick_scoring(_inputs("サービス業全般"))
    assert r["benchmark_matched"] is False
    assert r["bench_op_margin"] is None and r["bench_equity_ratio"] is None
    assert r["bench_lease_cost_ratio"] is None
    assert "照合不可" in r["comparison"] and "平均より" not in r["comparison"]
    assert r["lease_ratio_adj"] == 0.0


def test_run_quick_scoring_matches_by_name():
    r = run_quick_scoring(_inputs("24 金属製品製造業"))
    assert r["benchmark_matched"] is True
    assert r["bench_op_margin"] is not None and r["bench_op_margin"] > 0
    assert "平均より" in r["comparison"]
