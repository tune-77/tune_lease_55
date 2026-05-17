"""
tests/spec_phase3/test_P3-001.py — P3-001 Acceptance Criteria テスト (AC-701〜AC-717)
"""
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mobile_app.aurion.stealth_competitor import detect_stealth_competitor


# AC-701: 競合未申告・低スプレッドで COMP-STEALTH-001 が検知される
def test_701_stealth_001_detected_when_no_competitor_low_spread():
    result = detect_stealth_competitor(spread_pred=1.2, base_rate=1.0, competitor=0, grade=5)
    assert "COMP-STEALTH-001" in result["patterns"]
    assert result["level"] == "caution"


# AC-702: 競合未申告でも spread が 1.5% 以上なら COMP-STEALTH-001 が出ない
def test_702_stealth_001_not_triggered_when_spread_at_boundary():
    result = detect_stealth_competitor(spread_pred=1.5, base_rate=1.0, competitor=0, grade=5)
    assert "COMP-STEALTH-001" not in result["patterns"]


# AC-703: 相場外れの申告競合金利で COMP-STEALTH-002 が検知される
def test_703_stealth_002_detected_when_competitor_rate_too_low():
    result = detect_stealth_competitor(
        spread_pred=1.8, base_rate=1.0, competitor=1, competitor_rate=1.1, grade=5
    )
    assert "COMP-STEALTH-002" in result["patterns"]


# AC-704: 申告競合金利が base_rate + 0.3 以上なら COMP-STEALTH-002 が出ない（境界値）
def test_704_stealth_002_not_triggered_at_boundary():
    result = detect_stealth_competitor(
        spread_pred=1.8, base_rate=1.0, competitor=1, competitor_rate=1.3, grade=5
    )
    assert "COMP-STEALTH-002" not in result["patterns"]


# AC-705: grade ≤ 3 の優良先でスプレッドが 1.0% 未満なら COMP-STEALTH-003 が検知される
def test_705_stealth_003_detected_grade_le3_low_spread():
    result = detect_stealth_competitor(spread_pred=0.8, base_rate=1.0, competitor=0, grade=2)
    assert "COMP-STEALTH-003" in result["patterns"]


# AC-706: grade 4〜6 でスプレッドが 0.8% 未満なら COMP-STEALTH-003 が検知される
def test_706_stealth_003_detected_grade_4to6_low_spread():
    result = detect_stealth_competitor(spread_pred=0.6, base_rate=1.0, competitor=0, grade=5)
    assert "COMP-STEALTH-003" in result["patterns"]


# AC-707: grade ≥ 7 でスプレッドが 0.5% 未満なら COMP-STEALTH-003 が検知される
def test_707_stealth_003_detected_grade_ge7_low_spread():
    result = detect_stealth_competitor(spread_pred=0.3, base_rate=1.0, competitor=0, grade=8)
    assert "COMP-STEALTH-003" in result["patterns"]


# AC-708: grade 4〜6 でスプレッドが 0.8% 以上なら COMP-STEALTH-003 が出ない（境界値）
def test_708_stealth_003_not_triggered_at_boundary_grade5():
    result = detect_stealth_competitor(spread_pred=0.8, base_rate=1.0, competitor=0, grade=5)
    assert "COMP-STEALTH-003" not in result["patterns"]


# AC-709: 申告競合金利と予測スプレッドの乖離が 1.5% 超で COMP-STEALTH-004 が検知される
def test_709_stealth_004_detected_when_large_divergence():
    # |1.2 - (4.0 - 1.0)| = |1.2 - 3.0| = 1.8 > 1.5
    result = detect_stealth_competitor(
        spread_pred=1.2, base_rate=1.0, competitor=1, competitor_rate=4.0, grade=5
    )
    assert "COMP-STEALTH-004" in result["patterns"]


# AC-710: 申告競合金利と予測スプレッドの乖離が 1.5% 以下では COMP-STEALTH-004 が出ない
def test_710_stealth_004_not_triggered_when_small_divergence():
    # |2.0 - (2.5 - 1.0)| = |2.0 - 1.5| = 0.5 ≤ 1.5
    result = detect_stealth_competitor(
        spread_pred=2.0, base_rate=1.0, competitor=1, competitor_rate=2.5, grade=5
    )
    assert "COMP-STEALTH-004" not in result["patterns"]


# AC-711: competitor=0 で competitor_rate が指定されても BR-302/BR-304 は発動しない
def test_711_competitor_zero_ignores_br302_br304():
    result = detect_stealth_competitor(
        spread_pred=2.0, base_rate=1.0, competitor=0, competitor_rate=1.1, grade=5
    )
    assert "COMP-STEALTH-002" not in result["patterns"]
    assert "COMP-STEALTH-004" not in result["patterns"]


# AC-712: 全パターン未検知で ok が返る
def test_712_all_patterns_clear_returns_ok():
    result = detect_stealth_competitor(spread_pred=2.5, base_rate=1.0, competitor=0, grade=5)
    assert result["score"] == 0
    assert result["level"] == "ok"
    assert result["patterns"] == []


# AC-713: スコア計算の検証（high × 1 件）
def test_713_score_high_x1():
    # competitor=0, spread=1.2 → COMP-STEALTH-001(high)
    # grade=5 → 4<=grade<=6 → threshold=0.8、spread=1.2 >= 0.8 → COMP-STEALTH-003 非該当
    result = detect_stealth_competitor(spread_pred=1.2, base_rate=1.0, competitor=0, grade=5)
    assert result["score"] == 35
    assert result["level"] == "caution"


# AC-714: スコア計算の検証（high×1 + medium×2 で high_risk）
def test_714_score_high1_medium2():
    # competitor=1, base_rate=3.0, competitor_rate=0.1, spread_pred=0.5, grade=5
    # 002: 0.1 > 0 かつ 0.1 < 3.0+0.3=3.3 → high(35)
    # 003: 4<=grade<=6 かつ 0.5 < 0.8 → medium(12)
    # 004: comp_spread=0.1-3.0=-2.9, |0.5-(-2.9)|=3.4 > 1.5 → medium(12)
    # score = 35+12+12 = 59 → high_risk
    result = detect_stealth_competitor(
        spread_pred=0.5, base_rate=3.0, competitor=1, competitor_rate=0.1, grade=5
    )
    assert "COMP-STEALTH-002" in result["patterns"]
    assert "COMP-STEALTH-003" in result["patterns"]
    assert "COMP-STEALTH-004" in result["patterns"]
    assert result["score"] == 59
    assert result["level"] == "high_risk"


# AC-715: grade 範囲外は 5 にクリップして計算を継続する（例外なし）
def test_715_grade_out_of_range_no_exception():
    result = detect_stealth_competitor(spread_pred=2.5, base_rate=1.0, competitor=0, grade=15)
    assert result is not None
    assert "score" in result
    assert "level" in result


# AC-716: competitor_rate が負値でも例外が発生しない（BR-302/BR-304 はスキップ）
def test_716_negative_competitor_rate_no_exception():
    result = detect_stealth_competitor(
        spread_pred=2.0, base_rate=1.0, competitor=1, competitor_rate=-0.5, grade=5
    )
    assert result is not None
    assert "COMP-STEALTH-002" not in result["patterns"]
    assert "COMP-STEALTH-004" not in result["patterns"]


# AC-717: パフォーマンス要件（100回連続で 5000ms 以内）
def test_717_performance_100_calls_under_5000ms():
    start = time.time()
    for _ in range(100):
        detect_stealth_competitor(
            spread_pred=1.2, base_rate=1.0, competitor=1, competitor_rate=1.1, grade=5
        )
    elapsed_ms = (time.time() - start) * 1000
    assert elapsed_ms < 5000, f"100回の処理が {elapsed_ms:.1f}ms かかった（上限5000ms）"
