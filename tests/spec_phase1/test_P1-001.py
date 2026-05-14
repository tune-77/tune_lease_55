"""
Auto-generated test skeleton for P1-001.
DO NOT EDIT the AC docstrings manually — regenerate via:
    python scripts/gen_tests_from_spec.py specs/phase1/P1-001-*.md
Each test_ac_xxx corresponds 1:1 with AC-xxx in the SPEC.
"""
import time
import sys
import os
import pytest

SPEC_ID = "P1-001"
PHASE = 1

# mobile_app/ をパスに追加
_MOBILE_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "mobile_app")
sys.path.insert(0, os.path.abspath(_MOBILE_APP_DIR))

from lease_rule_checks import check_lease_rules, LEGAL_USEFUL_LIFE_YEARS


def _codes(result) -> list[str]:
    return [w["code"] for w in result["warnings"]]


def _severities(result) -> list[str]:
    return [w["severity"] for w in result["warnings"]]


def test_ac_101_term_exceeds_legal_life_high_risk():
    """AC-101: 法定耐用年数超過で high_risk が返る
    電子計算機=4年=48ヶ月。lease_term=60ヶ月で超過。
    """
    assert LEGAL_USEFUL_LIFE_YEARS["電子計算機"] == 4
    result = check_lease_rules(lease_term_months=60, asset_type="電子計算機")
    assert result["status"] == "high_risk"
    assert any(w["code"] == "TERM_EXCEEDS_LEGAL_LIFE" and w["severity"] == "high"
               for w in result["warnings"])


def test_ac_102_term_within_legal_life_ok():
    """AC-102: 法定耐用年数内は ok が返る
    電子計算機=48ヶ月。lease_term=36ヶ月なら問題なし。
    """
    assert LEGAL_USEFUL_LIFE_YEARS["電子計算機"] == 4
    result = check_lease_rules(lease_term_months=36, asset_type="電子計算機")
    assert result["status"] == "ok"
    assert result["warnings"] == []


def test_ac_103_term_near_legal_life_medium():
    """AC-103: 法定耐用年数の90%超（近接）で medium 警告が返る
    閾値: 48 × 0.9 = 43.2ヶ月。45ヶ月は近接に該当。
    """
    assert LEGAL_USEFUL_LIFE_YEARS["電子計算機"] == 4
    result = check_lease_rules(lease_term_months=45, asset_type="電子計算機")
    codes = _codes(result)
    assert "TERM_NEAR_LEGAL_LIFE" in codes
    assert "TERM_EXCEEDS_LEGAL_LIFE" not in codes
    assert any(w["code"] == "TERM_NEAR_LEGAL_LIFE" and w["severity"] == "medium"
               for w in result["warnings"])


def test_ac_104_term_exceeds_expected_usage():
    """AC-104: 期待使用期間超過で medium 警告が返る
    電子計算機の max_years=4 (=48ヶ月)。60ヶ月で超過。
    """
    result = check_lease_rules(lease_term_months=60, asset_type="電子計算機")
    codes = _codes(result)
    assert "TERM_EXCEEDS_EXPECTED_USAGE" in codes
    assert any(w["code"] == "TERM_EXCEEDS_EXPECTED_USAGE" and w["severity"] == "medium"
               for w in result["warnings"])


def test_ac_105_insurance_not_covered_low():
    """AC-105: 動産保険未付保で low 警告が返る"""
    result = check_lease_rules(
        lease_term_months=36,
        asset_type="建設機械",
        insurance_applicable="未付保",
    )
    codes = _codes(result)
    assert "INSURANCE_NOT_COVERED" in codes
    assert any(w["code"] == "INSURANCE_NOT_COVERED" and w["severity"] == "low"
               for w in result["warnings"])


def test_ac_106_insurance_covered_no_warning():
    """AC-106: 保険付保済みで保険警告が出ない"""
    result = check_lease_rules(
        lease_term_months=36,
        asset_type="建設機械",
        insurance_applicable="付保済",
    )
    assert "INSURANCE_NOT_COVERED" not in _codes(result)


def test_ac_107_re_lease_insurance_not_covered():
    """AC-107: 再リース予定かつ再リース保険未付保で medium 警告が返る"""
    result = check_lease_rules(
        lease_term_months=36,
        asset_type="工作機械",
        is_re_lease=True,
        re_lease_insurance="未付保",
    )
    codes = _codes(result)
    assert "RE_LEASE_INSURANCE_NOT_COVERED" in codes
    assert any(w["code"] == "RE_LEASE_INSURANCE_NOT_COVERED" and w["severity"] == "medium"
               for w in result["warnings"])


def test_ac_108_no_re_lease_skips_re_lease_check():
    """AC-108: 再リース予定なしの場合、再リース保険チェックがスキップされる"""
    result = check_lease_rules(
        lease_term_months=36,
        asset_type="工作機械",
        is_re_lease=False,
        re_lease_insurance="未付保",
    )
    assert "RE_LEASE_INSURANCE_NOT_COVERED" not in _codes(result)


def test_ac_109_zero_lease_term_returns_error():
    """AC-109: 不正リース期間でエラーが返る（境界値: 0）"""
    result = check_lease_rules(lease_term_months=0, asset_type="電子計算機")
    assert result["status"] == "error"
    assert result["warnings"] == []


def test_ac_110_negative_lease_term_returns_error():
    """AC-110: 負のリース期間でエラーが返る"""
    result = check_lease_rules(lease_term_months=-12, asset_type="電子計算機")
    assert result["status"] == "error"
    assert result["warnings"] == []


def test_ac_111_unknown_asset_type_no_exception():
    """AC-111: マスタ不在の物件種別でエラーにならない"""
    result = check_lease_rules(lease_term_months=60, asset_type="存在しない種別ABC")
    codes = _codes(result)
    assert "TERM_EXCEEDS_LEGAL_LIFE" not in codes
    assert "TERM_EXCEEDS_EXPECTED_USAGE" not in codes


def test_ac_112_empty_asset_type_returns_unknown():
    """AC-112: asset_type 空文字で unknown が返る（保険も全て不明）"""
    result = check_lease_rules(lease_term_months=60, asset_type="")
    assert result["status"] == "unknown"
    assert result["warnings"] == []


def test_ac_113_multiple_warnings_high_risk():
    """AC-113: 複数警告が同時に返る（耐用年数超過 + 動産保険未付保）"""
    result = check_lease_rules(
        lease_term_months=72,
        asset_type="電子計算機",
        insurance_applicable="未付保",
    )
    codes = _codes(result)
    assert "TERM_EXCEEDS_LEGAL_LIFE" in codes
    assert "INSURANCE_NOT_COVERED" in codes
    assert len(result["warnings"]) >= 2
    assert result["status"] == "high_risk"


def test_ac_114_all_conditions_clear_returns_ok():
    """AC-114: 全チェック通過で ok が返る"""
    result = check_lease_rules(
        lease_term_months=24,
        asset_type="工作機械",
        is_re_lease=False,
        insurance_applicable="付保済",
    )
    assert result["status"] == "ok"
    assert result["warnings"] == []


def test_ac_115_performance_100_calls_within_5000ms():
    """AC-115: パフォーマンス要件（1件50ms以内 → 100回で5000ms以内）"""
    start = time.monotonic()
    for _ in range(100):
        check_lease_rules(
            lease_term_months=36,
            asset_type="電子計算機",
            is_re_lease=True,
            insurance_applicable="未付保",
            re_lease_insurance="未付保",
        )
    elapsed_ms = (time.monotonic() - start) * 1000
    assert elapsed_ms < 5000, f"100回呼び出しに {elapsed_ms:.1f}ms かかった（上限: 5000ms）"
