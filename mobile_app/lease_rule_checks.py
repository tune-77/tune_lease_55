"""
P1-001: リースルールチェックモジュール

スコア計算ロジックには一切触れず、警告（warnings）を返すだけのサイドカー設計。
"""
from __future__ import annotations

import logging
import sys
import os
from typing import TypedDict, Literal, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

# 法定耐用年数マスタ（法人税法施行令別表第一に基づく代表値）単位：年
LEGAL_USEFUL_LIFE_YEARS: dict[str, int] = {
    # 電子機器・OA機器
    "電子計算機": 4,
    "複写機": 5,
    "ファクシミリ": 5,
    "複合機": 5,
    "通信機器": 10,
    "カメラ": 5,
    # 工業機械
    "工作機械": 10,
    "印刷機械": 10,
    "農業機械": 7,
    "建設機械": 6,
    "フォークリフト": 3,
    # 輸送機器
    "自動車（普通）": 6,
    "自動車（小型）": 4,
    "トラック": 5,
    "バス": 5,
    # 医療機器
    "医療機器": 6,
    "歯科用機器": 7,
    # 設備機器
    "エアコン": 6,
    "冷凍・冷蔵機器": 6,
    "厨房機器": 8,
    "運搬機具": 5,
    "自動販売機": 5,
}


class WarningItem(TypedDict):
    code: str
    severity: str
    message: str
    source: str


class RuleCheckResult(TypedDict):
    status: str
    warnings: list[WarningItem]


def _determine_status(warnings: list[WarningItem], asset_type: str) -> str:
    if any(w["severity"] == "high" for w in warnings):
        return "high_risk"
    if any(w["severity"] in ("high", "medium") for w in warnings):
        return "warning"
    if any(w["severity"] == "low" for w in warnings):
        return "warning"
    if not asset_type:
        return "unknown"
    return "ok"


def check_lease_rules(
    lease_term_months: int,
    asset_type: str = "",
    is_re_lease: bool = False,
    insurance_applicable: str = "不明",
    re_lease_insurance: str = "不明",
) -> RuleCheckResult:
    """
    リースルールチェックを実行し、警告リストを返す。

    スコア計算には影響しない。例外を外部に伝播させない設計。
    """
    try:
        # BR-108: 不正入力ガード
        if lease_term_months <= 0:
            return RuleCheckResult(status="error", warnings=[])

        warnings: list[WarningItem] = []

        # BR-101 / BR-102: 法定耐用年数チェック
        if asset_type and asset_type in LEGAL_USEFUL_LIFE_YEARS:
            legal_months = LEGAL_USEFUL_LIFE_YEARS[asset_type] * 12
            if lease_term_months > legal_months:
                # BR-101: 超過
                warnings.append(WarningItem(
                    code="TERM_EXCEEDS_LEGAL_LIFE",
                    severity="high",
                    message=f"リース期間（{lease_term_months}ヶ月）が法定耐用年数（{legal_months}ヶ月）を超過しています。",
                    source="法人税法施行令別表第一（器具及び備品）",
                ))
            elif lease_term_months > legal_months * 0.9:
                # BR-102: 近接
                warnings.append(WarningItem(
                    code="TERM_NEAR_LEGAL_LIFE",
                    severity="medium",
                    message=f"リース期間（{lease_term_months}ヶ月）が法定耐用年数（{legal_months}ヶ月）の90%を超えています。",
                    source="法人税法施行令別表第一（器具及び備品）",
                ))
        # BR-106: マスタ不在 → スキップ

        # BR-103: 期待使用期間チェック
        if asset_type:
            try:
                from expected_usage_period import find_item_by_name
                item = find_item_by_name(asset_type)
                if item is not None:
                    expected_period_months = int(item["max_years"]) * 12
                    if lease_term_months > expected_period_months:
                        warnings.append(WarningItem(
                            code="TERM_EXCEEDS_EXPECTED_USAGE",
                            severity="medium",
                            message=f"リース期間（{lease_term_months}ヶ月）が期待使用期間（{expected_period_months}ヶ月）を超過しています。",
                            source="期待使用期間.json マスタ（社内基準）",
                        ))
                # BR-107: find_item_by_name が None → スキップ
            except Exception:
                pass  # BR-109: 例外サイレント（期待使用期間モジュール障害時も続行）

        # BR-104: 動産保険未付保チェック
        if insurance_applicable == "未付保":
            warnings.append(WarningItem(
                code="INSURANCE_NOT_COVERED",
                severity="low",
                message="動産保険が未付保です。物件滅失リスクに注意してください。",
                source="動産総合保険付保推奨事項",
            ))

        # BR-105: 再リース保険未付保チェック
        if is_re_lease and re_lease_insurance == "未付保":
            warnings.append(WarningItem(
                code="RE_LEASE_INSURANCE_NOT_COVERED",
                severity="medium",
                message="再リース予定にもかかわらず再リース保険が未付保です。",
                source="再リース期間中の保険継続確認要件",
            ))

        if warnings:
            codes = [w["code"] for w in warnings]
            print(f"[lease_rule_checks] asset_type={asset_type!r} lease_term_months={lease_term_months} warnings={codes}")

        status = _determine_status(warnings, asset_type)
        return RuleCheckResult(status=status, warnings=warnings)

    except Exception as e:
        # BR-109: 例外サイレント処理
        logger.warning(f"[lease_rule_checks] unexpected error: {e}")
        return RuleCheckResult(status="unknown", warnings=[])
