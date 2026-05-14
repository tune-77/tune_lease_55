"""
aurion/q_risk.py — 財務矛盾検知モジュール (P2-001)

RF/LGBM スコアリングには影響しない参考値（サイドカー）として動作する。
例外を外部に伝播させない設計。入力はすべて百万円単位。
"""
from typing import TypedDict


class PatternDetail(TypedDict):
    code: str
    severity: str
    message: str
    values: dict


class QRiskResult(TypedDict):
    score: int
    level: str
    patterns: list
    pattern_details: list


def detect_q_risk(
    gross_profit: float = 0.0,
    op_profit: float = 0.0,
    net_income: float = 0.0,
    nenshu: float = 0.0,
    dep_expense: float = 0.0,
    depreciation: float = 0.0,
    machines: float = 0.0,
    bank_credit: float = 0.0,
    lease_credit: float = 0.0,
    acquisition_cost: float = 0.0,
) -> QRiskResult:
    """
    財務データから矛盾パターンを検知し、Q_risk スコアを返す。

    スコア計算（RF/LGBM）には影響しない参考値。
    例外を外部に伝播させない設計。入力はすべて百万円単位。

    Returns:
        QRiskResult: score, level, patterns, pattern_details を持つ dict
    """
    try:
        details: list[PatternDetail] = []

        # BR-201: 粗利率異常検知
        if nenshu > 0:
            gpm = gross_profit / nenshu
            if gpm < -0.5 or gpm > 1.0:
                details.append(PatternDetail(
                    code="FIN-CONTRADICT-001",
                    severity="high",
                    message="粗利率が異常値（-50%未満または100%超）です。財務諸表の誤入力・粉飾の可能性があります。",
                    values={"gross_profit": gross_profit, "nenshu": nenshu, "gpm": round(gpm, 4)},
                ))

        # BR-202: 売上ゼロ・費用正の矛盾検知
        if nenshu == 0 and dep_expense > 0:
            details.append(PatternDetail(
                code="FIN-CONTRADICT-002",
                severity="high",
                message="売上ゼロにもかかわらず支払リース料が発生しています。データ入力ミスまたは休眠状態の可能性があります。",
                values={"nenshu": nenshu, "dep_expense": dep_expense},
            ))

        # BR-203: 営業利益・粗利矛盾検知（1百万円の許容差）
        if op_profit > gross_profit + 1:
            details.append(PatternDetail(
                code="FIN-CONTRADICT-003",
                severity="high",
                message="営業利益が粗利益を超えています。営業利益=粗利-販管費であるため、数学的に不可能な値です。",
                values={"op_profit": op_profit, "gross_profit": gross_profit, "diff": round(op_profit - gross_profit, 4)},
            ))

        # BR-204: リース残高/年商 超過検知（nenshu > 0 ガード）
        if nenshu > 0 and lease_credit / nenshu > 0.5:
            details.append(PatternDetail(
                code="FIN-CONTRADICT-004",
                severity="medium",
                message="リース残高が年商の50%を超えています。過剰なオフバランス活用・実質的な過剰債務の可能性があります。",
                values={"lease_credit": lease_credit, "nenshu": nenshu, "ratio": round(lease_credit / nenshu, 4)},
            ))

        # BR-205: 総債務/年商 超過検知（nenshu > 0 ガード）
        if nenshu > 0 and (bank_credit + lease_credit) / nenshu > 1.0:
            details.append(PatternDetail(
                code="FIN-CONTRADICT-005",
                severity="medium",
                message="総借入（銀行+リース）が年商を超えています。返済能力に疑義が生じます。",
                values={"bank_credit": bank_credit, "lease_credit": lease_credit, "nenshu": nenshu,
                        "ratio": round((bank_credit + lease_credit) / nenshu, 4)},
            ))

        # BR-206: 取得額/年商 超過検知（nenshu > 0 ガード）
        if nenshu > 0 and acquisition_cost / nenshu > 0.3:
            details.append(PatternDetail(
                code="FIN-CONTRADICT-006",
                severity="medium",
                message="今回リース取得額が年商の30%を超えています。規模に対して過大な設備投資を示します。",
                values={"acquisition_cost": acquisition_cost, "nenshu": nenshu,
                        "ratio": round(acquisition_cost / nenshu, 4)},
            ))

        # BR-207: 減価償却費欠落検知
        if machines > 1.0 and depreciation == 0:
            details.append(PatternDetail(
                code="FIN-CONTRADICT-007",
                severity="low",
                message="機械設備残高があるにもかかわらず減価償却費がゼロです。入力漏れの可能性があります。",
                values={"machines": machines, "depreciation": depreciation},
            ))

        # BR-208: 純利益・営業利益の大幅乖離検知
        if op_profit > 0 and net_income < op_profit * -2.0:
            details.append(PatternDetail(
                code="FIN-CONTRADICT-008",
                severity="medium",
                message="営業利益が正であるにもかかわらず純利益が著しくマイナスです。特別損失や隠れた債務の可能性があります。",
                values={"op_profit": op_profit, "net_income": net_income,
                        "threshold": round(op_profit * -2.0, 4)},
            ))

        high_count   = sum(1 for d in details if d["severity"] == "high")
        medium_count = sum(1 for d in details if d["severity"] == "medium")
        low_count    = sum(1 for d in details if d["severity"] == "low")
        score = min(100, high_count * 35 + medium_count * 12 + low_count * 4)

        if score <= 19:
            level = "ok"
        elif score <= 49:
            level = "caution"
        else:
            level = "high_risk"

        patterns = [d["code"] for d in details]

        if patterns:
            print(f"[aurion.q_risk] patterns={patterns} score={score}")

        return QRiskResult(
            score=score,
            level=level,
            patterns=patterns,
            pattern_details=details,
        )

    except Exception as e:
        print(f"[aurion.q_risk] unexpected error: {e}")
        return QRiskResult(
            score=0,
            level="ok",
            patterns=[],
            pattern_details=[],
        )
