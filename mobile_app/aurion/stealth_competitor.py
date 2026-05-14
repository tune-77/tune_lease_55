"""
aurion/stealth_competitor.py — ステルス競合推定モジュール (P3-001)

spread_pred（RF 回帰モデルが出力した推奨スプレッド）と申告競合情報の乖離を検知し、
ステルス競合圧力スコア（0〜100）と圧力パターンリストを返す独立モジュール。
RF/LGBM スコアリングには影響しない参考値（サイドカー）として動作する。
例外を外部に伝播させない設計。
"""
from typing import TypedDict


class PatternDetail(TypedDict):
    code: str
    severity: str
    message: str
    values: dict


class StealthCompetitorResult(TypedDict):
    score: int
    level: str
    patterns: list
    pattern_details: list


def detect_stealth_competitor(
    spread_pred: float,
    base_rate: float,
    competitor: int,
    competitor_rate: float = 0.0,
    grade: int = 5,
    acquisition_cost: float = 0.0,
    nenshu: float = 0.0,
) -> StealthCompetitorResult:
    """
    spread_pred と申告競合情報の乖離からステルス競合圧力を検知する。

    スコア計算（RF/LGBM）には影響しない参考値（サイドカー）。
    例外を外部に伝播させない設計。

    Returns:
        StealthCompetitorResult: score, level, patterns, pattern_details を持つ dict
    """
    try:
        # grade クリップ（範囲外入力ガード）
        grade = max(1, min(10, grade))

        details: list[PatternDetail] = []

        # BR-301: 競合未申告・低スプレッド検知
        if competitor == 0 and spread_pred < 1.5:
            details.append(PatternDetail(
                code="COMP-STEALTH-001",
                severity="high",
                message="競合未申告にもかかわらずスプレッドが1.5%未満です。営業担当が競合の存在を未申告のまま競争入札相当の条件を提示している可能性があります。",
                values={"competitor": competitor, "spread_pred": spread_pred},
            ))

        # BR-302: 相場外れの申告競合金利検知
        if competitor == 1 and competitor_rate > 0 and competitor_rate < base_rate + 0.3:
            details.append(PatternDetail(
                code="COMP-STEALTH-002",
                severity="high",
                message="申告競合金利が基準金利+0.3%未満です。実在しない・あるいは条件を誇張した競合申告の可能性があります。",
                values={"competitor_rate": competitor_rate, "base_rate": base_rate, "threshold": base_rate + 0.3},
            ))

        # BR-303: 格付け対比スプレッド圧縮検知
        comp_003_triggered = False
        if grade <= 3 and spread_pred < 1.0:
            comp_003_triggered = True
        elif 4 <= grade <= 6 and spread_pred < 0.8:
            comp_003_triggered = True
        elif grade >= 7 and spread_pred < 0.5:
            comp_003_triggered = True

        if comp_003_triggered:
            details.append(PatternDetail(
                code="COMP-STEALTH-003",
                severity="medium",
                message="格付け対比でスプレッドが過度に圧縮されています。未申告の競合圧力が価格決定に影響していると推定されます。",
                values={"grade": grade, "spread_pred": spread_pred},
            ))

        # BR-304: 申告競合金利・予測スプレッド乖離検知
        if competitor == 1 and competitor_rate > 0:
            comp_spread = competitor_rate - base_rate
            if abs(spread_pred - comp_spread) > 1.5:
                details.append(PatternDetail(
                    code="COMP-STEALTH-004",
                    severity="medium",
                    message="申告競合スプレッドと予測スプレッドの乖離が1.5%超です。別の未申告競合が存在するか申告内容が実態と異なる可能性があります。",
                    values={
                        "spread_pred": spread_pred,
                        "competitor_rate": competitor_rate,
                        "base_rate": base_rate,
                        "comp_spread": round(comp_spread, 4),
                        "diff": round(abs(spread_pred - comp_spread), 4),
                    },
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
            print(f"[aurion.stealth_competitor] patterns={patterns} score={score}")

        return StealthCompetitorResult(
            score=score,
            level=level,
            patterns=patterns,
            pattern_details=details,
        )

    except Exception as e:
        print(f"[aurion.stealth_competitor] unexpected error: {e}")
        return StealthCompetitorResult(
            score=0,
            level="ok",
            patterns=[],
            pattern_details=[],
        )
