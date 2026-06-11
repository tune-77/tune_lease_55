"""Shared e-Stat context builder for lease screening.

This module consolidates three separate e-Stat views:
- industry benchmark gap
- lease / capex fit
- macro cycle from machinery orders

It keeps the core borrower score untouched and returns a lightweight
context object for explanation, conditions, and UI rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
MACHINERY_ORDERS_PATH = (
    REPO_ROOT / "data" / "external" / "estat_machinery_orders" / "machinery_orders_analysis.json"
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _score_to_status(score: float) -> str:
    if score >= 70:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


def _load_machinery_orders_analysis() -> dict[str, Any]:
    if not MACHINERY_ORDERS_PATH.exists():
        return {}
    try:
        return json.loads(MACHINERY_ORDERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_industry_snapshot(
    selected_sub: str,
    user_op_margin: float | None,
    user_equity_ratio: float | None,
    user_current_ratio: float | None,
    user_debt_ratio: float | None,
    user_asset_turnover: float | None,
    benchmarks_data: dict[str, Any] | None,
) -> dict[str, Any]:
    bench = (benchmarks_data or {}).get(selected_sub, {})
    op_margin = _safe_float(bench.get("op_margin"))
    equity_ratio = _safe_float(bench.get("equity_ratio"))
    current_ratio = _safe_float(bench.get("current_ratio"))
    debt_ratio = _safe_float(bench.get("debt_ratio"))
    asset_turnover = _safe_float(bench.get("asset_turnover"))
    quick_ratio = _safe_float(bench.get("quick_ratio"))
    updated = str(bench.get("updated") or "")

    op_gap = None if op_margin is None or user_op_margin is None else round(user_op_margin - op_margin, 1)
    eq_gap = None if equity_ratio is None or user_equity_ratio is None else round(user_equity_ratio - equity_ratio, 1)
    curr_gap = None if current_ratio is None or user_current_ratio is None else round(user_current_ratio - current_ratio, 1)
    debt_gap = None if debt_ratio is None or user_debt_ratio is None else round(user_debt_ratio - debt_ratio, 1)
    turn_gap = None if asset_turnover is None or user_asset_turnover is None else round(user_asset_turnover - asset_turnover, 2)

    score = 50.0
    if op_gap is not None:
        score += max(-18.0, min(18.0, op_gap * 4.0))
    if eq_gap is not None:
        score += max(-12.0, min(12.0, eq_gap * 0.5))
    if curr_gap is not None:
        score += max(-8.0, min(8.0, curr_gap * 0.06))
    if debt_gap is not None:
        score -= max(-8.0, min(8.0, debt_gap * 0.04))
    if turn_gap is not None:
        score += max(-6.0, min(6.0, turn_gap * 5.0))
    score = _clamp(score)

    if op_gap is None and eq_gap is None and curr_gap is None and debt_gap is None and turn_gap is None:
        status = "yellow"
    else:
        status = _score_to_status(score)

    summary_parts: list[str] = []
    if op_gap is not None:
        summary_parts.append(f"営業利益率差 {op_gap:+.1f}pt")
    if eq_gap is not None:
        summary_parts.append(f"自己資本差 {eq_gap:+.1f}pt")
    if curr_gap is not None:
        summary_parts.append(f"流動比率差 {curr_gap:+.1f}pt")
    if debt_gap is not None:
        summary_parts.append(f"負債比率差 {debt_gap:+.1f}pt")

    comments: list[str] = []
    if op_margin is not None and user_op_margin is not None:
        comments.append(f"営業利益率 {user_op_margin:.1f}% vs 業種平均 {op_margin:.1f}%")
    if equity_ratio is not None and user_equity_ratio is not None:
        comments.append(f"自己資本比率 {user_equity_ratio:.1f}% vs 業種平均 {equity_ratio:.1f}%")
    if current_ratio is not None and user_current_ratio is not None:
        comments.append(f"流動比率 {user_current_ratio:.0f}% vs 業種平均 {current_ratio:.0f}%")
    if debt_ratio is not None and user_debt_ratio is not None:
        comments.append(f"負債比率 {user_debt_ratio:.1f}% vs 業種平均 {debt_ratio:.1f}%")
    if quick_ratio is not None:
        comments.append(f"当座比率の業種平均 {quick_ratio:.1f}%")

    return {
        "available": bool(bench),
        "label": "同業平均との差",
        "status": status,
        "score": round(score, 1),
        "summary": " / ".join(summary_parts) if summary_parts else "業種平均との差分を算出できませんでした。",
        "comment": bench.get("comment", ""),
        "updated": updated,
        "metrics": {
            "op_margin": op_margin,
            "equity_ratio": equity_ratio,
            "current_ratio": current_ratio,
            "debt_ratio": debt_ratio,
            "asset_turnover": asset_turnover,
            "quick_ratio": quick_ratio,
        },
        "gaps": {
            "op_margin_gap": op_gap,
            "equity_ratio_gap": eq_gap,
            "current_ratio_gap": curr_gap,
            "debt_ratio_gap": debt_gap,
            "asset_turnover_gap": turn_gap,
        },
        "comments": comments[:4],
    }


def _build_lease_snapshot(
    selected_sub: str,
    user_annual_lease_pct: float | None,
    user_lease_credit_pct: float | None,
    benchmarks_data: dict[str, Any] | None,
    capex_lease_data: dict[str, Any] | None,
) -> dict[str, Any]:
    bench = (capex_lease_data or {}).get(selected_sub, {})
    bench_lease_burden = _safe_float(bench.get("lease_burden_rate"))
    bench_capex_to_sales = _safe_float(bench.get("capex_to_sales"))
    bench_lease_to_capex = _safe_float(bench.get("lease_to_capex"))

    compare_value = user_annual_lease_pct if user_annual_lease_pct is not None else user_lease_credit_pct
    compare_label = "年換算推定" if user_annual_lease_pct is not None else "与信/売上比（参考）"
    burden_ratio = None
    burden_score = 50.0

    if bench_lease_burden is not None and bench_lease_burden > 0 and compare_value is not None:
        burden_ratio = round(compare_value / bench_lease_burden, 2)
        if burden_ratio <= 0.5:
            burden_score = 82.0
        elif burden_ratio <= 1.0:
            burden_score = 70.0
        elif burden_ratio <= 1.5:
            burden_score = 55.0
        elif burden_ratio <= 2.0:
            burden_score = 38.0
        else:
            burden_score = 22.0

    normality_bonus = 0.0
    if bench_lease_to_capex is not None:
        if bench_lease_to_capex >= 40:
            normality_bonus += 10.0
        elif bench_lease_to_capex >= 25:
            normality_bonus += 5.0
    if bench_capex_to_sales is not None:
        if bench_capex_to_sales >= 5:
            normality_bonus += 5.0
        elif bench_capex_to_sales >= 2:
            normality_bonus += 3.0

    score = _clamp(burden_score + normality_bonus)
    status = _score_to_status(score) if compare_value is not None or bench else "yellow"

    summary = "業種平均と比較したリース負担率の妥当性を判定できませんでした。"
    if compare_value is not None and bench_lease_burden is not None:
        summary = (
            f"{compare_label} {compare_value:.2f}% / 業種平均 {bench_lease_burden:.2f}% "
            f"→ 比率 {burden_ratio:.2f}倍" if burden_ratio is not None else summary
        )

    comments: list[str] = []
    if bench_lease_burden is not None:
        comments.append(f"業種平均リース料/売上高 {bench_lease_burden:.2f}%")
    if bench_capex_to_sales is not None:
        comments.append(f"業種平均設備投資率 {bench_capex_to_sales:.2f}%")
    if bench_lease_to_capex is not None:
        comments.append(f"業種平均リース/設備投資 {bench_lease_to_capex:.1f}%")

    return {
        "available": bool(bench),
        "label": "リース負担の重さ",
        "status": status,
        "score": round(score, 1),
        "summary": summary,
        "comment": bench.get("comment", ""),
        "updated": str(bench.get("updated") or ""),
        "metrics": {
            "user_annual_lease_pct": user_annual_lease_pct,
            "user_lease_credit_pct": user_lease_credit_pct,
            "bench_lease_burden": bench_lease_burden,
            "bench_capex_to_sales": bench_capex_to_sales,
            "bench_lease_to_capex": bench_lease_to_capex,
            "comparison_label": compare_label,
            "comparison_ratio": burden_ratio,
        },
        "comments": comments[:4],
    }


def _build_macro_snapshot(selected_major: str) -> dict[str, Any]:
    raw = _load_machinery_orders_analysis()
    core = raw.get("core_indicator", {}) if isinstance(raw, dict) else {}
    meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    major_text = str(selected_major or "")
    is_manufacturing = "製造" in major_text or major_text.startswith("E")

    signal = str(core.get("macro_signal") or "中立")
    mom = _safe_float(core.get("latest_mom_pct"))
    yoy = _safe_float(core.get("latest_yoy_pct_original"))
    trend = _safe_float(core.get("three_month_average_change_vs_3m_ago_pct"))
    slope = _safe_float(core.get("monthly_linear_trend_100m_yen"))

    score = {"改善": 74.0, "中立": 56.0, "減速": 36.0}.get(signal, 50.0)
    if mom is not None:
        score += 4.0 if mom > 0 else -4.0 if mom < 0 else 0.0
    if yoy is not None:
        score += 3.0 if yoy > 0 else -3.0 if yoy < 0 else 0.0
    if trend is not None:
        score += 3.0 if trend > 0 else -3.0 if trend < 0 else 0.0
    if slope is not None:
        score += 2.0 if slope > 0 else -2.0 if slope < 0 else 0.0
    if is_manufacturing and signal == "改善":
        score += 2.0
    elif is_manufacturing and signal == "減速":
        score -= 2.0
    score = _clamp(score)

    status = _score_to_status(score)
    summary = (
        f"{'製造寄り' if is_manufacturing else '非製造寄り'}の機械受注マクロ判定は {signal}。前月比 {mom:+.1f}% / 前年同月比 {yoy:+.1f}% / "
        f"3か月平均の3か月前比 {trend:+.1f}%"
        if mom is not None and yoy is not None and trend is not None
        else f"{'製造寄り' if is_manufacturing else '非製造寄り'}の機械受注マクロ判定は {signal}。"
    )

    comments: list[str] = []
    if mom is not None:
        comments.append(f"前月比 {mom:+.1f}%")
    if yoy is not None:
        comments.append(f"前年同月比 {yoy:+.1f}%")
    if trend is not None:
        comments.append(f"3か月平均の3か月前比 {trend:+.1f}%")
    if meta.get("latest_month"):
        comments.append(f"最新月 {meta.get('latest_month')}")

    return {
        "available": bool(raw),
        "label": "景気の向き",
        "status": status,
        "score": round(score, 1),
        "summary": summary,
        "comment": " / ".join(comments) if comments else "",
        "updated": str(meta.get("retrieved_at") or ""),
        "metrics": {
            "macro_signal": signal,
            "latest_mom_pct": mom,
            "latest_yoy_pct_original": yoy,
            "three_month_average_change_vs_3m_ago_pct": trend,
            "monthly_linear_trend_100m_yen": slope,
        },
        "source": {
            "stats_data_id": meta.get("stats_data_id"),
            "latest_month": meta.get("latest_month"),
        },
        "comments": comments[:4],
    }


def build_estat_context(
    *,
    selected_major: str,
    selected_sub: str,
    user_op_margin: float | None,
    user_equity_ratio: float | None,
    user_current_ratio: float | None,
    user_debt_ratio: float | None,
    user_asset_turnover: float | None,
    user_annual_lease_pct: float | None,
    user_lease_credit_pct: float | None,
    benchmarks_data: dict[str, Any] | None,
    capex_lease_data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a small e-Stat context bundle for explanations and UI display."""
    industry = _build_industry_snapshot(
        selected_sub=selected_sub,
        user_op_margin=user_op_margin,
        user_equity_ratio=user_equity_ratio,
        user_current_ratio=user_current_ratio,
        user_debt_ratio=user_debt_ratio,
        user_asset_turnover=user_asset_turnover,
        benchmarks_data=benchmarks_data,
    )
    lease = _build_lease_snapshot(
        selected_sub=selected_sub,
        user_annual_lease_pct=user_annual_lease_pct,
        user_lease_credit_pct=user_lease_credit_pct,
        benchmarks_data=benchmarks_data,
        capex_lease_data=capex_lease_data,
    )
    macro = _build_macro_snapshot(selected_major=selected_major)

    if not any([industry.get("available"), lease.get("available"), macro.get("available")]):
        return None

    overall_score = round(
        industry["score"] * 0.4 + lease["score"] * 0.35 + macro["score"] * 0.25,
        1,
    )
    overall_status = _score_to_status(overall_score)

    recommendations: list[str] = []
    if industry["status"] == "red":
        recommendations.append("業種差が大きい。試算表と資金繰りを先に確認する。")
    elif industry["status"] == "yellow":
        recommendations.append("業種差は小さい。乖離項目だけ残す。")
    if lease["status"] == "red":
        recommendations.append("リース負担が重い。期間短縮か前受を検討する。")
    elif lease["status"] == "yellow":
        recommendations.append("リース負担は概ね許容。推移を確認する。")
    if macro["status"] == "red":
        recommendations.append("景気は逆風。残価と保守条件を厚めに見る。")
    elif macro["status"] == "yellow":
        recommendations.append("景気は中立。個社実績を重ねて確認する。")
    if not recommendations:
        recommendations.append("3層コンテキストは整合的。通常確認で進める。")

    summary = (
        f"同業差={industry['status']} / "
        f"リース負担={lease['status']} / "
        f"景気の向き={macro['status']}"
    )

    source_paths = [
        "static_data/industry_benchmarks.json",
        "data/industry_capex_lease.json",
        "data/external/estat_machinery_orders/machinery_orders_analysis.json",
    ]

    return {
        "available": True,
        "summary": summary,
        "score": overall_score,
        "status": overall_status,
        "score_components": {
            "industry_gap_score": industry["score"],
            "lease_fit_score": lease["score"],
            "macro_cycle_score": macro["score"],
        },
        "dimensions": [industry, lease, macro],
        "recommendations": recommendations[:5],
        "source_paths": source_paths,
    }
