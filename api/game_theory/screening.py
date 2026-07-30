"""
審査ゲーム理論モジュール（REV-224 / Game Theory #1）

申込者が「情報を操作するゲーム」を理論化する。

理論的基盤:
  - シグナリングゲーム（Spence 1973）: 申告値は情報の非対称性下でのシグナル
  - メカニズムデザイン: 「正直申告が支配戦略になる」スコアリング設計の検証
  - 顕示選好: 申告値のパターンから真の財務状態を推定

出力:
  - manipulation_suspicion_score: 0.0〜1.0（高いほど操作の疑いが強い）
  - dominant_strategy_analysis: 正直申告 vs 操作の期待利得比較
  - strategy_flags: どのシグナルが疑わしいか
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_PATHS = [
    _REPO_ROOT / "static_data" / "industry_benchmarks.json",
    _REPO_ROOT / "data" / "industry_benchmarks.json",
]


def _load_benchmarks() -> dict[str, Any]:
    for p in _BENCHMARK_PATHS:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v not in (None, "", "nan") else default
    except (TypeError, ValueError):
        return default


# ── 業種ベンチマーク ──────────────────────────────────────────────────────────

def _get_benchmark(industry_major: str) -> dict[str, Any]:
    benchmarks = _load_benchmarks()
    return benchmarks.get(str(industry_major).strip(), {})


# ── シグナル検出関数 ──────────────────────────────────────────────────────────

def _z_score_from_benchmark(value: float, bench: float, sigma_estimate: float) -> float:
    """
    業種ベンチマークからの z スコアを返す。
    sigma_estimate は業種標準偏差の推定値（ベンチ値の30%を目安とする）。
    """
    if sigma_estimate <= 0:
        return 0.0
    return (value - bench) / sigma_estimate


def _detect_round_number_bias(value: float, label: str) -> dict[str, Any] | None:
    """
    申告値が「綺麗な丸め数字」かどうかを検出する。
    例: 10000, 5000, 20000 は丸め疑い。
    実際の財務数値は端数があるはずで、ちょうど良い数字は操作を示唆する。
    """
    if value <= 0:
        return None
    # 1000万単位や500万単位で割り切れる場合は疑い
    for unit in [10000, 5000, 1000]:
        if abs(value % unit) < 1:
            return {
                "signal": "round_number_bias",
                "label": label,
                "value": value,
                "unit": unit,
                "suspicion": min(0.3, unit / 10000 * 0.3),
                "message": f"{label}が{unit:,}千円単位でぴったり。実財務は端数が出るはず。",
            }
    return None


def _detect_impossible_ratios(
    op_margin: float,
    equity_ratio: float,
    bench_op_margin: float,
    bench_equity_ratio: float,
    industry: str,
) -> list[dict[str, Any]]:
    """
    財務比率の非合理的な値を検出する。
    業種ベンチの3σ以上離れた値は操作または誤記を示唆。
    """
    flags = []

    # 営業利益率: 業種平均の3倍超は異常
    if bench_op_margin > 0 and op_margin > bench_op_margin * 3 and op_margin > 20:
        flags.append({
            "signal": "op_margin_outlier",
            "value": op_margin,
            "benchmark": bench_op_margin,
            "ratio": op_margin / bench_op_margin,
            "suspicion": 0.4,
            "message": f"営業利益率 {op_margin:.1f}% は業種平均 {bench_op_margin:.1f}% の {op_margin/bench_op_margin:.1f}倍。過大申告の可能性。",
        })

    # 自己資本比率: 業種平均の2倍超かつ80%超は珍しい
    if bench_equity_ratio > 0 and equity_ratio > bench_equity_ratio * 2 and equity_ratio > 80:
        flags.append({
            "signal": "equity_ratio_outlier",
            "value": equity_ratio,
            "benchmark": bench_equity_ratio,
            "suspicion": 0.3,
            "message": f"自己資本比率 {equity_ratio:.1f}% が業種平均 {bench_equity_ratio:.1f}% の2倍超。",
        })

    # 営業利益率が負で自己資本比率が高い矛盾
    if op_margin < -5 and equity_ratio > 60:
        flags.append({
            "signal": "margin_equity_inconsistency",
            "value": {"op_margin": op_margin, "equity_ratio": equity_ratio},
            "suspicion": 0.35,
            "message": "赤字が続いているのに自己資本比率が高い。増資や評価損計上の可能性。",
        })

    return flags


def _detect_internal_inconsistency(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """
    提出数値の内部整合性チェック。
    例: 純利益 > 営業利益（特別利益なしなら不自然）
    """
    flags = []
    op_profit = _safe_float(inputs.get("op_profit"))
    net_income = _safe_float(inputs.get("net_income"))
    nenshu = _safe_float(inputs.get("nenshu"))
    net_assets = _safe_float(inputs.get("net_assets"))
    total_assets = _safe_float(inputs.get("total_assets"))

    # 純利益 > 営業利益（非常に大きな差）
    if op_profit > 0 and net_income > op_profit * 1.5 and (net_income - op_profit) > 1000:
        flags.append({
            "signal": "net_vs_op_inconsistency",
            "suspicion": 0.25,
            "message": f"純利益({net_income:,.0f}千円)が営業利益({op_profit:,.0f}千円)を大幅に上回る。特別利益の内容要確認。",
        })

    # 総資産 < 純資産（定義上不可）
    if total_assets > 0 and net_assets > total_assets:
        flags.append({
            "signal": "impossible_balance_sheet",
            "suspicion": 0.8,
            "message": "純資産が総資産を上回っている。貸借対照表の誤記または操作。",
        })

    # 売上高 0 で利益が正（無売上で利益は通常ない）
    if nenshu <= 0 and op_profit > 0:
        flags.append({
            "signal": "profit_without_revenue",
            "suspicion": 0.5,
            "message": "売上高がゼロなのに営業利益が正。入力誤りまたは意図的な操作。",
        })

    return flags


# ── 支配戦略分析 ──────────────────────────────────────────────────────────────

def _calc_dominant_strategy_payoff(
    suspicion_score: float,
    submitted_score: float,
) -> dict[str, Any]:
    """
    「正直申告」vs「操作申告」の期待利得を比較する。

    ゲーム理論的フレームワーク:
      プレイヤー: 申込者
      戦略: {正直申告, 操作申告}
      利得: 期待承認価値（承認確率 × リースで得られる価値）- 操作コスト

    操作がバレた場合:
      - 即時否決（承認確率 = 0）
      - ブラックリスト効果（将来の申込でも不利）
    """
    # 操作申告の期待承認確率（操作がバレる確率で減衰）
    detection_prob = suspicion_score  # 疑いスコアを検出確率の代理指標とする
    manipulated_score = min(100, submitted_score + 15)  # 操作で+15点の過大申告と仮定
    honest_approval_prob = _logistic(submitted_score, midpoint=65)
    manipulated_approval_prob = _logistic(manipulated_score, midpoint=65) * (1 - detection_prob)

    # 期待利得（正規化: 承認=1.0, 否決=0.0 as baseline）
    honest_payoff = honest_approval_prob * 1.0
    manipulated_payoff = (
        manipulated_approval_prob * 1.0        # バレなかった場合の利得
        - detection_prob * 0.5                  # バレた場合のペナルティ（将来も含む）
    )

    is_honest_dominant = honest_payoff >= manipulated_payoff

    return {
        "honest_approval_prob": round(honest_approval_prob, 3),
        "manipulated_approval_prob_if_undetected": round(_logistic(manipulated_score, midpoint=65), 3),
        "detection_probability": round(detection_prob, 3),
        "honest_expected_payoff": round(honest_payoff, 3),
        "manipulated_expected_payoff": round(manipulated_payoff, 3),
        "is_honest_dominant_strategy": is_honest_dominant,
        "nash_equilibrium": "正直申告" if is_honest_dominant else "操作申告（検出リスク高）",
        "recommendation": (
            "正直申告が支配戦略です。操作の検出リスクを考慮すると正直申告の期待値が高い。"
            if is_honest_dominant else
            "操作が有利に見えますが検出リスクが低い状態です。審査精度の向上が必要。"
        ),
    }


def _logistic(score: float, midpoint: float = 65, steepness: float = 0.1) -> float:
    """スコアを承認確率に変換するロジスティック関数。"""
    return 1 / (1 + math.exp(-steepness * (score - midpoint)))


# ── メイン分析関数 ────────────────────────────────────────────────────────────

def analyze_screening_game(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    審査入力データを受け取り、ゲーム理論的分析を返す。

    inputs キー (scoring_core と同様):
      industry_major, nenshu, op_profit, net_income,
      net_assets, total_assets, user_equity_ratio (オプション)

    Returns:
      manipulation_suspicion_score, strategy_flags, dominant_strategy_analysis, summary
    """
    industry = str(inputs.get("industry_major") or "")
    bench = _get_benchmark(industry)

    nenshu = _safe_float(inputs.get("nenshu"))
    op_profit = _safe_float(inputs.get("op_profit"))
    net_income = _safe_float(inputs.get("net_income"))
    net_assets = _safe_float(inputs.get("net_assets"))
    total_assets = _safe_float(inputs.get("total_assets"))

    op_margin = (op_profit / nenshu * 100) if nenshu > 0 else 0.0
    equity_ratio = (
        _safe_float(inputs.get("user_equity_ratio"))
        if inputs.get("user_equity_ratio") not in (None, "")
        else ((net_assets / total_assets * 100) if total_assets > 0 else 0.0)
    )
    bench_op_margin = _safe_float(bench.get("op_margin"), default=5.0)
    bench_equity_ratio = _safe_float(bench.get("equity_ratio"), default=40.0)

    # ── シグナル収集 ──────────────────────────────────────────────────────────
    strategy_flags: list[dict] = []

    # 丸め数字バイアス
    for label, val in [
        ("売上高", nenshu), ("営業利益", op_profit),
        ("純利益", net_income), ("純資産", net_assets),
    ]:
        flag = _detect_round_number_bias(val, label)
        if flag:
            strategy_flags.append(flag)

    # 比率の外れ値
    strategy_flags.extend(
        _detect_impossible_ratios(op_margin, equity_ratio, bench_op_margin, bench_equity_ratio, industry)
    )

    # 内部整合性
    strategy_flags.extend(_detect_internal_inconsistency(inputs))

    # ── 操作疑いスコア集計 ────────────────────────────────────────────────────
    total_suspicion = sum(f.get("suspicion", 0) for f in strategy_flags)
    manipulation_suspicion_score = round(min(1.0, total_suspicion), 3)

    # ── 支配戦略分析 ──────────────────────────────────────────────────────────
    # 仮のスコア（scoring_core を呼ばずに簡易計算）
    proxy_score = _estimate_proxy_score(op_margin, equity_ratio, bench_op_margin, bench_equity_ratio)
    dominant_strategy = _calc_dominant_strategy_payoff(manipulation_suspicion_score, proxy_score)

    # ── サマリー ──────────────────────────────────────────────────────────────
    risk_level = (
        "高" if manipulation_suspicion_score >= 0.6 else
        "中" if manipulation_suspicion_score >= 0.3 else
        "低"
    )

    return {
        "manipulation_suspicion_score": manipulation_suspicion_score,
        "risk_level": risk_level,
        "strategy_flags": strategy_flags,
        "flag_count": len(strategy_flags),
        "dominant_strategy_analysis": dominant_strategy,
        "proxy_score": round(proxy_score, 1),
        "benchmark_used": {
            "industry": industry,
            "op_margin": bench_op_margin,
            "equity_ratio": bench_equity_ratio,
        },
        "summary": (
            f"操作疑いスコア {manipulation_suspicion_score:.2f}（リスク: {risk_level}）。"
            f"{len(strategy_flags)}件のシグナルを検出。"
            f"{'正直申告が支配戦略。' if dominant_strategy['is_honest_dominant_strategy'] else '検出精度の向上が必要。'}"
        ),
    }


def _estimate_proxy_score(
    op_margin: float,
    equity_ratio: float,
    bench_op_margin: float,
    bench_equity_ratio: float,
) -> float:
    """スコアの代理推定（簡易）。"""
    margin_score = min(40, max(0, 20 + (op_margin - bench_op_margin) * 2))
    equity_score = min(40, max(0, 20 + (equity_ratio - bench_equity_ratio) * 0.5))
    base = 40.0
    return min(100, max(0, base + margin_score / 2 + equity_score / 2))
