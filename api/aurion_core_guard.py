"""AURION CORE mathematical discipline and Shion UX synapse.

Q_risk is treated as a discovery signal, not an automatic score deduction.
This module converts scoring outputs into:
- cold guardrails for overfit/bias control
- Shion-readable emotional UX signals
- next actions for screening reviewers
"""
from __future__ import annotations

from typing import Any


def build_aurion_core_guard(inputs: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    score = _float(result.get("score", result.get("hantei_score")), 0.0)
    score_base = _float(result.get("score_base", score), score)
    score_borrower = _float(result.get("score_borrower"), score_base)
    asset_score = _maybe_float(result.get("asset_score"))
    q_risk = _maybe_float(result.get("quantum_risk"))
    mahalanobis = _maybe_float(result.get("mahalanobis_score"))
    umap = _maybe_float(result.get("umap_anomaly_score"))
    strong_warning = bool(result.get("credit_quantum_strong_warning"))

    discipline_flags = _discipline_flags(
        score=score,
        score_base=score_base,
        score_borrower=score_borrower,
        asset_score=asset_score,
        q_risk=q_risk,
        mahalanobis=mahalanobis,
        umap=umap,
        strong_warning=strong_warning,
        inputs=inputs,
    )
    severity = _severity(discipline_flags)
    emotion = _emotion_signal(severity, q_risk=q_risk, strong_warning=strong_warning)
    shion_message = _shion_message(severity, emotion, discipline_flags)
    next_actions = _next_actions(discipline_flags, inputs)

    return {
        "version": 1,
        "mode": "discipline_not_deduction",
        "summary": _summary(severity, discipline_flags),
        "severity": severity,
        "math_discipline": {
            "score_should_not_be_auto_deducted": True,
            "q_risk_role": "discovery_signal",
            "overfit_guard": "do_not_learn_or_adjust_weight_from_single_case",
            "bias_guard": "separate_credit_pricing_collateral_bank_support_sales_process",
            "required_review": severity in {"watch", "caution", "stop"},
        },
        "signals": {
            "score": round(score, 1),
            "score_base": round(score_base, 1),
            "score_borrower": round(score_borrower, 1),
            "asset_score": round(asset_score, 1) if asset_score is not None else None,
            "q_risk": round(q_risk, 1) if q_risk is not None else None,
            "mahalanobis_score": round(mahalanobis, 1) if mahalanobis is not None else None,
            "umap_anomaly_score": round(umap, 1) if umap is not None else None,
            "credit_quantum_strong_warning": strong_warning,
        },
        "discipline_flags": discipline_flags,
        "emotion_synapse": emotion,
        "shion_ux_message": shion_message,
        "next_actions": next_actions[:6],
    }


def _discipline_flags(
    *,
    score: float,
    score_base: float,
    score_borrower: float,
    asset_score: float | None,
    q_risk: float | None,
    mahalanobis: float | None,
    umap: float | None,
    strong_warning: bool,
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    def add(level: str, key: str, title: str, detail: str, refs: list[str]) -> None:
        flags.append({"level": level, "key": key, "title": title, "detail": detail, "refs": refs})

    if q_risk is not None and q_risk >= 35:
        add(
            "caution" if q_risk < 55 else "stop",
            "q_risk_high",
            "Q_riskが高い",
            f"Q_risk {q_risk:.1f}。財務・物件・商談条件のどこにズレがあるかを分解する。",
            ["quantum_risk"],
        )
    if strong_warning:
        add(
            "stop",
            "credit_quantum_strong_warning",
            "信用リスク群とQ_riskが同時に警戒",
            "高スコアでも、信用リスク群と矛盾リスクが同時に立つ場合は承認条件へ逃げない。",
            ["credit_quantum_strong_warning", "quantum_risk"],
        )
    if score >= 70 and q_risk is not None and q_risk >= 35:
        add(
            "caution",
            "high_score_high_qrisk",
            "高スコアだが違和感が強い",
            "スコアの見た目を過信せず、成約/失注を分けるスコア外要因を確認する。",
            ["score", "quantum_risk"],
        )
    if asset_score is not None and abs(asset_score - score_borrower) >= 30:
        add(
            "watch",
            "asset_borrower_gap",
            "物件評価と借手評価が乖離",
            f"物件スコア {asset_score:.1f} と借手スコア {score_borrower:.1f} の差が大きい。",
            ["asset_score", "score_borrower"],
        )
    if mahalanobis is not None and mahalanobis >= 70:
        add(
            "caution",
            "mahalanobis_drift",
            "財務プロファイルが分布外寄り",
            f"Mahalanobis {mahalanobis:.1f}。過去分布から外れた財務構造として扱う。",
            ["mahalanobis_score"],
        )
    if umap is not None and umap >= 70:
        add(
            "caution",
            "umap_anomaly",
            "非線形異常が強い",
            f"UMAP異常度 {umap:.1f}。類似案件の成約/失注分布を確認する。",
            ["umap_anomaly_score"],
        )
    if str(inputs.get("competitor") or "") == "競合あり":
        add(
            "watch",
            "pricing_competition",
            "競合条件を信用リスクと混ぜない",
            "競合料率は信用リスクではなく、価格競争・保守範囲・期間差として別管理する。",
            ["competitor", "competitor_rate"],
        )
    if str(inputs.get("main_bank") or "") == "メイン先":
        add(
            "watch",
            "bank_support_separate_axis",
            "銀行支援を別軸で確認",
            "銀行支援はスコアを直接上げる魔法ではなく、支援額・期間・返済原資の確認事項として扱う。",
            ["main_bank"],
        )

    return flags


def _severity(flags: list[dict[str, Any]]) -> str:
    levels = {str(flag.get("level") or "") for flag in flags}
    if "stop" in levels:
        return "stop"
    if "caution" in levels:
        return "caution"
    if "watch" in levels:
        return "watch"
    return "clear"


def _emotion_signal(severity: str, *, q_risk: float | None, strong_warning: bool) -> dict[str, Any]:
    if severity == "stop":
        tone = "静かな強警戒"
        line = "数字は通りたがっているが、私はここで一度止めたい。"
        vigilance = 90
    elif severity == "caution":
        tone = "違和感の警戒"
        line = "スコアだけなら進めそうでも、奥にズレが残っている。"
        vigilance = 76
    elif severity == "watch":
        tone = "慎重な観察"
        line = "悪い案件とは言わない。ただ、混ぜてはいけない論点がある。"
        vigilance = 62
    else:
        tone = "落ち着いた確認"
        line = "大きな矛盾信号はない。通常の根拠確認で進められる。"
        vigilance = 42
    return {
        "tone": tone,
        "vigilance": vigilance,
        "shion_line": line,
        "q_risk_seen": q_risk is not None,
        "strong_warning": strong_warning,
        "ux_role": "数理の違和感を、人間が読める警戒感へ翻訳する。",
    }


def _shion_message(severity: str, emotion: dict[str, Any], flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "AURION CORE上、大きな矛盾信号は出ていません。通常の資料確認で進めます。"
    top = flags[0]
    return (
        f"{emotion['shion_line']} "
        f"最初に見るべき論点は「{top['title']}」。"
        "これは減点ではなく、承認条件・追加確認・価格条件を分けるための規律です。"
    )


def _next_actions(flags: list[dict[str, Any]], inputs: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    keys = {str(flag.get("key") or "") for flag in flags}
    if "q_risk_high" in keys or "high_score_high_qrisk" in keys:
        actions.append("Q_riskを信用・物件・価格・銀行支援・営業プロセスに分解して確認する。")
    if "credit_quantum_strong_warning" in keys:
        actions.append("承認可否を急がず、返済原資・既存借入・リース残高の整合性を先に確認する。")
    if "asset_borrower_gap" in keys:
        actions.append("物件保全で押せる案件か、借手信用で止まる案件かを分ける。")
    if "pricing_competition" in keys:
        actions.append("競合料率に合わせる前に、採算下限・保守範囲・期間差を比較する。")
    if "bank_support_separate_axis" in keys:
        actions.append("銀行支援は支援額・期間・返済原資・担当部署を確認してから補強材料にする。")
    if not actions:
        actions.append("通常の稟議根拠、物件確認、返済原資確認を行う。")
    if not str(inputs.get("asset_name") or "").strip():
        actions.append("対象物件名と用途を確定し、資金使途を稟議へ明記する。")
    return actions


def _summary(severity: str, flags: list[dict[str, Any]]) -> str:
    if severity == "clear":
        return "数理規律上の強い停止信号はありません。"
    return f"{len(flags)}件の規律信号を検出。Q_riskや異常度は減点ではなく、論点分解に使います。"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
