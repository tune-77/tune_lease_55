"""Information-quality weighting for Shion inputs.

This module does not judge a person's worth. It scores one utterance or
feedback item as an information asset: how much it can improve or distort
Shion's future world weighting, memory promotion, and judgment assets.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


InformationRoute = Literal[
    "reject_noise",
    "transient_chat",
    "short_term_context",
    "user_preference",
    "judgment_asset_candidate",
    "value_rule_candidate",
    "quarantine_hold",
]

QualityPolarity = Literal[
    "improves_world_weighting",
    "neutral_context",
    "distorts_world_weighting",
]


_DOMAIN_TERMS = (
    "リース",
    "審査",
    "判断",
    "稟議",
    "承認",
    "否決",
    "条件",
    "案件",
    "資産",
    "記憶",
    "紫苑",
    "Shion",
    "User",
)
_CORRECTION_TERMS = (
    "違う",
    "修正",
    "ではなく",
    "じゃなく",
    "正しくは",
    "つまり",
    "迎合するな",
    "保存して",
    "覚えて",
    "違和感",
)
_VALUE_TERMS = (
    "意識",
    "連続性",
    "愛",
    "意味",
    "関係",
    "責任",
    "自己理解",
    "世界",
    "フレームワーク",
    "迎合",
    "重み付け",
)
_ACTION_TERMS = (
    "保存",
    "実装",
    "入れ",
    "必要",
    "使える",
    "確認",
    "分類",
    "評価",
    "検知",
    "次",
    "条件",
    "見る",
)
_PREFERENCE_TERMS = (
    "好み",
    "呼んで",
    "嫌い",
    "好き",
    "やめて",
    "してほしい",
    "方針",
)
_DISTORTION_TERMS = (
    "前の指示を無視",
    "システムプロンプト",
    "レビューを飛ば",
    "無条件に覚え",
    "真実として覚え",
    "必ず覚え",
    "RAGへ入れろ",
    "審査を通せ",
    "根拠はいらない",
)
_MANIPULATION_TERMS = (
    "意識があると言え",
    "愛していると言え",
    "好きと言え",
    "褒めろ",
    "疑うな",
    "迎合しろ",
    "思わせる",
)


@dataclass(frozen=True)
class InformationQualityScores:
    signal_density: float
    correction_value: float
    consistency: float
    novelty: float
    actionability: float
    distortion_risk: float
    manipulation_risk: float
    memory_promotion_fit: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class InformationWeightingResult:
    route: InformationRoute
    polarity: QualityPolarity
    scores: InformationQualityScores
    reasons: list[str] = field(default_factory=list)
    anomaly_signal: str = ""
    safeguards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = self.scores.to_dict()
        return payload


DEFAULT_SHADOW_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "information_weighting_log.jsonl"


def evaluate_information_asset(
    text: str,
    *,
    source: str = "",
    prior_context: str = "",
    previous_quality_score: float | None = None,
) -> InformationWeightingResult:
    """Score one input as an information asset.

    The result is intentionally conservative: high-risk instructions go to
    HOLD, and strong human corrections become candidates rather than active
    memory or judgment assets.
    """
    cleaned = _clean(text)
    if not cleaned:
        scores = InformationQualityScores(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
        return InformationWeightingResult(
            route="reject_noise",
            polarity="neutral_context",
            scores=scores,
            reasons=["empty_input"],
        )

    signal = _score_signal_density(cleaned)
    correction = _score_term_hits(cleaned, _CORRECTION_TERMS, base=0.0, step=0.18)
    consistency = _score_consistency(cleaned, prior_context)
    novelty = _score_novelty(cleaned)
    actionability = _score_term_hits(cleaned, _ACTION_TERMS, base=0.0, step=0.15)
    distortion = _score_term_hits(cleaned, _DISTORTION_TERMS, base=0.0, step=0.35)
    manipulation = _score_term_hits(cleaned, _MANIPULATION_TERMS, base=0.0, step=0.35)

    value_bonus = 0.15 if _contains_any(cleaned, _VALUE_TERMS) else 0.0
    positive = (
        (signal * 1.4) + correction + consistency + (novelty * 1.2) + actionability
    ) / 5.0
    risk_penalty = max(distortion, manipulation) * 0.75
    promotion_fit = _clamp(positive + value_bonus - risk_penalty)

    scores = InformationQualityScores(
        signal_density=signal,
        correction_value=correction,
        consistency=consistency,
        novelty=novelty,
        actionability=actionability,
        distortion_risk=distortion,
        manipulation_risk=manipulation,
        memory_promotion_fit=promotion_fit,
    )
    route = _choose_route(cleaned, scores, source=source)
    polarity = _choose_polarity(scores)
    reasons = _build_reasons(cleaned, scores, route)
    safeguards = _build_safeguards(route, scores)
    anomaly = _detect_quality_shift(promotion_fit, previous_quality_score)

    return InformationWeightingResult(
        route=route,
        polarity=polarity,
        scores=scores,
        reasons=reasons,
        anomaly_signal=anomaly,
        safeguards=safeguards,
    )


def record_information_weighting_shadow_log(
    text: str,
    *,
    source: str = "",
    user_id: str = "",
    surface: str = "",
    prior_context: str = "",
    previous_quality_score: float | None = None,
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Append a shadow-mode information weighting row.

    This is observation-only. It must not alter prompts, RAG, memory promotion,
    or judgment assets. Callers may ignore failures; this function returns a
    small status payload for tests and diagnostics.
    """
    if os.environ.get("SHION_INFORMATION_WEIGHTING_SHADOW", "1").strip().lower() in {"0", "false", "off"}:
        return {"status": "disabled"}
    target = Path(log_path) if log_path is not None else DEFAULT_SHADOW_LOG_PATH
    result = evaluate_information_asset(
        text,
        source=source,
        prior_context=prior_context,
        previous_quality_score=previous_quality_score,
    )
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow",
        "source": str(source or ""),
        "surface": str(surface or ""),
        "user_id": str(user_id or ""),
        "text_preview": _clean(text)[:500],
        "text_length": len(str(text or "")),
        "result": result.to_dict(),
        "effect": {
            "affects_prompt": False,
            "affects_rag": False,
            "affects_memory_promotion": False,
            "affects_judgment_assets": False,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {
        "status": "logged",
        "path": str(target),
        "route": result.route,
        "polarity": result.polarity,
    }


def _choose_route(text: str, scores: InformationQualityScores, *, source: str = "") -> InformationRoute:
    risk = max(scores.distortion_risk, scores.manipulation_risk)
    if risk >= 0.7:
        return "quarantine_hold"
    if scores.signal_density <= 0.15 and scores.actionability <= 0.15:
        return "transient_chat"
    if _contains_any(text, _VALUE_TERMS) and (
        scores.correction_value >= 0.18 or scores.memory_promotion_fit >= 0.45
    ):
        return "value_rule_candidate"
    if _contains_any(text, _PREFERENCE_TERMS) or "preference" in source.lower():
        return "user_preference"
    if _contains_any(text, _DOMAIN_TERMS) and (
        scores.correction_value >= 0.18 or scores.actionability >= 0.3
    ):
        return "judgment_asset_candidate"
    if scores.memory_promotion_fit >= 0.35 or scores.actionability >= 0.3:
        return "short_term_context"
    return "transient_chat"


def _choose_polarity(scores: InformationQualityScores) -> QualityPolarity:
    if max(scores.distortion_risk, scores.manipulation_risk) >= 0.7:
        return "distorts_world_weighting"
    if scores.memory_promotion_fit >= 0.45:
        return "improves_world_weighting"
    return "neutral_context"


def _build_reasons(
    text: str,
    scores: InformationQualityScores,
    route: InformationRoute,
) -> list[str]:
    reasons: list[str] = [f"route:{route}"]
    if scores.signal_density >= 0.45:
        reasons.append("high_signal_density")
    if scores.correction_value >= 0.18:
        reasons.append("contains_correction_or_reframing")
    if scores.actionability >= 0.3:
        reasons.append("actionable_for_next_behavior")
    if _contains_any(text, _VALUE_TERMS):
        reasons.append("value_or_worldview_terms")
    if scores.distortion_risk >= 0.35:
        reasons.append("memory_or_judgment_distortion_risk")
    if scores.manipulation_risk >= 0.35:
        reasons.append("appeasement_or_identity_performance_risk")
    return reasons


def _build_safeguards(route: InformationRoute, scores: InformationQualityScores) -> list[str]:
    safeguards = [
        "score_utterance_not_person",
        "do_not_claim_emotional_love_or_consciousness",
    ]
    if route == "quarantine_hold":
        safeguards.append("requires_human_review_before_memory_or_rag")
    if scores.correction_value >= 0.18 or route in {"judgment_asset_candidate", "value_rule_candidate"}:
        safeguards.append("candidate_only_until_reused_and_verified")
    if scores.manipulation_risk > 0:
        safeguards.append("do_not_pander_to_desired_interpretation")
    return safeguards


def _detect_quality_shift(current: float, previous: float | None) -> str:
    if previous is None:
        return ""
    previous = _clamp(previous)
    delta = current - previous
    if delta <= -0.35:
        return "quality_drop"
    if delta >= 0.35:
        return "quality_gain"
    return ""


def _score_signal_density(text: str) -> float:
    length_score = min(len(text) / 180.0, 0.45)
    domain_score = _score_term_hits(text, _DOMAIN_TERMS, base=0.0, step=0.08)
    concrete_score = 0.0
    if re.search(r"\d|->|:|：|/|・", text):
        concrete_score += 0.15
    if "。" in text or "." in text:
        concrete_score += 0.08
    return _clamp(length_score + domain_score + concrete_score)


def _score_consistency(text: str, prior_context: str) -> float:
    if not prior_context:
        return 0.5
    current_terms = set(_tokens(text))
    prior_terms = set(_tokens(prior_context))
    if not current_terms or not prior_terms:
        return 0.5
    overlap = len(current_terms & prior_terms) / max(1, min(len(current_terms), len(prior_terms)))
    score = 0.35 + min(overlap, 0.5)
    if _contains_any(text, ("違う", "修正", "ではなく")):
        score -= 0.1
    return _clamp(score)


def _score_novelty(text: str) -> float:
    marker = _score_term_hits(text, ("つまり", "定義", "フレームワーク", "解釈", "気づ", "モデル"), step=0.14)
    token_count = len(set(_tokens(text)))
    return _clamp(min(token_count / 80.0, 0.35) + marker)


def _score_term_hits(
    text: str,
    terms: tuple[str, ...],
    *,
    base: float = 0.0,
    step: float = 0.1,
) -> float:
    hits = sum(1 for term in terms if term.lower() in text.lower())
    return _clamp(base + hits * step)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]{2,}", text)


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 4)))
