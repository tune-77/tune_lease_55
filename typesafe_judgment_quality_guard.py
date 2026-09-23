"""Optional TypeSafe/Jev judgments for judgment-asset candidate quality.

``scripts/build_autoresearch_judgment_asset_candidates.py`` decides whether an
extracted claim is a reusable judgment asset ("actionable") or a textbook
generality, and that verdict alone decides whether the claim ever reaches a
human reviewer via ``_promotion_status()``. Today the verdict comes from
keyword-marker presence only, so a claim phrased without a listed marker is
dropped silently.

Design contract (mirrors ``typesafe_dedup_guard``):

* The deterministic ``_judgment_asset_quality()`` markers stay in front and
  remain the verdict whenever this module is disabled or fails.
* Every claim in a run goes out in ONE batched request; the judgments are
  mutually independent Choice questions over a shared class catalog.
* Only locally privacy-screened claims may be sent.
* Failure is fail-open: callers keep the rule verdict, reproducing the prior
  behavior exactly.
* A judgment that is not confidently separated is routed to ``review`` rather
  than silently suppressing the candidate. Suppression is the expensive error
  here; an extra review row is the cheap one.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_SECONDS = 12.0

# Jev accepts at most 64 items in one batched classification.
MAX_CLAIMS_PER_REQUEST = 64
DEFAULT_MAX_CLAIMS = 64

# Auto requires BOTH a high top probability and a clear winner-to-runner-up
# margin. Anything short of both is a review row, never a silent drop.
AUTO_ACCEPT_MIN = 0.85
MINIMUM_MARGIN = 0.50

MAX_CLAIM_CHARS = 400

# The class catalog carries the decision. Each description states the definition,
# what is excluded, its precedence against the overlapping class, and an example.
# An empty description here makes build_quality_request() raise, so the guard can
# never send a half-written catalog.
#
# Boundary tilt, chosen deliberately: an unresolvable claim goes to needs_context,
# never to textbook_general. Measurement on the 2026-09-13 corpus showed the rule
# filter suppressing 76.5% of candidates with no review trace, so dropping a
# usable claim is the expensive error and an extra review row is the cheap one.
JUDGMENT_QUALITY_CLASSES: dict[str, str] = {
    "actionable": (
        "個別案件の審査で、担当者の確認行動・承認条件・反証の取り方・否認理由の"
        "いずれかを実際に変えられる主張。対象（業種・物件・資金使途・時期のいずれか）が"
        "絞られているか、確認すべき具体的な事実・数値・書類が名指しされている。"
        "確認質問そのものの形（「〜はどの程度ですか？」「〜の実績データを提供できますか？」）"
        "でよく、「確認」「照合」という語を含んでいる必要はない。"
        "案件担当者がその文をそのまま相手に問える、あるいはそのまま社内で実行できるなら"
        "actionable とする。判断を変える条件が書かれていれば、結論自体が平凡でもよい。"
        "例:「補助金の交付決定前に検収が前倒しされていないか見る」"
        "「貴社の設備稼働率は業界平均と比較してどの程度か」"
    ),
    "textbook_general": (
        "どの案件に当てても同じことしか言えず、担当者の行動が何も変わらない一般論。"
        "教科書や入門記事の要約、原則論、用語の言い換えがここに入る。"
        "「総合的に判断する」「リスクを適切に評価する」「慎重に検討する」のように、"
        "何を見るか・どうなったら判断を変えるかが書かれていないもの。"
        "主張が正しいかどうかは関係ない。正しくても行動が変わらなければこのクラス。"
        "actionable と迷う場合にこのクラスを選んではならない（その場合は needs_context）。"
        "確信を持って「これは誰でも知っている、案件では何も変わらない」と言える時だけ選ぶ。"
        "例:「与信判断は財務内容と事業性を総合的に勘案する」"
    ),
    "needs_context": (
        "案件で使える可能性はあるが、この文だけでは actionable か一般論か決められない主張。"
        "対象・閾値・時期・主語のいずれかが欠けていて元のノートを見ないと意味が確定しない場合、"
        "および候補の種別（candidate_type）と中身が噛み合っておらず判定に迷う場合。"
        "actionable と textbook_general のどちらとも断定できないときは必ずこのクラスを選ぶ。"
        "このクラスは自動昇格せず人間のレビューに回るため、判断を保留する正しい置き場所である。"
        "例:「一定の水準を下回る場合は保全を見直す」（水準が書かれておらず、"
        "案件でそのまま使えるか判断できない）"
    ),
}

# The verdict vocabulary the caller already understands. ``needs_context`` is
# new and must never map onto the suppressing status by itself.
# One line per class, sent with every question so the option set is never bare.
# The full definitions travel once in `state.judgment_quality_classes`; these
# glosses are what the model falls back on if it does not resolve the pointer,
# so each one must stand alone as a weaker-but-correct rule.
CLASS_GLOSS: dict[str, str] = {
    "actionable": (
        "個別案件の確認行動・承認条件・反証・否認理由のいずれかを実際に変えられる主張。"
        "質問形のままでよい。全文定義は `judgment_quality_classes.actionable`。"
    ),
    "textbook_general": (
        "どの案件でも同じことしか言えず、担当者の行動が変わらない一般論。"
        "actionable と迷うなら選ばない。全文定義は `judgment_quality_classes.textbook_general`。"
    ),
    "needs_context": (
        "対象・閾値・時期・主語のいずれかが欠け、この文だけでは判定できない。"
        "断定できない時は必ずこれ。全文定義は `judgment_quality_classes.needs_context`。"
    ),
}

RULE_COMPATIBLE_CLASSES = ("actionable", "textbook_general")

_SENSITIVE_MARKERS = (
    "案件番号",
    "顧客番号",
    "申込人",
    "保証人",
    "代表者名",
    "担当者名",
)
_COMPANY_RE = re.compile(r"(?:株式会社|有限会社|合同会社)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:^|\D)0\d{1,4}-\d{1,4}-\d{3,4}(?:\D|$)")
_CASE_ID_RE = re.compile(r"(?:案件|顧客|申込|契約)[#＃:：\s-]*[A-Za-z0-9-]{4,}")

RequestFn = Callable[[dict[str, Any]], Mapping[str, Any]]


class TypeSafeJudgmentQualityError(RuntimeError):
    """Raised when a TypeSafe response cannot safely drive a quality verdict."""


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_api_key(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the credential without placing Keychain data in logs."""
    env = os.environ if environ is None else environ
    explicit = str(env.get("TYPESAFE_API_KEY") or "").strip()
    if explicit:
        return explicit

    service = str(env.get("TYPESAFE_API_KEYCHAIN_SERVICE") or "").strip()
    if not service or sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def typesafe_judgment_quality_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only when quality judgments are explicitly enabled.

    Deliberately separate from ``TYPESAFE_DEDUP_ENABLED`` so either surface can
    be switched off without disturbing the other.
    """
    env = os.environ if environ is None else environ
    return _env_truthy(env.get("TYPESAFE_JUDGMENT_QUALITY_ENABLED")) and bool(
        _resolve_api_key(env)
    )


def _max_claims(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        configured = int(
            env.get("TYPESAFE_JUDGMENT_QUALITY_MAX_CLAIMS", DEFAULT_MAX_CLAIMS)
        )
    except (TypeError, ValueError):
        return DEFAULT_MAX_CLAIMS
    return min(MAX_CLAIMS_PER_REQUEST, max(1, configured))


def is_safe_public_claim(claim: Mapping[str, Any]) -> bool:
    """Conservatively reject claim text that may carry case or PII data.

    Research-note claims are normally about public policy and market facts, but
    a note can quote a real case. Money figures are deliberately NOT rejected:
    published thresholds such as 補助金上限 are the substance of these claims.
    """
    text = str(claim.get("claim") or "")
    if any(marker in text for marker in _SENSITIVE_MARKERS):
        return False
    return not any(
        pattern.search(text)
        for pattern in (_COMPANY_RE, _EMAIL_RE, _PHONE_RE, _CASE_ID_RE)
    )


def filter_safe_claims(
    claims: Sequence[Mapping[str, Any]],
) -> tuple[list[int], int]:
    """Return the indices that pass the local privacy screen, and the skip count."""
    safe = [index for index, item in enumerate(claims) if is_safe_public_claim(item)]
    return safe, len(claims) - len(safe)


def _public_claim(claim: Mapping[str, Any]) -> dict[str, str]:
    return {
        "candidate_type": str(claim.get("candidate_type") or ""),
        "claim": str(claim.get("claim") or "")[:MAX_CLAIM_CHARS],
    }


def _claim_question(index: int, criteria: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    return {
        f"claim{index}_quality": {
            "type": "choice",
            "instructions": (
                f"`claims[{index}].claim` は、リース審査の個別案件で再利用できる判断資産か、"
                f"それとも教科書的な一般論かを判定する。"
                f" 候補の種別は `claims[{index}].candidate_type` を参照する"
                f"（confirmation_question=確認行動 / condition_signal=条件・兆候 /"
                f" caution=判断を誤らせない戒め）。"
                f" 種別に対して内容が噛み合っていない場合は actionable としない。"
                f" 各クラスの全文定義は `judgment_quality_classes.<クラス名>` を参照する。"
            ),
            "criteria": dict(criteria),
        }
    }


def build_quality_request(
    claims: Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    inline_criteria: bool = False,
) -> dict[str, Any]:
    """Build one batched System One request covering every claim.

    The class catalog is shared state, not per-question text: it travels once in
    ``state.judgment_quality_classes`` and every question carries only the short
    gloss plus a pointer. Repeating the full catalog once per claim made the
    payload grow with the batch (~1.3KB x N) for text that is identical every
    time. ``inline_criteria=True`` restores the repeated form so Phase 1 can A/B
    the two payload shapes without a code change.
    """
    missing = [
        name
        for name, text in JUDGMENT_QUALITY_CLASSES.items()
        if not str(text).strip()
    ]
    if missing:
        raise TypeSafeJudgmentQualityError(
            "JUDGMENT_QUALITY_CLASSES is missing descriptions for: " + ", ".join(missing)
        )
    if set(CLASS_GLOSS) != set(JUDGMENT_QUALITY_CLASSES):
        raise TypeSafeJudgmentQualityError(
            "CLASS_GLOSS and JUDGMENT_QUALITY_CLASSES must cover the same classes"
        )
    criteria: Mapping[str, str] = (
        JUDGMENT_QUALITY_CLASSES if inline_criteria else CLASS_GLOSS
    )
    questions: dict[str, dict[str, Any]] = {}
    state_claims: list[dict[str, str]] = []
    for position, claim in enumerate(claims):
        state_claims.append(_public_claim(claim))
        questions.update(_claim_question(position, criteria))
    state: dict[str, Any] = {"claims": state_claims}
    if not inline_criteria:
        state["judgment_quality_classes"] = dict(JUDGMENT_QUALITY_CLASSES)
    return {
        "state": state,
        "model": model,
        "questions": questions,
    }


def _default_request(payload: dict[str, Any]) -> Mapping[str, Any]:
    api_key = _resolve_api_key()
    if not api_key:
        raise TypeSafeJudgmentQualityError("TYPESAFE_API_KEY is not configured")
    timeout = float(
        os.environ.get(
            "TYPESAFE_JUDGMENT_QUALITY_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS
        )
    )
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            os.environ.get("TYPESAFE_ENDPOINT", TYPESAFE_ENDPOINT),
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, Mapping):
        raise TypeSafeJudgmentQualityError("TypeSafe response must be an object")
    return body


def _margin(raw: Mapping[str, Any], choice: str) -> float | None:
    """Return the winner-to-runner-up gap, or None when no distribution is given.

    The Choice answer is only contractually required to carry ``choice`` and
    ``confidence``; ``probabilities`` is read opportunistically so the margin
    rule tightens the gate when available and never loosens it when absent.
    """
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, Mapping) or len(probabilities) < 2:
        return None
    values: list[float] = []
    for key, value in probabilities.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        if str(key) != choice:
            values.append(number)
    if not values:
        return None
    try:
        top = float(probabilities[choice])
    except (KeyError, TypeError, ValueError):
        return None
    return top - max(values)


def _choice(answers: Mapping[str, Any], question_id: str) -> tuple[str, float, float | None]:
    """Return ``(choice, confidence, margin)`` for one Choice answer."""
    raw = answers.get(question_id)
    if not isinstance(raw, Mapping) or raw.get("type") != "choice":
        raise TypeSafeJudgmentQualityError(f"missing choice answer: {question_id}")
    choice = str(raw.get("choice") or "")
    if choice not in JUDGMENT_QUALITY_CLASSES:
        raise TypeSafeJudgmentQualityError(f"unknown class returned: {question_id}")
    try:
        confidence = float(raw["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeSafeJudgmentQualityError(
            f"invalid choice confidence: {question_id}"
        ) from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise TypeSafeJudgmentQualityError(f"confidence outside [0, 1]: {question_id}")
    return choice, confidence, _margin(raw, choice)


def route_quality(
    choice: str,
    confidence: float,
    margin: float | None,
    *,
    auto_accept: float = AUTO_ACCEPT_MIN,
    minimum_margin: float = MINIMUM_MARGIN,
) -> str:
    """Apply the reviewable routing policy to one raw Jev judgment.

    Kept separate from ``judge_claims`` so the policy can be changed and
    re-measured without re-running inference.
    """
    if confidence < auto_accept:
        return "review"
    if margin is not None and margin < minimum_margin:
        return "review"
    if choice not in RULE_COMPATIBLE_CLASSES:
        return "review"
    return "auto"


def judge_claims(
    claims: Sequence[Mapping[str, Any]],
    *,
    request_fn: RequestFn | None = None,
    model: str | None = None,
    auto_accept: float = AUTO_ACCEPT_MIN,
    minimum_margin: float = MINIMUM_MARGIN,
    max_claims: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Judge every privacy-screened claim with one TypeSafe request.

    Returned judgments carry the original index in ``claims`` so the caller can
    leave unscreened and truncated rows on their rule verdict.
    """
    safe_indices, skipped = filter_safe_claims(claims)
    limit = max_claims or _max_claims()
    selected = safe_indices[:limit]
    if not selected:
        return [], {
            "status": "skipped",
            "reason": "no_safe_claims",
            "privacy_skipped": skipped,
        }

    selected_model = model or os.environ.get("TYPESAFE_MODEL", DEFAULT_MODEL)
    payload = build_quality_request(
        [claims[index] for index in selected], model=selected_model
    )
    body = (request_fn or _default_request)(payload)
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeSafeJudgmentQualityError("TypeSafe response is missing answers")

    judged: list[dict[str, Any]] = []
    for position, index in enumerate(selected):
        choice, confidence, margin = _choice(answers, f"claim{position}_quality")
        judged.append(
            {
                "index": index,
                "asset_quality": choice,
                "confidence": confidence,
                "margin": margin,
                "decision": route_quality(
                    choice,
                    confidence,
                    margin,
                    auto_accept=auto_accept,
                    minimum_margin=minimum_margin,
                ),
            }
        )

    auto = sum(1 for item in judged if item["decision"] == "auto")
    return judged, {
        "status": "applied",
        "model": str(body.get("model") or selected_model),
        "claim_count": len(selected),
        "privacy_skipped": skipped,
        "truncated_claims": max(0, len(safe_indices) - len(selected)),
        "auto_count": auto,
        "review_count": len(judged) - auto,
        "auto_accept": auto_accept,
        "minimum_margin": minimum_margin,
        "usage": dict(body.get("usage") or {}),
    }


def judge_claims_if_enabled(
    claims: Sequence[Mapping[str, Any]],
    *,
    request_fn: RequestFn | None = None,
    auto_accept: float = AUTO_ACCEPT_MIN,
    minimum_margin: float = MINIMUM_MARGIN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply Jev only when explicitly enabled; fail open to the rule verdict.

    On any failure no judgments are returned, so every candidate keeps the
    ``_judgment_asset_quality()`` verdict it already has.
    """
    if request_fn is None and not typesafe_judgment_quality_enabled():
        return [], {"status": "disabled"}
    try:
        return judge_claims(
            claims,
            request_fn=request_fn,
            auto_accept=auto_accept,
            minimum_margin=minimum_margin,
        )
    except Exception as exc:
        return [], {"status": "fallback", "error_type": type(exc).__name__}


__all__ = [
    "AUTO_ACCEPT_MIN",
    "JUDGMENT_QUALITY_CLASSES",
    "MINIMUM_MARGIN",
    "RULE_COMPATIBLE_CLASSES",
    "TypeSafeJudgmentQualityError",
    "build_quality_request",
    "filter_safe_claims",
    "is_safe_public_claim",
    "judge_claims",
    "judge_claims_if_enabled",
    "route_quality",
    "typesafe_judgment_quality_enabled",
]
