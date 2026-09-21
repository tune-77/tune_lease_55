"""Optional TypeSafe/Jev estimation for the per-item asset scores.

Phase 0 scope: this module is deliberately **not imported by any existing
module**.  ``scoring_core.py``, ``asset_scorer.py`` and the FastAPI routes are
unchanged.  The only caller is ``experiments/typesafe_asset/measure.py`` until
the agreement rate against human-entered scores has been measured.

Why this slot: ``scoring_core.run_quick_scoring`` falls back to
``asset_score = 50.0`` whenever the operator leaves the field empty
(``used_default_asset_score``).  That default is a neutral placeholder, not a
judgment -- the item may be a highly liquid standard machine or a bespoke rig
with no resale market, and the score is identical either way.  Jev can close
exactly that "string -> number" gap.

Design contract:

* The arithmetic stays in code.  Jev answers one narrow question per dimension
  defined in ``category_config.CATEGORY_SCORE_ITEMS``; the weights and the
  weighted average remain in ``asset_scorer.calc_asset_score``, so the result is
  still auditable and a weight change does not require re-running inference.
* All dimensions for one asset go out in ONE batched request; the judgments are
  mutually independent.
* Only operator-authored item identification (name / detail / category / lease
  term) leaves the machine.  Company names, financial figures and case ids stay
  local -- see ``.claude/rules/security.md``.
* Failure is fail-open: callers receive an empty mapping, and
  ``calc_asset_score`` then fills every item with 50, which reproduces the
  current behavior exactly.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from category_config import CATEGORY_SCORE_ITEMS

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_SECONDS = 12.0

MAX_NAME_CHARS = 120
MAX_DETAIL_CHARS = 300

# A Score answer below this confidence is reported but flagged, so the caller
# can decide to keep the neutral default instead.  Provisional: the measurement
# in ``experiments/typesafe_asset`` decides the final value.
MIN_CONFIDENCE = 0.30

RequestFn = Callable[[dict[str, Any]], Mapping[str, Any]]


class TypeSafeAssetError(RuntimeError):
    """Raised when a TypeSafe response cannot safely drive an asset score."""


# TODO(human): define the shared level ladder used for every dimension.
#
# This list IS the calibration of every item score.  Jev returns a
# probability-weighted position on these levels (0 = first entry), and
# ``_rescale`` maps that position onto the 0-100 range that
# ``asset_scorer.calc_asset_score`` expects.  Each entry must describe a
# concrete situation that stands on its own when read together with the
# dimension's own instructions -- e.g. for "技術陳腐化リスク低さ" the top level
# should read as "陳腐化が遅い" rather than as a bare grade letter.
#
# Constraints: 2-10 entries, ordered low -> high (worst asset first, best last),
# and evenly spaced in meaning, because the rescale is linear.  4 levels roughly
# match the existing S/A/B/C/D grading in ``calc_asset_score``.
LEVEL_LADDER: list[str] = [
    "この観点が明確な弱点で、リース満了時の処分価値をほぼ期待できない。"
    "同種の物件は買い手が付かず、撤去・廃棄費用が発生することもある。",
    "この観点がやや不利。買い手は限られ、処分には時間がかかり相場より値引きが必要。",
    "この観点は標準的。同種の物件は中古市場で通常どおり取引され、相場並みの価格で処分できる。",
    "この観点が明確な強みで需要が安定しており、"
    "リース満了時でも短期間で相場以上の価格で処分できる。",
]

# Per-dimension overrides for items whose ladder cannot be expressed by the
# shared wording.  Keyed by the ``id`` in ``CATEGORY_SCORE_ITEMS``.  Empty until
# the measurement shows which dimensions need their own levels.
DIMENSION_LEVELS: dict[str, list[str]] = {}


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


def typesafe_asset_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only when asset-score estimation is explicitly enabled.

    Deliberately separate from ``TYPESAFE_DEDUP_ENABLED`` and
    ``TYPESAFE_RAG_ENABLED`` so this surface can be switched off on its own.
    """
    env = os.environ if environ is None else environ
    return _env_truthy(env.get("TYPESAFE_ASSET_ENABLED")) and bool(_resolve_api_key(env))


def resolve_items(category: str) -> list[Mapping[str, Any]]:
    """Return the scoring dimensions defined for ``category``."""
    items = CATEGORY_SCORE_ITEMS.get(category)
    if not items:
        raise TypeSafeAssetError(f"unknown asset category: {category!r}")
    return list(items)


def _levels(item: Mapping[str, Any]) -> list[str]:
    levels = DIMENSION_LEVELS.get(str(item.get("id") or "")) or LEVEL_LADDER
    if len(levels) < 2:
        raise TypeSafeAssetError(
            "LEVEL_LADDER must define at least 2 ordered levels "
            f"(dimension: {item.get('id')!r})"
        )
    if len(levels) > 10:
        raise TypeSafeAssetError("a Score question accepts at most 10 levels")
    return list(levels)


def _public_asset(asset: Mapping[str, Any]) -> dict[str, Any]:
    """Build the minimum asset description sent externally.

    Only the item identification leaves the machine.  Company name, applicant
    id, financial figures and any free-form memo stay local.
    """
    payload: dict[str, Any] = {
        "category": str(asset.get("category") or "")[:MAX_NAME_CHARS],
        "name": str(asset.get("asset_name") or asset.get("name") or "")[:MAX_NAME_CHARS],
    }
    detail = str(asset.get("asset_detail") or asset.get("detail") or "").strip()
    if detail:
        payload["detail"] = detail[:MAX_DETAIL_CHARS]
    maker = str(asset.get("maker") or "").strip()
    if maker:
        payload["maker"] = maker[:MAX_NAME_CHARS]
    term = asset.get("lease_term_months") or asset.get("term_months")
    try:
        if term is not None:
            payload["lease_term_months"] = int(term)
    except (TypeError, ValueError):
        pass
    return payload


def _item_question(item: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    item_id = str(item["id"])
    return {
        f"item_{item_id}": {
            "type": "score",
            "instructions": (
                f"リース物件 `asset.name`（補足: `asset.detail`）について、"
                f"「{item['label']}」の度合いを判定する。"
                f" 判定基準: {item.get('help') or ''}"
                f" 物件の種別は `asset.category` を参照する。"
            ),
            "criteria": _levels(item),
        }
    }


def build_asset_request(
    asset: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Build one batched System One request covering every dimension."""
    questions: dict[str, dict[str, Any]] = {}
    for item in items:
        questions.update(_item_question(item))
    return {
        "state": {"asset": _public_asset(asset)},
        "model": model,
        "questions": questions,
    }


def _default_request(payload: dict[str, Any]) -> Mapping[str, Any]:
    api_key = _resolve_api_key()
    if not api_key:
        raise TypeSafeAssetError("TYPESAFE_API_KEY is not configured")
    timeout = float(os.environ.get("TYPESAFE_ASSET_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            os.environ.get("TYPESAFE_ENDPOINT", TYPESAFE_ENDPOINT),
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, Mapping):
        raise TypeSafeAssetError("TypeSafe response must be an object")
    return body


def _score(
    answers: Mapping[str, Any],
    question_id: str,
    level_count: int,
) -> tuple[float, float]:
    """Return ``(raw_score, confidence)`` for one Score answer."""
    raw = answers.get(question_id)
    if not isinstance(raw, Mapping) or raw.get("type") != "score":
        raise TypeSafeAssetError(f"missing score answer: {question_id}")
    try:
        value = float(raw["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeSafeAssetError(f"invalid score answer: {question_id}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= level_count - 1:
        raise TypeSafeAssetError(
            f"score outside [0, {level_count - 1}]: {question_id}"
        )
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return value, confidence


def _rescale(value: float, level_count: int) -> float:
    """Map a Score position onto the 0-100 range ``calc_asset_score`` expects."""
    span = max(1, level_count - 1)
    return round(value / span * 100.0, 1)


def judge_asset(
    asset: Mapping[str, Any],
    *,
    category: str | None = None,
    request_fn: RequestFn | None = None,
    model: str | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Estimate every item score for one asset with a single TypeSafe request.

    Returns ``(scores, meta)`` where ``scores`` is the ``{item_id: 0-100}``
    mapping ``asset_scorer.calc_asset_score`` already accepts.
    """
    resolved_category = str(category or asset.get("category") or "")
    items = resolve_items(resolved_category)

    selected_model = model or os.environ.get("TYPESAFE_MODEL", DEFAULT_MODEL)
    payload = build_asset_request(asset, items, model=selected_model)
    body = (request_fn or _default_request)(payload)
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeSafeAssetError("TypeSafe response is missing answers")

    scores: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    low_confidence: list[str] = []
    for item in items:
        item_id = str(item["id"])
        level_count = len(_levels(item))
        raw, confidence = _score(answers, f"item_{item_id}", level_count)
        scores[item_id] = _rescale(raw, level_count)
        details[item_id] = {
            "raw": raw,
            "levels": level_count,
            "confidence": round(confidence, 3),
        }
        if confidence < min_confidence:
            low_confidence.append(item_id)

    return scores, {
        "status": "applied",
        "model": str(body.get("model") or selected_model),
        "category": resolved_category,
        "item_count": len(items),
        "low_confidence_items": low_confidence,
        "min_confidence": min_confidence,
        "details": details,
        "usage": dict(body.get("usage") or {}),
    }


def judge_asset_if_enabled(
    asset: Mapping[str, Any],
    *,
    category: str | None = None,
    request_fn: RequestFn | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Apply Jev only when explicitly enabled; fail open to current behavior.

    On any failure an empty mapping is returned.  ``calc_asset_score`` then
    fills every item with 50, which is exactly what happens today when the
    operator leaves the asset score blank.
    """
    if request_fn is None and not typesafe_asset_enabled():
        return {}, {"status": "disabled"}
    try:
        return judge_asset(
            asset,
            category=category,
            request_fn=request_fn,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        return {}, {
            "status": "fallback",
            "error_type": type(exc).__name__,
        }


__all__ = [
    "DIMENSION_LEVELS",
    "LEVEL_LADDER",
    "MIN_CONFIDENCE",
    "TypeSafeAssetError",
    "build_asset_request",
    "judge_asset",
    "judge_asset_if_enabled",
    "resolve_items",
    "typesafe_asset_enabled",
]
