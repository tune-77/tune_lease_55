"""Optional TypeSafe/Jev judgments for improvement-candidate deduplication.

The Phase 1 measurement is documented in ``docs/typesafe_dedup_spec.md`` and
``experiments/typesafe_dedup/README.md``. Phase 2 connects this module to the
daily extractor behind ``TYPESAFE_DEDUP_ENABLED``. The daily pipeline enables
it explicitly; direct callers still make no request unless the flag is set.

Design contract:

* The cheap deterministic layers in ``deduplicate_improvements()`` stay in front.
  Jev is asked only about pairs whose character-bigram Jaccard lands in the
  ambiguous band, where the cheap signal is uninformative.
* All ambiguous pairs go out in ONE batched request; the judgments are mutually
  independent.
* Only locally privacy-screened pairs may be sent.
* Failure is fail-open: callers fall back to "distinct", which reproduces the
  prior behavior exactly.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_PAIRS = 40

# Gray band bounds.  Below GRAY_LOW the titles share almost no bigrams; at or
# above GRAY_HIGH the existing rule already calls them duplicates.  Both values
# are provisional and must be confirmed by the measurement in spec section 8.
GRAY_LOW = 0.30
GRAY_HIGH = 0.55

# Provisional.  The measurement in spec section 8 and the operator policy in the
# spec's TODO(human) decide the final value.
SAME_ISSUE_MIN = 0.60
AUTO_MERGE_MIN = 0.75

MAX_TITLE_CHARS = 200
MAX_REASON_CHARS = 600

_SENSITIVE_MARKERS = (
    "案件番号",
    "顧客番号",
    "申込人",
    "保証人",
    "代表者名",
    "担当者名",
    "氏名",
    "住所",
    "生年月日",
    "自己資本",
    "純資産",
    "年収",
    "所得",
    "借入",
    "現預金",
    "決算書",
)
_COMPANY_RE = re.compile(r"(?:株式会社|有限会社|合同会社|[A-ZＡ-Ｚ][\s　]*社)")
_BUSINESS_NAME_RE = re.compile(
    r"[一-龯]{2,20}(?:商店|工業|建設|運輸|物流|製作所|医院|クリニック|事務所)"
)
_PERSON_NAME_RE = re.compile(r"[一-龯]{2,4}(?:さん|様|氏|は年収|の年収)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:^|\D)0\d{1,4}-\d{1,4}-\d{3,4}(?:\D|$)")
_CASE_ID_RE = re.compile(r"(?:案件|顧客|申込|契約)[#＃:：\s-]*[A-Za-z0-9-]{4,}")
_MONEY_RE = re.compile(r"\d[\d,，.]*\s*(?:円|万円|億円)")

RequestFn = Callable[[dict[str, Any]], Mapping[str, Any]]
SimilarityFn = Callable[[str, str], float]


class TypeSafeDedupError(RuntimeError):
    """Raised when a TypeSafe response cannot safely drive a dedup decision."""


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


def typesafe_dedup_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only when dedup judgments are explicitly enabled.

    Deliberately separate from ``TYPESAFE_RAG_ENABLED`` so either surface can be
    switched off without disturbing the other.
    """
    env = os.environ if environ is None else environ
    return _env_truthy(env.get("TYPESAFE_DEDUP_ENABLED")) and bool(_resolve_api_key(env))


def _max_pairs(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        configured = int(env.get("TYPESAFE_DEDUP_MAX_PAIRS", DEFAULT_MAX_PAIRS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_PAIRS
    return min(200, max(1, configured))


def select_gray_pairs(
    candidates: Sequence[Mapping[str, Any]],
    similarity_fn: SimilarityFn,
    *,
    low: float = GRAY_LOW,
    high: float = GRAY_HIGH,
) -> list[tuple[int, int]]:
    """Return index pairs whose similarity is too ambiguous for the cheap rule.

    ``similarity_fn`` is injected rather than reimplemented here so that
    ``scripts/extract_obsidian_improvements._jaccard_similarity`` stays the single
    source of truth for the bigram metric.

    Phase 1 enumerates every unordered pair, which is the superset needed for
    measurement.  The eventual integration will only ask about pairs of the form
    (incoming candidate, already-kept candidate), matching the existing loop.
    """
    if low > high:
        raise ValueError("low must not exceed high")
    pairs: list[tuple[int, int]] = []
    for i in range(len(candidates)):
        title_i = str(candidates[i].get("title") or "")
        for j in range(i + 1, len(candidates)):
            score = similarity_fn(title_i, str(candidates[j].get("title") or ""))
            if low <= score < high:
                pairs.append((i, j))
    return pairs


def _public_pair(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> dict[str, str]:
    """Build the minimum pair sent externally.

    Only the operator-authored titles leave the machine. Reasons can inherit
    arbitrary Obsidian/chat text, so they remain local together with paths and
    source filenames.
    """
    return {
        "a_title": str(a.get("title") or "")[:MAX_TITLE_CHARS],
        "b_title": str(b.get("title") or "")[:MAX_TITLE_CHARS],
    }


def is_safe_public_candidate(candidate: Mapping[str, Any]) -> bool:
    """Conservatively reject improvement text that may contain case or PII data."""
    text = f"{candidate.get('title') or ''}\n{candidate.get('reason') or ''}"
    if any(marker in text for marker in _SENSITIVE_MARKERS):
        return False
    return not any(
        pattern.search(text)
        for pattern in (
            _COMPANY_RE,
            _BUSINESS_NAME_RE,
            _PERSON_NAME_RE,
            _EMAIL_RE,
            _PHONE_RE,
            _CASE_ID_RE,
            _MONEY_RE,
        )
    )


def filter_safe_pairs(
    candidates: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """Keep only pairs whose two candidates pass the local privacy screen."""
    safe: list[tuple[int, int]] = []
    skipped = 0
    for i, j in pairs:
        if is_safe_public_candidate(candidates[i]) and is_safe_public_candidate(candidates[j]):
            safe.append((i, j))
        else:
            skipped += 1
    return safe, skipped


def _pair_question(index: int) -> dict[str, dict[str, Any]]:
    a_ref = f"`pairs[{index}].a_title`"
    b_ref = f"`pairs[{index}].b_title`"
    return {
        f"pair{index}_same_issue": {
            "type": "noul",
            "instructions": (
                f"{a_ref} と {b_ref} は、同一の根本課題に対する改善案か？"
            ),
            "criteria": {
                "true": "対象箇所と目的が同じで、片方を実装すればもう片方も解消される。",
                "false": "対象箇所または目的が異なり、両方を別々に実装する必要がある。",
            },
        }
    }


def build_pair_request(
    candidates: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int]],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Build one batched System One request covering every ambiguous pair."""
    questions: dict[str, dict[str, Any]] = {}
    state_pairs: list[dict[str, str]] = []
    for position, (i, j) in enumerate(pairs):
        state_pairs.append(_public_pair(candidates[i], candidates[j]))
        questions.update(_pair_question(position))
    return {
        "state": {"pairs": state_pairs},
        "model": model,
        "questions": questions,
    }


def _send_request(payload: dict[str, Any], *, api_key: str, timeout: float) -> Mapping[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            os.environ.get("TYPESAFE_ENDPOINT", TYPESAFE_ENDPOINT),
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def _request_worker(connection, payload: dict[str, Any], api_key: str, timeout: float) -> None:
    """Run the blocking HTTP stack in a process that the parent can terminate."""
    try:
        body = _send_request(payload, api_key=api_key, timeout=timeout)
        connection.send(("ok", body))
    except BaseException as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _request_with_hard_deadline(
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout: float,
    process_context=None,
) -> Mapping[str, Any]:
    """Terminate DNS/socket work that outlives the configured wall-clock bound."""
    context = process_context or multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_request_worker,
        args=(sender, payload, api_key, timeout),
        daemon=True,
    )
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout + 2.0):
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            raise TypeSafeDedupError("TypeSafe request exceeded the configured timeout")
        try:
            status, value = receiver.recv()
        except EOFError as exc:
            raise TypeSafeDedupError("TypeSafe request process exited without a response") from exc
    finally:
        receiver.close()
        if process.is_alive():
            process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    if status != "ok":
        raise TypeSafeDedupError(f"TypeSafe request failed: {value}")
    if not isinstance(value, Mapping):
        raise TypeSafeDedupError("TypeSafe response must be an object")
    return value


def _default_request(payload: dict[str, Any]) -> Mapping[str, Any]:
    api_key = _resolve_api_key()
    if not api_key:
        raise TypeSafeDedupError("TYPESAFE_API_KEY is not configured")
    timeout = float(os.environ.get("TYPESAFE_DEDUP_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    return _request_with_hard_deadline(payload, api_key=api_key, timeout=timeout)


def _noul(answers: Mapping[str, Any], question_id: str) -> float:
    raw = answers.get(question_id)
    if not isinstance(raw, Mapping) or raw.get("type") != "noul":
        raise TypeSafeDedupError(f"missing noul answer: {question_id}")
    try:
        value = float(raw["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeSafeDedupError(f"invalid noul answer: {question_id}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise TypeSafeDedupError(f"noul outside [0, 1]: {question_id}")
    return value


def route_pair(same_issue: float, *, threshold: float = SAME_ISSUE_MIN) -> str:
    """Apply the reviewable routing policy to one raw Jev probability.

    Kept as a separate function so the policy can be changed and re-measured
    without re-running inference.
    """
    return "duplicate" if same_issue >= threshold else "distinct"


def judge_pairs(
    candidates: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int]],
    *,
    request_fn: RequestFn | None = None,
    model: str | None = None,
    threshold: float = SAME_ISSUE_MIN,
    max_pairs: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Judge every ambiguous pair with one TypeSafe request."""
    limit = max_pairs or _max_pairs()
    selected = list(pairs)[:limit]
    if not selected:
        return [], {"status": "skipped", "reason": "no_pairs"}

    selected_model = model or os.environ.get("TYPESAFE_MODEL", DEFAULT_MODEL)
    payload = build_pair_request(candidates, selected, model=selected_model)
    body = (request_fn or _default_request)(payload)
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeSafeDedupError("TypeSafe response is missing answers")

    judged: list[dict[str, Any]] = []
    for position, (i, j) in enumerate(selected):
        same_issue = _noul(answers, f"pair{position}_same_issue")
        judged.append(
            {
                "a": i,
                "b": j,
                "same_issue": same_issue,
                "route": route_pair(same_issue, threshold=threshold),
            }
        )

    duplicates = sum(1 for item in judged if item["route"] == "duplicate")
    return judged, {
        "status": "applied",
        "model": str(body.get("model") or selected_model),
        "pair_count": len(selected),
        "truncated_pairs": max(0, len(pairs) - len(selected)),
        "duplicate_count": duplicates,
        "distinct_count": len(judged) - duplicates,
        "threshold": threshold,
        "usage": dict(body.get("usage") or {}),
    }


def judge_pairs_if_enabled(
    candidates: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int]],
    *,
    request_fn: RequestFn | None = None,
    threshold: float = SAME_ISSUE_MIN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply Jev only when explicitly enabled; fail open to current behavior.

    On any failure every ambiguous pair is reported as "distinct", which is what
    ``deduplicate_improvements()`` already concludes for the gray band today.
    """
    if request_fn is None and not typesafe_dedup_enabled():
        return _all_distinct(pairs), {"status": "disabled"}
    try:
        return judge_pairs(candidates, pairs, request_fn=request_fn, threshold=threshold)
    except Exception as exc:
        return _all_distinct(pairs), {
            "status": "fallback",
            "error_type": type(exc).__name__,
        }


def _all_distinct(pairs: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {"a": i, "b": j, "same_issue": None, "route": "distinct"} for i, j in pairs
    ]


__all__ = [
    "AUTO_MERGE_MIN",
    "GRAY_HIGH",
    "GRAY_LOW",
    "SAME_ISSUE_MIN",
    "TypeSafeDedupError",
    "build_pair_request",
    "filter_safe_pairs",
    "is_safe_public_candidate",
    "judge_pairs",
    "judge_pairs_if_enabled",
    "route_pair",
    "select_gray_pairs",
    "typesafe_dedup_enabled",
]
