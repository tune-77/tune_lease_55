"""Read-only candidate inbox plus separate human review state for memory items."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from difflib import SequenceMatcher
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from runtime_paths import get_data_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = get_data_dir()

REVIEW_STATE_PATH = DATA_DIR / "memory_review_inbox_state.json"
REVIEW_AUDIT_PATH = DATA_DIR / "memory_review_inbox_audit.jsonl"

SOURCE_PATHS: dict[str, Path] = {
    "judgment_materials_preview": DATA_DIR / "judgment_materials_preview.jsonl",
    "autoresearch_candidates": DATA_DIR / "autoresearch_judgment_asset_candidates.jsonl",
    "reflection_action_candidates": DATA_DIR / "reflection_action_candidates.jsonl",
    "prediction_error_candidates": DATA_DIR / "prediction_error_update_candidates.jsonl",
    "obsidian_memory_insight_candidates": DATA_DIR / "obsidian_memory_insight_candidates.jsonl",
}

Decision = Literal["adopted", "revised", "held", "rejected"]

TERMINAL_STATUSES = {"adopted", "revised", "held", "rejected"}
AUTO_REJECT_JACCARD_THRESHOLD = 0.82
AUTO_REJECT_SEQUENCE_THRESHOLD = 0.93
AUTO_REJECT_MIN_TEXT_LEN = 18
AUTO_REJECT_REVIEW_WEEKDAY = 0  # Monday
# Statuses a fresh near-duplicate candidate should inherit instead of resurfacing
# as an unreviewed "candidate". Excludes "rejected", which already has its own
# persisted-pattern path via _apply_auto_rejections.
CARRY_OVER_STATUSES = {"adopted", "revised", "held"}

# Guards the review-state read-modify-write cycle in review_candidate(). Without
# it, two near-simultaneous reviews can each read the state before the other's
# write lands, and the later save silently drops the earlier decision -- making
# an already-reviewed item look like an unreviewed candidate again.
_STATE_LOCK = threading.RLock()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_review_state(path: Path | None = None) -> dict[str, Any]:
    path = path or REVIEW_STATE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "reviews": {}}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "reviews": {}}
    reviews = payload.get("reviews")
    if not isinstance(reviews, dict):
        payload["reviews"] = {}
    rejection_patterns = payload.get("rejection_patterns")
    if not isinstance(rejection_patterns, list):
        payload["rejection_patterns"] = []
    return payload


def save_review_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = path or REVIEW_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    state["schema_version"] = 1
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _source_item_id(row: dict[str, Any]) -> str:
    for key in ("id", "candidate_id", "canonical_key", "event_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    seed = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def make_inbox_id(source_name: str, row: dict[str, Any]) -> str:
    return f"{source_name}__{_source_item_id(row)}"


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _auto_reject_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("source", "candidate_type", "topic", "title", "claim")
        if str(item.get(key) or "").strip()
    )


def _normalize_auto_reject_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _text_similarity(left: str, right: str) -> float:
    left = _normalize_auto_reject_text(left)
    right = _normalize_auto_reject_text(right)
    if len(left) < AUTO_REJECT_MIN_TEXT_LEN or len(right) < AUTO_REJECT_MIN_TEXT_LEN:
        return 0.0
    if left == right:
        return 1.0
    if len(left) >= 28 and left in right:
        return 0.96
    if len(right) >= 28 and right in left:
        return 0.96
    left_ngrams = _char_ngrams(left)
    right_ngrams = _char_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    jaccard = len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)
    sequence = SequenceMatcher(None, left, right).ratio()
    return max(jaccard, sequence)


def _pattern_matches_item(pattern: dict[str, Any], item: dict[str, Any]) -> tuple[bool, float]:
    if str(pattern.get("source") or "") != str(item.get("source") or ""):
        return False, 0.0
    pattern_type = str(pattern.get("candidate_type") or "")
    item_type = str(item.get("candidate_type") or "")
    if pattern_type and item_type and pattern_type != item_type:
        return False, 0.0
    similarity = _text_similarity(str(pattern.get("text") or ""), _auto_reject_text(item))
    return (
        similarity >= AUTO_REJECT_JACCARD_THRESHOLD
        or similarity >= AUTO_REJECT_SEQUENCE_THRESHOLD,
        similarity,
    )


def _rejection_pattern_for_item(item: dict[str, Any], *, reviewed_at: str = "") -> dict[str, Any]:
    text = _auto_reject_text(item)
    normalized = _normalize_auto_reject_text(text)
    return {
        "source": str(item.get("source") or ""),
        "source_item_id": str(item.get("source_item_id") or ""),
        "inbox_id": str(item.get("inbox_id") or ""),
        "candidate_type": str(item.get("candidate_type") or ""),
        "topic": str(item.get("topic") or ""),
        "title": str(item.get("title") or ""),
        "claim": str(item.get("claim") or ""),
        "text": text,
        "text_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16] if normalized else "",
        "reviewed_at": reviewed_at or str(item.get("reviewed_at") or ""),
    }


def _learn_rejection_pattern(state: dict[str, Any], item: dict[str, Any], *, reviewed_at: str) -> None:
    patterns = state.setdefault("rejection_patterns", [])
    if not isinstance(patterns, list):
        patterns = []
        state["rejection_patterns"] = patterns
    pattern = _rejection_pattern_for_item(item, reviewed_at=reviewed_at)
    if not pattern["text_hash"]:
        return
    patterns[:] = [
        existing
        for existing in patterns
        if str(existing.get("inbox_id") or "") != pattern["inbox_id"]
        and str(existing.get("text_hash") or "") != pattern["text_hash"]
    ]
    patterns.append(pattern)


def _forget_rejection_pattern(state: dict[str, Any], inbox_id: str) -> None:
    patterns = state.get("rejection_patterns")
    if not isinstance(patterns, list):
        return
    patterns[:] = [pattern for pattern in patterns if str(pattern.get("inbox_id") or "") != inbox_id]


def normalize_candidate(source_name: str, row: dict[str, Any], review: dict[str, Any] | None = None) -> dict[str, Any]:
    review = review or {}
    inbox_id = make_inbox_id(source_name, row)
    source_item_id = _source_item_id(row)
    claim = _first_text(row, ("claim", "canonical_statement", "proposed_change", "suggested_update", "body", "hypothesis"))
    title = _first_text(row, ("title", "research_title", "target_belief", "kind", "candidate_type"))
    evidence_path = _first_text(row, ("evidence_path", "source_path", "evidence"))
    if not evidence_path:
        paths = row.get("evidence_paths") or row.get("source_paths") or []
        if isinstance(paths, list) and paths:
            evidence_path = str(paths[0])
    status = str(review.get("status") or "candidate")
    has_human_review = bool(review.get("status"))
    edited_claim = str(review.get("edited_claim") or row.get("edited_claim") or "")
    return {
        "inbox_id": inbox_id,
        "source": source_name,
        "source_item_id": source_item_id,
        "status": status,
        "has_human_review": has_human_review,
        "auto_rejected": False,
        "auto_reject_reason": "",
        "auto_reject_matched_inbox_id": "",
        "auto_reject_score": 0.0,
        "auto_carried_over": False,
        "auto_carried_over_reason": "",
        "auto_carried_over_matched_inbox_id": "",
        "auto_carried_over_score": 0.0,
        "title": title or source_item_id,
        "claim": claim,
        "edited_claim": edited_claim,
        "note": str(review.get("note") or ""),
        "reviewed_at": str(review.get("reviewed_at") or ""),
        "candidate_type": _first_text(row, ("candidate_type", "material_type", "action_schema", "error_type")),
        "quality": _first_text(row, ("asset_quality", "quality", "priority", "confidence")),
        "topic": _first_text(row, ("research_topic", "concept", "target", "domain")),
        "date_hint": _first_text(row, ("research_date", "date", "target_date", "created_at", "generated_at")),
        "evidence_path": evidence_path,
        "raw_status": _first_text(row, ("status", "review_status", "promotion_status", "human_decision_status", "promote_status")),
        "source_path": str(SOURCE_PATHS.get(source_name, Path(""))),
    }


def _apply_auto_rejections(items: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    stored_patterns = [
        pattern
        for pattern in state.get("rejection_patterns", [])
        if isinstance(pattern, dict) and str(pattern.get("text_hash") or "")
    ]
    learned_from_reviews = [
        _rejection_pattern_for_item(item, reviewed_at=str(item.get("reviewed_at") or ""))
        for item in items
        if item.get("has_human_review") and str(item.get("status") or "") == "rejected"
    ]
    patterns = stored_patterns + [
        pattern
        for pattern in learned_from_reviews
        if str(pattern.get("text_hash") or "")
        and str(pattern.get("text_hash") or "")
        not in {str(stored.get("text_hash") or "") for stored in stored_patterns}
    ]
    if not patterns:
        return items
    for item in items:
        if item.get("has_human_review") or str(item.get("status") or "candidate") != "candidate":
            continue
        best_pattern: dict[str, Any] | None = None
        best_score = 0.0
        for pattern in patterns:
            matched, score = _pattern_matches_item(pattern, item)
            if matched and score > best_score:
                best_pattern = pattern
                best_score = score
        if best_pattern:
            item["status"] = "rejected"
            item["auto_rejected"] = True
            item["auto_reject_reason"] = "過去に却下された同種候補と高一致したため自動却下"
            item["auto_reject_matched_inbox_id"] = str(best_pattern.get("inbox_id") or "")
            item["auto_reject_score"] = round(best_score, 3)
            item["note"] = item["auto_reject_reason"]
    return items


def _apply_carried_over_reviews(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Carry an existing adopted/revised/held decision onto a near-duplicate candidate.

    Some source pipelines mint a fresh row id per detection run rather than per
    underlying claim (e.g. prediction_error_loop's timestamp-based candidate_id),
    so the same already-reviewed item can re-emit under a new inbox_id and show
    up again as an unreviewed "candidate". _apply_auto_rejections already guards
    against this for rejected items via persisted rejection_patterns; this does
    the equivalent, session-local check for the other terminal decisions so a
    human isn't asked to re-review something already decided.
    """
    decided = [
        item
        for item in items
        if item.get("has_human_review") and str(item.get("status") or "") in CARRY_OVER_STATUSES
    ]
    if not decided:
        return items
    decided_patterns = [(_rejection_pattern_for_item(item), item) for item in decided]
    for item in items:
        if item.get("has_human_review") or str(item.get("status") or "candidate") != "candidate":
            continue
        best_match: dict[str, Any] | None = None
        best_score = 0.0
        for pattern, source_item in decided_patterns:
            matched, score = _pattern_matches_item(pattern, item)
            if matched and score > best_score:
                best_match = source_item
                best_score = score
        if best_match:
            item["status"] = str(best_match.get("status") or "")
            item["auto_carried_over"] = True
            item["auto_carried_over_reason"] = "既にレビュー済みの類似候補と高一致したため前回の判断を引き継ぎ"
            item["auto_carried_over_matched_inbox_id"] = str(best_match.get("inbox_id") or "")
            item["auto_carried_over_score"] = round(best_score, 3)
            item["note"] = item["auto_carried_over_reason"]
    return items


def load_candidates() -> list[dict[str, Any]]:
    state = load_review_state()
    reviews = state.get("reviews") if isinstance(state.get("reviews"), dict) else {}
    items: list[dict[str, Any]] = []
    for source_name, path in SOURCE_PATHS.items():
        for row in _read_jsonl(path):
            inbox_id = make_inbox_id(source_name, row)
            review = reviews.get(inbox_id) if isinstance(reviews.get(inbox_id), dict) else {}
            items.append(normalize_candidate(source_name, row, review))
    items = _apply_auto_rejections(items, state)
    return _apply_carried_over_reviews(items)


def summarize_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(item.get("status") or "candidate") for item in items)
    by_source = Counter(str(item.get("source") or "unknown") for item in items)
    return {
        "total": len(items),
        "open": sum(1 for item in items if str(item.get("status") or "candidate") == "candidate"),
        "by_status": dict(sorted(by_status.items())),
        "by_source": dict(sorted(by_source.items())),
    }


def _next_review_date(today: date | None = None) -> str:
    today = today or date.today()
    days_until_review = (AUTO_REJECT_REVIEW_WEEKDAY - today.weekday()) % 7
    if days_until_review == 0:
        days_until_review = 7
    return (today + timedelta(days=days_until_review)).isoformat()


def auto_reject_review_policy(state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = state or load_review_state()
    patterns = state.get("rejection_patterns")
    persisted_pattern_count = len(patterns) if isinstance(patterns, list) else 0
    reviews = state.get("reviews") if isinstance(state.get("reviews"), dict) else {}
    rejected_review_count = sum(
        1
        for review in reviews.values()
        if isinstance(review, dict) and str(review.get("status") or "") == "rejected"
    )
    return {
        "cadence": "weekly",
        "review_weekday": "Monday",
        "next_review_date": _next_review_date(),
        "pattern_count": persisted_pattern_count + rejected_review_count,
        "persisted_pattern_count": persisted_pattern_count,
        "rejected_review_count": rejected_review_count,
        "policy": "auto_reject_runs_inline_review_patterns_weekly_or_on_false_positive",
        "reason": "毎日見るとレビュー負荷が増えるため、定期棚卸しは週1回に抑える。",
    }


def list_inbox(
    *,
    status: str = "candidate",
    source: str = "all",
    q: str = "",
    limit: int = 30,
    offset: int = 0,
) -> dict[str, Any]:
    items = load_candidates()
    state = load_review_state()
    filtered = items
    if status != "all":
        filtered = [item for item in filtered if str(item.get("status") or "candidate") == status]
    if source != "all":
        filtered = [item for item in filtered if str(item.get("source") or "") == source]
    query = q.strip().lower()
    if query:
        filtered = [
            item
            for item in filtered
            if query
            in " ".join(
                str(item.get(key) or "")
                for key in ("title", "claim", "topic", "candidate_type", "evidence_path")
            ).lower()
        ]
    filtered.sort(
        key=lambda item: (
            str(item.get("date_hint") or ""),
            str(item.get("source") or ""),
            str(item.get("source_item_id") or ""),
        ),
        reverse=True,
    )
    limit = max(1, min(int(limit or 30), 100))
    offset = max(0, int(offset or 0))
    return {
        "summary": summarize_items(items),
        "filtered_total": len(filtered),
        "limit": limit,
        "offset": offset,
        "items": filtered[offset : offset + limit],
        "review_policy": "state_only_no_source_mutation",
        "auto_reject_review_policy": auto_reject_review_policy(state),
        "state_path": _display_path(REVIEW_STATE_PATH),
    }


def review_candidate(
    inbox_id: str,
    *,
    decision: Decision,
    note: str = "",
    edited_claim: str = "",
) -> dict[str, Any]:
    candidates = {str(item["inbox_id"]): item for item in load_candidates()}
    if inbox_id not in candidates:
        raise KeyError(inbox_id)
    if decision == "revised" and not edited_claim.strip():
        raise ValueError("edited_claim is required when decision is revised")
    now = datetime.now().isoformat(timespec="seconds")
    with _STATE_LOCK:
        state = load_review_state()
        reviews = state.setdefault("reviews", {})
        current = {
            "status": decision,
            "note": note.strip()[:1000],
            "edited_claim": edited_claim.strip()[:4000],
            "reviewed_at": now,
            "source": candidates[inbox_id]["source"],
            "source_item_id": candidates[inbox_id]["source_item_id"],
        }
        reviews[inbox_id] = current
        if decision == "rejected":
            _learn_rejection_pattern(state, candidates[inbox_id], reviewed_at=now)
        else:
            _forget_rejection_pattern(state, inbox_id)
        save_review_state(state)
    REVIEW_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now, "inbox_id": inbox_id, **current}, ensure_ascii=False) + "\n")
    return {
        **candidates[inbox_id],
        "status": decision,
        "note": current["note"],
        "edited_claim": current["edited_claim"],
        "reviewed_at": now,
    }
