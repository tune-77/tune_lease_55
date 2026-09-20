"""Read-only Obsidian note alignment with optional TypeSafe/Jev judgments.

The module never edits a Vault.  It first builds a small candidate set with
deterministic title, alias, and tag signals.  Only an explicitly enabled live
run sends ambiguous pairs to TypeSafe, and local paths are never included in
the request.
"""

from __future__ import annotations

import math
import os
import re
import signal
import threading
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts._obsidian_common import (
    note_tags,
    parse_note_frontmatter,
    safe_rel,
    strip_frontmatter,
)

DEFAULT_INCLUDE_ROOTS = (
    "03-知識_業界",
    "リース実務知識",
    "Projects/tune_lease_55/Asset Knowledge",
)
DEFAULT_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".obsidian",
        ".trash",
        "Archive",
        "Daily",
        "Work Logs",
        "AI Chat",
        "Dialogue",
        "Private Reflection",
        "System Improvement Reflection",
        "leaseDb_データ",
    }
)
SENSITIVE_NAME_MARKERS = (
    "screening_records",
    "past_cases",
    "案件一覧",
    "顧客一覧",
)

ALIGNMENT_LEVELS = (
    "The notes describe different concepts and do not need a relationship link.",
    "The notes describe related or complementary knowledge and may benefit from a Related link, but should remain separate.",
    "The notes substantially describe the same reusable knowledge and should be reviewed as duplicate or merge candidates.",
)

MAX_TITLE_CHARS = 200
MAX_ALIASES = 12
MAX_TAGS = 20
MAX_EXCERPT_CHARS = 1200
DEFAULT_MIN_SIMILARITY = 0.18
DEFAULT_MAX_PAIRS = 40
DEFAULT_MAX_LINKED_ONLY_PAIRS = 5
LOW_CONFIDENCE = 0.45
CONTRADICTION_REVIEW_MIN = 0.70
MACOS_DATALESS_FLAG = 0x40000000

RequestFn = Callable[[dict[str, Any]], Mapping[str, Any]]


class ObsidianAlignmentError(RuntimeError):
    """Raised when an alignment response cannot safely drive a report."""


@dataclass(frozen=True)
class NoteEntity:
    path: str
    title: str
    aliases: tuple[str, ...]
    tags: tuple[str, ...]
    excerpt: str
    wikilinks: tuple[str, ...]
    outline: tuple[str, ...] = ()

    def public_state(self) -> dict[str, Any]:
        """Return the minimum note representation allowed to leave the machine."""
        return {
            "title": self.title[:MAX_TITLE_CHARS],
            "aliases": list(self.aliases[:MAX_ALIASES]),
            "tags": list(self.tags[:MAX_TAGS]),
            "outline": list(self.outline[:20]),
        }


@dataclass(frozen=True)
class CandidatePair:
    a: int
    b: int
    similarity: float
    reasons: tuple[str, ...]
    already_linked: bool


def _as_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        return ()
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _note_title(meta: Mapping[str, Any], body: str, path: Path) -> str:
    frontmatter_title = str(meta.get("title") or "").strip()
    if frontmatter_title:
        return frontmatter_title
    for line in _unfenced_lines(body):
        if line.startswith("# "):
            heading = line[2:].strip()
            if heading:
                return heading
    return path.stem


def _unfenced_lines(body: str) -> list[str]:
    """Return Markdown lines outside matching backtick or tilde fences."""
    lines: list[str] = []
    fence_char = ""
    fence_length = 0
    for raw in body.splitlines():
        if fence_char:
            closing = re.match(r"^ {0,3}([`~]+)[ \t]*$", raw)
            if (
                closing
                and set(closing.group(1)) == {fence_char}
                and len(closing.group(1)) >= fence_length
            ):
                fence_char = ""
                fence_length = 0
            continue
        opening = re.match(r"^ {0,3}(`{3,}|~{3,}).*$", raw)
        if opening:
            fence_char = opening.group(1)[0]
            fence_length = len(opening.group(1))
            continue
        lines.append(raw)
    return lines


def _excerpt(body: str) -> str:
    lines: list[str] = []
    for raw in _unfenced_lines(body):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
        if sum(len(item) for item in lines) >= MAX_EXCERPT_CHARS:
            break
    return "\n".join(lines)[:MAX_EXCERPT_CHARS]


def _outline(body: str) -> tuple[str, ...]:
    headings: list[str] = []
    for raw in _unfenced_lines(body):
        match = re.match(r"^#{1,3}\s+(.+?)\s*$", raw)
        if not match:
            continue
        heading = match.group(1).strip()
        if heading and heading not in headings:
            headings.append(heading[:200])
        if len(headings) >= 20:
            break
    return tuple(headings)


def _wikilinks(body: str) -> tuple[str, ...]:
    links: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", body):
        target = raw.strip()
        if target.endswith(".md"):
            target = target[:-3]
        key = Path(target).stem
        if key and key not in seen:
            seen.add(key)
            links.append(key)
    return tuple(links)


def load_notes(
    vault: Path,
    *,
    include_roots: Sequence[str] = DEFAULT_INCLUDE_ROOTS,
    excluded_parts: frozenset[str] = DEFAULT_EXCLUDED_PARTS,
    max_notes: int = 1000,
    read_timeout_seconds: float = 0.5,
) -> list[NoteEntity]:
    """Load safe knowledge-note metadata and compact excerpts from a Vault."""
    if not vault.is_dir():
        raise FileNotFoundError(f"Obsidian Vault not found: {vault}")
    paths: set[Path] = set()
    for relative_root in include_roots:
        root = vault / relative_root
        if root.is_dir():
            paths.update(root.rglob("*.md"))

    notes: list[NoteEntity] = []
    vault_resolved = vault.resolve()
    for path in sorted(paths):
        relative = path.relative_to(vault)
        prefixes = (vault.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1))
        if any(prefix.is_symlink() for prefix in prefixes):
            continue
        try:
            resolved_relative = path.resolve(strict=True).relative_to(vault_resolved)
        except (OSError, ValueError):
            continue
        if any(part in excluded_parts for part in relative.parts[:-1]):
            continue
        if any(
            marker.casefold() in part.casefold()
            for part in (*relative.parts, *resolved_relative.parts)
            for marker in SENSITIVE_NAME_MARKERS
        ):
            continue
        if _is_dataless(path):
            continue
        try:
            raw = _read_text_with_timeout(path, read_timeout_seconds)
        except (OSError, TimeoutError):
            continue
        meta = parse_note_frontmatter(raw)
        body = strip_frontmatter(raw)
        title = _note_title(meta, body, path)
        notes.append(
            NoteEntity(
                path=safe_rel(path, vault),
                title=title,
                aliases=_as_string_list(meta.get("aliases") or meta.get("alias")),
                tags=tuple(note_tags(dict(meta))),
                excerpt=_excerpt(body),
                wikilinks=_wikilinks(body),
                outline=_outline(body),
            )
        )
        if len(notes) >= max_notes:
            break
    return notes


def _is_dataless(path: Path) -> bool:
    """Return true for an iCloud placeholder whose content is not local."""
    try:
        flags = int(getattr(path.stat(), "st_flags", 0) or 0)
    except OSError:
        return True
    return bool(flags & MACOS_DATALESS_FLAG)


def _read_text_with_timeout(path: Path, timeout_seconds: float) -> str:
    """Read one note without letting an iCloud placeholder stall the scan."""
    timeout = max(0.0, float(timeout_seconds))
    can_alarm = (
        timeout > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
    )
    if not can_alarm:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _raise_timeout(_signum, _frame) -> None:
        raise TimeoutError(f"timed out reading note: {path.name}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"^20\d{2}[-_/]\d{1,2}[-_/]\d{1,2}[_\s-]*", "", text)
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龠]+", "", text)


def _bigrams(value: str) -> set[str]:
    normalized = normalize_label(value)
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _label_set(note: NoteEntity) -> set[str]:
    return {normalized for value in (note.title, *note.aliases) if (normalized := normalize_label(value))}


def _tag_set(note: NoteEntity) -> set[str]:
    return {normalize_label(tag) for tag in note.tags if normalize_label(tag)}


def _link_identity_set(note: NoteEntity) -> set[str]:
    """Return labels by which an Obsidian link can identify this note."""
    values = (note.title, Path(note.path).stem, *note.aliases)
    return {normalized for value in values if (normalized := normalize_label(value))}


def _linked(left: NoteEntity, right: NoteEntity) -> bool:
    left_targets = {normalize_label(value) for value in left.wikilinks}
    right_targets = {normalize_label(value) for value in right.wikilinks}
    return bool(
        left_targets & _link_identity_set(right)
        or right_targets & _link_identity_set(left)
    )


def select_candidate_pairs(
    notes: Sequence[NoteEntity],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    max_pairs: int = DEFAULT_MAX_PAIRS,
) -> list[CandidatePair]:
    """Select likely duplicate or related pairs using only local deterministic signals."""
    candidates: list[CandidatePair] = []
    raw_title_bigrams = [_bigrams(note.title) for note in notes]
    document_frequency = Counter(
        gram for grams in raw_title_bigrams for gram in grams
    )
    # Domain-wide fragments such as 「リース」 and template suffixes should not
    # make otherwise unrelated knowledge notes candidates. Keep rarer bigrams.
    frequent_cutoff = max(2, math.ceil(len(notes) * 0.05))
    title_bigrams = [
        {gram for gram in grams if document_frequency[gram] <= frequent_cutoff}
        for grams in raw_title_bigrams
    ]
    labels = [_label_set(note) for note in notes]
    tags = [_tag_set(note) for note in notes]

    for i, left in enumerate(notes):
        for j in range(i + 1, len(notes)):
            right = notes[j]
            reasons: list[str] = []
            title_similarity = _jaccard(title_bigrams[i], title_bigrams[j])
            alias_overlap = labels[i] & labels[j]
            tag_overlap = tags[i] & tags[j]
            containment = any(
                min(len(a), len(b)) >= 4 and (a in b or b in a)
                for a in labels[i]
                for b in labels[j]
            )

            if alias_overlap:
                reasons.append("title_or_alias_match")
            if containment:
                reasons.append("title_containment")
            if title_similarity >= min_similarity:
                reasons.append("title_similarity")
            if tag_overlap and title_similarity >= min_similarity * 0.65:
                reasons.append("shared_tags")
            direct_linked = _linked(left, right)
            if direct_linked:
                reasons.append("existing_link")

            if not reasons:
                continue
            bonus = 0.25 if alias_overlap else 0.12 if containment else 0.0
            tag_bonus = min(0.12, 0.04 * len(tag_overlap))
            similarity = min(1.0, title_similarity + bonus + tag_bonus)
            candidates.append(
                CandidatePair(
                    a=i,
                    b=j,
                    similarity=round(similarity, 4),
                    reasons=tuple(reasons),
                    already_linked=direct_linked,
                )
            )

    candidates.sort(
        key=lambda pair: (
            pair.reasons == ("existing_link",),
            -pair.similarity,
            notes[pair.a].path,
            notes[pair.b].path,
        )
    )
    selected: list[CandidatePair] = []
    linked_only_count = 0
    for pair in candidates:
        if pair.reasons == ("existing_link",):
            if linked_only_count >= DEFAULT_MAX_LINKED_ONLY_PAIRS:
                continue
            linked_only_count += 1
        selected.append(pair)
        if len(selected) >= max(1, max_pairs):
            break
    return selected


def build_alignment_request(
    notes: Sequence[NoteEntity],
    pairs: Sequence[CandidatePair],
    *,
    model: str = "jev-latest",
) -> dict[str, Any]:
    state_pairs: list[dict[str, Any]] = []
    questions: dict[str, dict[str, Any]] = {}
    for index, pair in enumerate(pairs):
        state_pairs.append(
            {
                "note_a": notes[pair.a].public_state(),
                "note_b": notes[pair.b].public_state(),
            }
        )
        pair_ref = f"`pairs[{index}]`"
        questions[f"pair{index}_alignment"] = {
            "type": "score",
            "instructions": f"How should the two Obsidian knowledge notes in {pair_ref} be aligned?",
            "criteria": list(ALIGNMENT_LEVELS),
        }
        questions[f"pair{index}_same_subject"] = {
            "type": "noul",
            "instructions": f"Do both notes in {pair_ref} describe the same principal subject?",
        }
        questions[f"pair{index}_same_conclusion"] = {
            "type": "noul",
            "instructions": f"Do both notes in {pair_ref} support substantially the same reusable conclusion or rule?",
        }
        questions[f"pair{index}_contradiction"] = {
            "type": "noul",
            "instructions": f"Do the two notes in {pair_ref} make materially conflicting claims that should be reviewed before linking or merging?",
        }
    return {"state": {"pairs": state_pairs}, "model": model, "questions": questions}


def _score_answer(answers: Mapping[str, Any], question_id: str) -> dict[str, Any]:
    raw = answers.get(question_id)
    if not isinstance(raw, Mapping) or raw.get("type") != "score":
        raise ObsidianAlignmentError(f"missing score answer: {question_id}")
    try:
        score = float(raw["score"])
        confidence = float(raw["confidence"])
        probabilities = {str(key): float(value) for key, value in dict(raw["probabilities"]).items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ObsidianAlignmentError(f"invalid score answer: {question_id}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= len(ALIGNMENT_LEVELS) - 1:
        raise ObsidianAlignmentError(f"score outside range: {question_id}")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ObsidianAlignmentError(f"confidence outside [0, 1]: {question_id}")
    return {"score": score, "confidence": confidence, "probabilities": probabilities}


def _noul_answer(answers: Mapping[str, Any], question_id: str) -> float:
    raw = answers.get(question_id)
    if not isinstance(raw, Mapping) or raw.get("type") != "noul":
        raise ObsidianAlignmentError(f"missing noul answer: {question_id}")
    try:
        value = float(raw["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ObsidianAlignmentError(f"invalid noul answer: {question_id}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ObsidianAlignmentError(f"noul outside [0, 1]: {question_id}")
    return value


def route_alignment(
    *,
    score: float,
    confidence: float,
    contradiction: float,
    already_linked: bool,
) -> str:
    """Turn raw judgments into a read-only review route."""
    if contradiction >= CONTRADICTION_REVIEW_MIN:
        return "contradiction_review"
    if confidence < LOW_CONFIDENCE:
        return "uncertain_review"
    level = min(int(score + 0.5), len(ALIGNMENT_LEVELS) - 1)
    if level == 2:
        return "duplicate_review"
    if level == 1:
        return "already_linked" if already_linked else "related_link_candidate"
    return "no_action"


def judge_pairs(
    notes: Sequence[NoteEntity],
    pairs: Sequence[CandidatePair],
    *,
    request_fn: RequestFn | None = None,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not pairs:
        return [], {"status": "skipped", "reason": "no_pairs"}
    selected_model = model or os.environ.get("TYPESAFE_MODEL", "jev-latest")
    payload = build_alignment_request(notes, pairs, model=selected_model)
    if request_fn is None:
        from typesafe_rag_guard import request_system_one

        request_fn = request_system_one
    body = request_fn(payload)
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise ObsidianAlignmentError("TypeSafe response is missing answers")

    judged: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        alignment = _score_answer(answers, f"pair{index}_alignment")
        same_subject = _noul_answer(answers, f"pair{index}_same_subject")
        same_conclusion = _noul_answer(answers, f"pair{index}_same_conclusion")
        contradiction = _noul_answer(answers, f"pair{index}_contradiction")
        route = route_alignment(
            score=alignment["score"],
            confidence=alignment["confidence"],
            contradiction=contradiction,
            already_linked=pair.already_linked,
        )
        judged.append(
            {
                "a": pair.a,
                "b": pair.b,
                "local_similarity": pair.similarity,
                "reasons": list(pair.reasons),
                "already_linked": pair.already_linked,
                **alignment,
                "same_subject": same_subject,
                "same_conclusion": same_conclusion,
                "contradiction": contradiction,
                "route": route,
            }
        )
    return judged, {
        "status": "applied",
        "model": str(body.get("model") or selected_model),
        "pair_count": len(judged),
        "usage": dict(body.get("usage") or {}),
    }


def alignment_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    enabled = str(env.get("TYPESAFE_OBSIDIAN_ALIGNMENT_ENABLED") or "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return False
    try:
        from typesafe_rag_guard import typesafe_available

        return bool(typesafe_available(env))
    except Exception:
        return False


def serialize_notes(notes: Sequence[NoteEntity]) -> list[dict[str, Any]]:
    return [asdict(note) for note in notes]
