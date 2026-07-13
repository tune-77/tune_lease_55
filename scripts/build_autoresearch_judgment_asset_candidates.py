#!/usr/bin/env python3
"""Build review-only judgment asset candidates from Auto Research notes.

Auto Research notes are useful, but they are not judgment assets yet. This
script turns substantive research sections into candidates that must be tested
in real cases and reviewed by a human before promotion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.auto_research_lease_judgment import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAULT,
    _section_content,
    _substantive_sections_present,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "autoresearch_judgment_asset_candidates.jsonl"
DEFAULT_STATE_JSON = DATA_DIR / "autoresearch_judgment_asset_candidate_state.json"
DEFAULT_SIMILARITY_THRESHOLD = 0.72
_STOP_TERMS = {
    "する",
    "します",
    "いる",
    "ある",
    "こと",
    "ため",
    "場合",
    "確認",
    "リース",
    "審査",
    "顧客",
    "物件",
    "判断",
}

DEFAULT_CANDIDATE_STATE = {
    "use_count": 0,
    "useful_count": 0,
    "rejected_count": 0,
    "neutral_count": 0,
    "last_used_at": "",
    "last_feedback_at": "",
    "verified_status": "unverified",
    "verification_note": "",
    "edited_claim": "",
    "edit_count": 0,
    "last_edited_at": "",
}

CANDIDATE_SECTIONS = {
    "リース審査への適用": "application_rule",
    "担当者が確認する質問": "confirmation_question",
    "承認条件を変える兆候": "condition_signal",
    "反証・過信してはいけない点": "caution",
}

TEXTBOOK_GENERAL_MARKERS = (
    "財務内容を確認",
    "返済原資を確認",
    "業界動向を確認",
    "担保価値を確認",
    "資金使途を確認",
    "収益性を確認",
    "安全性を確認",
    "総合的に判断",
    "慎重に判断",
    "必要がある",
    "確認する必要がある",
)

ACTIONABLE_MARKERS = (
    "直近",
    "3か月",
    "6か月",
    "12か月",
    "月次",
    "期間",
    "比率",
    "粗利",
    "稼働率",
    "稼働状況",
    "固定費",
    "短期借入",
    "価格転嫁",
    "外注費",
    "故障頻度",
    "荷主",
    "契約期間",
    "ドライバー",
    "補助金",
    "入金時期",
    "つなぎ資金",
    "撤退",
    "設置撤去費",
    "二次流通",
    "追加保全",
    "承認条件",
    "否認",
    "反証",
    "過信",
    "決めつけ",
    "だけで",
    "条件にする",
    "確認する",
    "検討する",
    "見る",
)

SPECIFICITY_MARKERS = tuple(
    marker
    for marker in ACTIONABLE_MARKERS
    if marker not in {"確認する", "検討する", "見る"}
)


def _vault_path() -> Path:
    raw = os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or str(DEFAULT_VAULT)
    return Path(raw).expanduser()


def _clean_item(value: str, limit: int = 260) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip().strip("'\"")


def _note_title(text: str, path: Path) -> str:
    title = _frontmatter_value(text, "title")
    if title:
        return title
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else path.stem


def _note_date(text: str, path: Path) -> str:
    date = _frontmatter_value(text, "date")
    if date:
        return date
    match = re.match(r"(\d{4}-\d{2}-\d{2})_", path.name)
    return match.group(1) if match else ""


def _note_topic(text: str, path: Path) -> str:
    return _frontmatter_value(text, "research_topic") or path.stem


def _bullet_items(content: str) -> list[str]:
    items: list[str] = []
    for line in content.splitlines():
        item = _clean_item(line)
        if not item or item.startswith("#") or item.startswith("```"):
            continue
        if len(item) < 8:
            continue
        items.append(item)
    return items


def _candidate_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _candidate_state(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    state = dict(DEFAULT_CANDIDATE_STATE)
    if raw:
        for key in state:
            if key not in raw:
                continue
            if key.endswith("_count"):
                try:
                    state[key] = max(0, int(raw[key]))
                except (TypeError, ValueError):
                    state[key] = 0
            else:
                state[key] = str(raw[key] or "")
    if state["verified_status"] not in {"unverified", "supported", "contradicted", "unclear"}:
        state["verified_status"] = "unverified"
    return state


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    states: dict[str, dict[str, Any]] = {}
    for candidate_id, value in raw.items():
        if isinstance(candidate_id, str) and isinstance(value, dict):
            states[candidate_id] = _candidate_state(value)
    return states


def write_state(path: Path, candidates: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(existing)
    for item in candidates:
        merged.setdefault(str(item["id"]), _candidate_state(item))
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _judgment_asset_quality(claim: str, candidate_type: str) -> tuple[str, list[str]]:
    text = str(claim or "")
    reasons: list[str] = []
    if any(marker in text for marker in TEXTBOOK_GENERAL_MARKERS):
        reasons.append("textbook_general_marker")
    if "textbook_general_marker" in reasons and not any(marker in text for marker in SPECIFICITY_MARKERS):
        reasons.append("no_specific_case_marker")
    if len(text) < 18 and len(_similarity_terms(text)) <= 3:
        reasons.append("too_short_or_broad")
    has_action = any(marker in text for marker in ACTIONABLE_MARKERS)
    if not has_action:
        reasons.append("no_case_action_or_condition")
    if candidate_type == "confirmation_question" and not any(marker in text for marker in ("確認", "聞", "質問", "見る", "照合")):
        reasons.append("not_a_confirmation_action")
    if candidate_type == "condition_signal" and not any(marker in text for marker in ("場合", "兆候", "増え", "低下", "悪化", "条件", "保全", "否認")):
        reasons.append("not_a_condition_signal")
    if candidate_type == "caution" and not any(marker in text for marker in ("だけで", "決めつけ", "過信", "反証", "限らない", "除く")):
        reasons.append("not_a_caution")
    if "textbook_general_marker" in reasons and len(reasons) >= 2:
        return "textbook_general", reasons
    if "no_case_action_or_condition" in reasons or "too_short_or_broad" in reasons:
        return "textbook_general", reasons
    if candidate_type == "confirmation_question" and "not_a_confirmation_action" in reasons:
        return "textbook_general", reasons
    if candidate_type == "condition_signal" and "not_a_condition_signal" in reasons:
        return "textbook_general", reasons
    if candidate_type == "caution" and "not_a_caution" in reasons:
        return "textbook_general", reasons
    return "actionable", []


def _promotion_status(state: dict[str, Any], asset_quality: str = "actionable") -> str:
    if asset_quality != "actionable":
        return "not_promoted_textbook_general"
    if state["verified_status"] == "supported" and (int(state["useful_count"]) > 0 or int(state.get("edit_count") or 0) > 0):
        return "ready_for_promotion"
    if state["verified_status"] == "contradicted" or int(state["rejected_count"]) > int(state["useful_count"]):
        return "rejected_or_deprioritized"
    return "not_promoted"


def _similarity_terms(text: str) -> set[str]:
    normalized = re.sub(r"[、。,.()\[\]（）「」『』:：/・]", " ", text)
    tokens = re.findall(r"[A-Za-z0-9_]{3,}|[一-龥ぁ-んァ-ンー]{2,}", normalized)
    return {token.lower() for token in tokens if token not in _STOP_TERMS and len(token) >= 2}


def _claim_similarity(left: str, right: str) -> float:
    left_terms = _similarity_terms(left)
    right_terms = _similarity_terms(right)
    left_norm = re.sub(r"\s+", "", re.sub(r"[、。,.()\[\]（）「」『』:：/・]", "", left))
    right_norm = re.sub(r"\s+", "", re.sub(r"[、。,.()\[\]（）「」『』:：/・]", "", right))
    sequence_similarity = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    if not left_terms or not right_terms:
        return sequence_similarity
    overlap = len(left_terms & right_terms)
    smaller = min(len(left_terms), len(right_terms))
    term_similarity = overlap / smaller if smaller else 0.0
    return max(term_similarity, sequence_similarity)


def _candidate_rank(item: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(item.get("edit_count") or 0),
        int(item.get("useful_count") or 0),
        int(item.get("use_count") or 0),
        -int(item.get("rejected_count") or 0),
        len(str(item.get("claim") or "")),
    )


def _merge_similar_candidate(target: dict[str, Any], source: dict[str, Any], similarity: float) -> None:
    similar = list(target.get("similar_candidates") or [])
    similar.append(
        {
            "id": source["id"],
            "claim": source["claim"],
            "research_topic": source["research_topic"],
            "evidence_path": source["evidence_path"],
            "similarity": round(similarity, 2),
        }
    )
    target["similar_candidates"] = similar[:8]
    target["deduped_count"] = int(target.get("deduped_count") or 0) + 1
    evidence_paths = list(target.get("evidence_paths") or [target.get("evidence_path", "")])
    source_path = str(source.get("evidence_path") or "")
    if source_path and source_path not in evidence_paths:
        evidence_paths.append(source_path)
    target["evidence_paths"] = evidence_paths[:8]


def dedupe_similar_candidates(
    candidates: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: (value["research_date"], value["research_topic"], value["candidate_type"], value["claim"])):
        match_index = -1
        match_similarity = 0.0
        for index, existing in enumerate(kept):
            if existing["candidate_type"] != item["candidate_type"]:
                continue
            similarity = _claim_similarity(existing["claim"], item["claim"])
            if similarity >= threshold and similarity > match_similarity:
                match_index = index
                match_similarity = similarity
        if match_index < 0:
            item.setdefault("deduped_count", 0)
            item.setdefault("similar_candidates", [])
            item.setdefault("evidence_paths", [item.get("evidence_path", "")])
            kept.append(item)
            continue
        existing = kept[match_index]
        if _candidate_rank(item) > _candidate_rank(existing):
            _merge_similar_candidate(item, existing, match_similarity)
            kept[match_index] = item
        else:
            _merge_similar_candidate(existing, item, match_similarity)
    return kept


def _source_path(path: Path, vault: Path) -> str:
    try:
        return str(path.relative_to(vault))
    except ValueError:
        return str(path)


def _note_candidates(path: Path, *, vault: Path, states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not _substantive_sections_present(text):
        return []
    title = _note_title(text, path)
    date = _note_date(text, path)
    topic = _note_topic(text, path)
    evidence_path = _source_path(path, vault)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for section, candidate_type in CANDIDATE_SECTIONS.items():
        for item in _bullet_items(_section_content(text, section)):
            key = (candidate_type, item)
            if key in seen:
                continue
            seen.add(key)
            candidate_id = _candidate_id(date, topic, candidate_type, item)
            state = _candidate_state(states.get(candidate_id))
            asset_quality, quality_reasons = _judgment_asset_quality(item, candidate_type)
            candidates.append(
                {
                    "id": candidate_id,
                    "candidate_type": candidate_type,
                    "research_topic": topic,
                    "research_title": title,
                    "research_date": date,
                    "claim": item,
                    "source_section": section,
                    "evidence_path": evidence_path,
                    "review_status": "candidate",
                    "asset_quality": asset_quality,
                    "quality_reasons": quality_reasons,
                    "promotion_status": _promotion_status(state, asset_quality),
                    **state,
                    "requires_human_use_feedback": True,
                    "requires_result_verification": True,
                    "use_policy": (
                        "当たり前な一般論は候補止まり。案件の確認行動・承認条件・反証材料・否認理由へ"
                        "変換でき、有用性が確認できたものだけ昇格する。"
                    ),
                }
            )
    return candidates


def _recent_notes(research_dir: Path, *, end_date: dt.date, days: int) -> list[Path]:
    start_date = end_date - dt.timedelta(days=max(1, days) - 1)
    notes: list[tuple[str, Path]] = []
    if not research_dir.exists():
        return []
    for path in research_dir.glob("*.md"):
        text_head = path.read_text(encoding="utf-8", errors="ignore")[:1500]
        note_date = _note_date(text_head, path)
        try:
            parsed = dt.date.fromisoformat(note_date)
        except ValueError:
            continue
        if start_date <= parsed <= end_date:
            notes.append((note_date, path))
    return [path for _, path in sorted(notes)]


def extract_candidates(
    *,
    vault: Path,
    output_dir: str,
    end_date: dt.date,
    days: int,
    state_path: Path = DEFAULT_STATE_JSON,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    states = load_state(state_path)
    for path in _recent_notes(vault / output_dir, end_date=end_date, days=days):
        for candidate in _note_candidates(path, vault=vault, states=states):
            if candidate["id"] in seen_ids:
                continue
            seen_ids.add(candidate["id"])
            candidates.append(candidate)
    return dedupe_similar_candidates(candidates)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _markdown(candidates: list[dict[str, Any]], *, end_date: dt.date, days: int) -> str:
    counts = {
        kind: sum(1 for item in candidates if item["candidate_type"] == kind)
        for kind in sorted(set(CANDIDATE_SECTIONS.values()))
    }
    lines = [
        f"# Auto Research Judgment Asset Candidates ({(end_date - dt.timedelta(days=days - 1)).isoformat()} to {end_date.isoformat()})",
        "",
        "## Summary",
        "",
        f"- Candidates: {len(candidates)}",
        f"- Deduped similar candidates: {sum(int(item.get('deduped_count') or 0) for item in candidates)}",
        f"- application_rule: {counts.get('application_rule', 0)}",
        f"- confirmation_question: {counts.get('confirmation_question', 0)}",
        f"- condition_signal: {counts.get('condition_signal', 0)}",
        f"- caution: {counts.get('caution', 0)}",
        f"- used: {sum(1 for item in candidates if int(item.get('use_count') or 0) > 0)}",
        f"- edited: {sum(1 for item in candidates if int(item.get('edit_count') or 0) > 0)}",
        f"- ready_for_promotion: {sum(1 for item in candidates if item.get('promotion_status') == 'ready_for_promotion')}",
        f"- rejected_or_deprioritized: {sum(1 for item in candidates if item.get('promotion_status') == 'rejected_or_deprioritized')}",
        f"- textbook_general: {sum(1 for item in candidates if item.get('asset_quality') == 'textbook_general')}",
        "",
        "## Promotion Policy",
        "",
        "- Auto Research is material, not memory.",
        "- Candidates stay `not_promoted` until a human uses them in a case and confirms they changed or improved judgment.",
        "- Do not promote textbook generalities. A candidate must change a case action, approval condition, rebuttal, or rejection reason.",
        "- Rule: `当たり前なこと言ってやった気になるな`.",
        "- Edited candidates are prioritized because a human has already shaped them into a usable judgment.",
        "- Weak notes that fail the substantive section gate are excluded.",
        "- `ready_for_promotion` requires useful human feedback and `verified_status=supported`.",
        "",
        "## Candidates",
        "",
    ]
    for item in candidates:
        lines += [
            f"### {item['research_date']} / {item['candidate_type']} / {item['research_topic']}",
            "",
            f"- Claim: {item['claim']}",
            f"- Edited claim: {item.get('edited_claim') or 'none'}",
            f"- Source section: {item['source_section']}",
            f"- Evidence: `{item['evidence_path']}`",
            f"- Status: {item['review_status']} / {item['promotion_status']}",
            f"- Asset quality: {item.get('asset_quality', 'actionable')} / reasons={', '.join(item.get('quality_reasons') or []) or 'none'}",
            f"- Metrics: use={item.get('use_count', 0)}, useful={item.get('useful_count', 0)}, rejected={item.get('rejected_count', 0)}, neutral={item.get('neutral_count', 0)}, verified={item.get('verified_status', 'unverified')}",
            f"- Deduped similar: {item.get('deduped_count', 0)}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(candidates: list[dict[str, Any]], *, end_date: dt.date, days: int, output_jsonl: Path) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_key = end_date.isoformat().replace("-", "")
    md_path = REPORTS_DIR / f"autoresearch_judgment_asset_candidates_{date_key}.md"
    latest_md = REPORTS_DIR / "autoresearch_judgment_asset_candidates_latest.md"
    summary_json = REPORTS_DIR / f"autoresearch_judgment_asset_candidates_{date_key}.json"
    latest_json = REPORTS_DIR / "autoresearch_judgment_asset_candidates_latest.json"
    md = _markdown(candidates, end_date=end_date, days=days)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "end_date": end_date.isoformat(),
        "days": days,
        "candidates": len(candidates),
        "deduped_similar_candidates": sum(int(item.get("deduped_count") or 0) for item in candidates),
        "counts": {
            kind: sum(1 for item in candidates if item["candidate_type"] == kind)
            for kind in sorted(set(CANDIDATE_SECTIONS.values()))
        },
        "used": sum(1 for item in candidates if int(item.get("use_count") or 0) > 0),
        "edited": sum(1 for item in candidates if int(item.get("edit_count") or 0) > 0),
        "ready_for_promotion": sum(1 for item in candidates if item.get("promotion_status") == "ready_for_promotion"),
        "rejected_or_deprioritized": sum(1 for item in candidates if item.get("promotion_status") == "rejected_or_deprioritized"),
        "textbook_general": sum(1 for item in candidates if item.get("asset_quality") == "textbook_general"),
        "output_jsonl": str(output_jsonl),
        "promotion_policy": "human_use_feedback_result_verification_and_non_textbook_actionability_required",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "markdown": str(md_path),
        "latest_markdown": str(latest_md),
        "summary_json": str(summary_json),
        "latest_summary_json": str(latest_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build review-only judgment asset candidates from Auto Research notes.")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--vault", default="", help="Obsidian Vault path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--state", default=str(DEFAULT_STATE_JSON), help="Candidate state JSON path")
    args = parser.parse_args()

    end_date = dt.date.fromisoformat(args.date)
    vault = Path(args.vault).expanduser() if args.vault else _vault_path()
    days = max(1, args.days)
    output_path = Path(args.output)
    state_path = Path(args.state)
    candidates = extract_candidates(
        vault=vault,
        output_dir=args.output_dir.strip("/") or DEFAULT_OUTPUT_DIR,
        end_date=end_date,
        days=days,
        state_path=state_path,
    )
    write_state(state_path, candidates, load_state(state_path))
    write_jsonl(output_path, candidates)
    paths = write_report(candidates, end_date=end_date, days=days, output_jsonl=output_path)
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "output_jsonl": str(output_path),
                "state": str(state_path),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
