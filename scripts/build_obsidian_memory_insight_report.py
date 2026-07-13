#!/usr/bin/env python3
"""Build a read-only Obsidian memory insight report.

This script inspects selected Obsidian note surfaces and creates sidecar
artifacts only:

- reports/obsidian_memory_insight_latest.md
- data/obsidian_memory_insight_candidates.jsonl

It does not write to Obsidian, RAG, prompts, scoring, Cloud Run, or memory
stores. The first purpose is quality inspection: identify which notes can become
usable memory, which are still raw material, and what deeper reasoning cards
should be reviewed by a human.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "data" / "obsidian_memory_insight_candidates.jsonl"
DEFAULT_REPORT = REPO_ROOT / "reports" / "obsidian_memory_insight_latest.md"

SOURCE_DIRS = {
    "daily": Path("Daily"),
    "private_reflection": Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Private Reflection",
    "dialogue": Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Dialogue",
    "cloudrun_conversation": Path("Projects") / "tune_lease_55" / "AI Chat" / "Cloud Run Conversation Log",
    "ai_chat": Path("Projects") / "tune_lease_55" / "AI Chat",
    "research": Path("Projects") / "tune_lease_55" / "Research",
}

CANDIDATE_TYPES = (
    "user_preference",
    "judgment_rule",
    "reflection_update",
    "research_material",
    "noise",
)

USER_PREFERENCE_TERMS = (
    "してほしい",
    "しないで",
    "Userは",
    "ユーザーは",
    "ユーザーが",
    "優先",
    "短く",
    "長く",
    "お願い",
    "壊さない",
    "デプロイはなし",
)
JUDGMENT_TERMS = (
    "審査",
    "判断",
    "承認",
    "否決",
    "条件",
    "確認",
    "返済",
    "競合",
    "成約",
    "信用",
    "物件",
    "リース",
    "稟議",
    "Q_risk",
    "Qrisk",
    "AURION",
)
REFLECTION_TERMS = (
    "内省",
    "見落とし",
    "すり替え",
    "誤読",
    "逃げ",
    "次回",
    "仮説",
    "更新",
    "信念",
    "Private Reflection",
)
RESEARCH_TERMS = (
    "Research",
    "調査",
    "出典",
    "統計",
    "業界",
    "市場",
    "要確認",
    "参考",
    "Auto Research",
)
NOISE_TERMS = (
    "wrote=",
    "report=",
    "pytest",
    "passed",
    "npm run",
    "python -m",
    "git ",
    "commit:",
    "source=",
    "http://",
    "https://",
    "node_modules",
    "以下の",
    "ご質問",
    "できます",
    "してください",
    "ありがとうございます",
    "重要な情報です",
    "お手伝い",
)
DEEP_REASONING_TERMS = (
    "前提",
    "破られ",
    "責任",
    "仮説",
    "更新",
    "検証",
    "判断が変わる",
    "次に",
    "User",
    "ユーザー",
    "望んだ",
)
REDACTIONS = (
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
    (re.compile(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b"), "[phone]"),
    (re.compile(r"\b\d{7,}\b"), "[number]"),
)


@dataclass(frozen=True)
class SourceNote:
    surface: str
    path: Path
    rel_path: str
    date_hint: str
    text: str


def _vault_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH")
    return Path(raw).expanduser() if raw else DEFAULT_VAULT


def _date_range(end_date: date, days: int) -> set[str]:
    return {(end_date - timedelta(days=offset)).isoformat() for offset in range(days)}


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)


def _read_text(path: Path, max_chars: int = 50000) -> str:
    try:
        return _strip_frontmatter(path.read_text(encoding="utf-8", errors="ignore")).strip()[:max_chars]
    except OSError:
        return ""


def _date_hint(path: Path, text: str) -> str:
    for value in (path.stem, path.name, text[:500]):
        match = re.search(r"(20\d{2}-\d{2}-\d{2})", value)
        if match:
            return match.group(1)
        compact = re.search(r"(20\d{6})", value)
        if compact:
            raw = compact.group(1)
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return ""


def load_notes(vault: Path, *, end_date: date, days: int, max_files_per_surface: int = 40) -> list[SourceNote]:
    wanted_dates = _date_range(end_date, days)
    notes: list[SourceNote] = []
    for surface, rel_dir in SOURCE_DIRS.items():
        root = vault / rel_dir
        if not root.exists():
            continue
        files = sorted(root.glob("*.md"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        accepted = 0
        for path in files:
            text = _read_text(path)
            if not text:
                continue
            hint = _date_hint(path, text)
            if hint and hint not in wanted_dates:
                continue
            try:
                rel_path = str(path.relative_to(vault))
            except ValueError:
                rel_path = str(path)
            notes.append(SourceNote(surface=surface, path=path, rel_path=rel_path, date_hint=hint, text=text))
            accepted += 1
            if accepted >= max_files_per_surface:
                break
    return notes


def _clean_text(value: str, limit: int = 260) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    for pattern, repl in REDACTIONS:
        text = pattern.sub(repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\n+", "。", text)
    parts = re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+", text)
    result: list[str] = []
    for part in parts:
        clean = _clean_text(part)
        if 14 <= len(clean) <= 260:
            result.append(clean)
    return result


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _candidate_type(sentence: str, surface: str) -> str:
    if _contains_any(sentence, NOISE_TERMS):
        return "noise"
    if _contains_any(sentence, RESEARCH_TERMS) or surface == "research":
        return "research_material"
    if _contains_any(sentence, USER_PREFERENCE_TERMS):
        return "user_preference"
    if _contains_any(sentence, REFLECTION_TERMS) or surface == "private_reflection":
        return "reflection_update"
    if _contains_any(sentence, JUDGMENT_TERMS):
        return "judgment_rule"
    return "noise"


def _candidate_quality(sentence: str, ctype: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if ctype == "noise":
        return "noise", ["technical_or_low_context"]
    if _looks_vague_or_assistantish(sentence):
        return "review", ["vague_or_assistantish"]
    if len(sentence) < 24:
        reasons.append("too_short")
    if ctype == "judgment_rule" and not any(term in sentence for term in ("確認", "条件", "判断", "変わる", "否決", "承認", "見る")):
        reasons.append("judgment_action_missing")
    if ctype == "reflection_update" and not any(term in sentence for term in ("次回", "変える", "禁止", "更新", "検証", "望んだ", "すり替え")):
        reasons.append("next_behavior_missing")
    if ctype == "research_material" and "要確認" in sentence and not any(term in sentence for term in ("根拠", "出典", "条件", "警戒", "反証")):
        reasons.append("thin_research")
    if reasons:
        return "review", reasons
    return "useful_candidate", ["has_operational_signal"]


def _looks_vague_or_assistantish(sentence: str) -> bool:
    vague = (
        "この点",
        "以下の",
        "何か",
        "重要な",
        "具体的な論点",
        "少し偉い",
        "できます",
        "してください",
        "ご確認",
        "お勧め",
    )
    if any(term in sentence for term in vague):
        return True
    if len(sentence) < 30 and not any(term in sentence for term in ("User", "ユーザー", "判断", "内省", "審査")):
        return True
    return False


def _candidate_id(sentence: str, rel_path: str, ctype: str) -> str:
    raw = f"{ctype}|{rel_path}|{sentence}"
    return "omi_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def collect_candidates(notes: list[SourceNote], *, limit_per_type: int = 30) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_meaning: set[tuple[str, str]] = set()
    for note in notes:
        for sentence in _sentences(note.text):
            ctype = _candidate_type(sentence, note.surface)
            quality, reasons = _candidate_quality(sentence, ctype)
            key = (ctype, _meaning_key(sentence))
            if key in seen_meaning:
                continue
            seen_meaning.add(key)
            item = {
                "candidate_id": _candidate_id(sentence, note.rel_path, ctype),
                "candidate_type": ctype,
                "quality": quality,
                "quality_reasons": reasons,
                "claim": sentence,
                "surface": note.surface,
                "source_path": note.rel_path,
                "date_hint": note.date_hint,
                "promote_status": "not_promoted_inspection_only",
            }
            buckets[ctype].append(item)

    ordered: list[dict[str, Any]] = []
    for ctype in CANDIDATE_TYPES:
        typed = buckets.get(ctype, [])
        typed.sort(key=lambda item: (item["quality"] != "useful_candidate", len(item["claim"])))
        ordered.extend(typed[:limit_per_type])
    return ordered


def _meaning_key(sentence: str) -> str:
    normalized = re.sub(r"[`*_#\[\]().,、。:：;；\s]+", "", sentence.lower())
    normalized = re.sub(r"\d+", "0", normalized)
    return normalized[:80]


def build_thinking_cards(candidates: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    priority_types = {"reflection_update", "judgment_rule", "user_preference"}
    eligible = [
        item
        for item in candidates
        if item["candidate_type"] in priority_types
        and item["quality"] != "noise"
        and _contains_any(item["claim"], DEEP_REASONING_TERMS + JUDGMENT_TERMS + USER_PREFERENCE_TERMS)
    ]
    for item in eligible:
        claim = item["claim"]
        card = {
            "card_id": "card_" + hashlib.sha1((item["candidate_id"] + claim).encode("utf-8")).hexdigest()[:10],
            "source_candidate_id": item["candidate_id"],
            "source_path": item["source_path"],
            "surface": item["surface"],
            "premise": _derive_premise(claim),
            "broken_or_risk": _derive_broken_or_risk(claim),
            "judgment_change": _derive_judgment_change(claim, item["candidate_type"]),
            "user_check": _derive_user_check(claim, item["candidate_type"]),
            "shion_next_action": _derive_shion_action(claim, item["candidate_type"]),
        }
        cards.append(card)
        if len(cards) >= limit:
            break
    return cards


def _derive_premise(claim: str) -> str:
    if "すり替え" in claim or "誤読" in claim:
        return "紫苑は、Userの要求をそのまま扱えているという前提に立ちがちだった。"
    if "判断資産" in claim:
        return "判断資産候補は、文面として保存すれば再利用に近づくという前提があった。"
    if "デプロイ" in claim or "壊さない" in claim:
        return "改善は本番環境へ接続して初めて価値が出るという前提を置きがちだった。"
    return "このノートは、現在の運用前提や判断材料として再利用できる可能性がある。"


def _derive_broken_or_risk(claim: str) -> str:
    if "壊さない" in claim or "デプロイはなし" in claim:
        return "ハッカソン中は、便利な接続でも環境を壊すリスクが価値を上回る。"
    if "退屈" in claim or "内省" in claim:
        return "内省らしい文章があっても、User要求や次回行動へ戻らなければ意味が薄い。"
    if "判断資産" in claim:
        return "候補が一般論のままだと、案件の確認行動や承認条件を変えない。"
    return "材料のままでは、次の判断や質問にどう効くかが不明なまま残る。"


def _derive_judgment_change(claim: str, ctype: str) -> str:
    if ctype == "user_preference":
        return "次回の提案は、Userの制約を先に固定してから深さや接続を決める。"
    if ctype == "reflection_update":
        return "次回の内省では、要求、誤読、次回変更をセットで確認する。"
    if ctype == "judgment_rule":
        return "案件レビューでは、この論点が承認条件・否決理由・追加確認のどれに当たるかを分ける。"
    return "まず材料扱いに留め、実案件で有用性を確認する。"


def _derive_user_check(claim: str, ctype: str) -> str:
    if ctype == "judgment_rule":
        return "この論点が現場で本当に確認すべき項目か、採用・修正・却下で確認してもらう。"
    if ctype == "reflection_update":
        return "この内省が実際に次回行動へ変わる内容かだけ確認してもらう。"
    if ctype == "user_preference":
        return "この制約・好みを今後の既定方針にしてよいか確認してもらう。"
    return "この材料を判断資産候補へ進める価値があるか確認してもらう。"


def _derive_shion_action(claim: str, ctype: str) -> str:
    if ctype == "user_preference":
        return "User制約に反する接続・自動化・デプロイを提案前に除外する。"
    if ctype == "reflection_update":
        return "深そうな文章より、次に禁止する癖と増やす行動を先に書く。"
    if ctype == "judgment_rule":
        return "今回案件へ応用する時は、元の文を丸写しせず確認行動に変換する。"
    return "RAGや記憶へ入れず、材料として隔離する。"


def build_report(vault: Path, notes: list[SourceNote], candidates: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(item["candidate_type"] for item in candidates)
    quality_counts = Counter(item["quality"] for item in candidates)
    surface_counts = Counter(note.surface for note in notes)
    noise_samples = [item for item in candidates if item["candidate_type"] == "noise"][:8]
    useful = [item for item in candidates if item["quality"] == "useful_candidate"][:20]
    review = [item for item in candidates if item["quality"] == "review"][:20]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "vault": str(vault),
        "source_note_count": len(notes),
        "surface_counts": dict(surface_counts),
        "candidate_counts": dict(type_counts),
        "quality_counts": dict(quality_counts),
        "useful_candidates": useful,
        "review_candidates": review,
        "noise_samples": noise_samples,
        "thinking_cards": cards,
        "guardrail": "inspection_only_no_rag_no_prompt_no_cloudrun_no_obsidian_write",
    }


def write_jsonl(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in candidates]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Obsidian Memory Insight Report",
        "",
        "## Scope",
        f"- generated_at: `{report['generated_at']}`",
        f"- source notes: `{report['source_note_count']}`",
        f"- guardrail: `{report['guardrail']}`",
        "",
        "## Surface Counts",
        *_counter_lines(report.get("surface_counts") or {}),
        "",
        "## Candidate Counts",
        *_counter_lines(report.get("candidate_counts") or {}),
        "",
        "## Quality Counts",
        *_counter_lines(report.get("quality_counts") or {}),
        "",
        "## Useful Memory Candidates",
        *_candidate_lines(report.get("useful_candidates") or []),
        "",
        "## Review Needed",
        *_candidate_lines(report.get("review_candidates") or []),
        "",
        "## Deep Reasoning Cards",
    ]
    cards = report.get("thinking_cards") or []
    if not cards:
        lines.append("- なし")
    for card in cards:
        lines.extend(
            [
                f"### {card['card_id']}",
                f"- source: `{card['source_path']}` / `{card['source_candidate_id']}`",
                f"- premise: {card['premise']}",
                f"- broken_or_risk: {card['broken_or_risk']}",
                f"- judgment_change: {card['judgment_change']}",
                f"- user_check: {card['user_check']}",
                f"- shion_next_action: {card['shion_next_action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Noise Samples",
            *_candidate_lines(report.get("noise_samples") or []),
            "",
            "## Next Safe Step",
            "- 3日ほど読み取り専用で見て、Useful候補とDeep Reasoning Cardが本当に判断・質問・内省を変えるか確認する。",
            "- まだRAG、チャットプロンプト、スコアリング、Cloud Run、Obsidian本文更新には接続しない。",
            "",
        ]
    )
    return "\n".join(lines)


def _counter_lines(counter: dict[str, int]) -> list[str]:
    if not counter:
        return ["- なし"]
    return [f"- {key}: `{value}`" for key, value in sorted(counter.items())]


def _candidate_lines(items: list[dict[str, Any]], limit: int = 12) -> list[str]:
    if not items:
        return ["- なし"]
    lines = []
    for item in items[:limit]:
        lines.append(
            f"- `{item.get('candidate_type')}` `{item.get('quality')}` "
            f"[{item.get('candidate_id')}] {item.get('claim')} "
            f"(source: `{item.get('source_path')}`)"
        )
    return lines


def _parse_date(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build read-only Obsidian memory insight report.")
    parser.add_argument("--vault", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vault = _vault_path(args.vault)
    notes = load_notes(vault, end_date=_parse_date(args.date), days=max(1, args.days))
    candidates = collect_candidates(notes)
    cards = build_thinking_cards(candidates)
    report = build_report(vault, notes, candidates, cards)

    if args.dry_run:
        print(render_markdown(report))
        return 0

    write_jsonl(args.output_jsonl, candidates)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(report), encoding="utf-8")
    print(f"candidates={args.output_jsonl}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
