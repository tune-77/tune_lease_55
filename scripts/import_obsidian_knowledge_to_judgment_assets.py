#!/usr/bin/env python3
"""Import actionable Obsidian knowledge notes into judgment asset candidates.

The importer is intentionally conservative:
- it reads only local Obsidian markdown;
- it writes review-only candidates to data/autoresearch_judgment_asset_candidates.jsonl;
- it does not promote candidates into active canonical rules;
- every imported item still requires field feedback before it should be trusted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "autoresearch_judgment_asset_candidates.jsonl"
DEFAULT_STATE_JSON = DATA_DIR / "autoresearch_judgment_asset_candidate_state.json"
DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
DEFAULT_SOURCE_DIRS = (
    Path("Projects") / "tune_lease_55" / "Asset Knowledge",
    Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Knowledge",
)

ACTIONABLE_MARKERS = (
    "確認",
    "見る",
    "重視",
    "条件",
    "稟議",
    "残価",
    "再販",
    "換金",
    "保守",
    "点検",
    "契約",
    "撤退",
    "控除",
    "説明",
    "リスク",
    "根拠",
    "優先",
    "前提",
)
GENERIC_MARKERS = (
    "総合的に判断",
    "慎重に判断",
    "必要がある",
    "重要です",
    "できます",
    "してください",
    "お問い合わせ",
)
HARD_REJECT_MARKERS = (
    "選定理由:",
    "抽出起点:",
    "structurednote",
    "reusableterms",
    "[ ]",
    "個別の相談",
    "相談を希望",
    "お問い合わせ",
    "テンプレート",
)
USEFUL_HEADINGS = (
    "Summary",
    "稟議で使えそうな根拠",
    "注意すべき物件特性",
    "残価・再販リスク",
    "中古相場・再販観点",
    "参照優先順位",
    "リース審査への適用",
    "担当者が確認する質問",
    "承認条件を変える兆候",
    "反証・過信してはいけない点",
)


def _vault_path() -> Path:
    raw = os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or str(DEFAULT_VAULT)
    return Path(raw).expanduser()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)


def _clean_item(value: str, limit: int = 280) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _sections(text: str) -> list[tuple[str, str]]:
    stripped = _strip_frontmatter(text)
    matches = list(re.finditer(r"^#{2,4}\s+(.+?)\s*$", stripped, flags=re.MULTILINE))
    if not matches:
        return [("本文", stripped)]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        sections.append((heading, stripped[start:end].strip()))
    return sections


def _candidate_type(heading: str, claim: str) -> str:
    joined = f"{heading} {claim}"
    if any(marker in joined for marker in ("注意", "リスク", "過信", "反証", "減点", "控除")):
        return "caution"
    if any(marker in joined for marker in ("条件", "兆候", "サイン", "前提")):
        return "condition_signal"
    if any(marker in joined for marker in ("確認", "質問", "見る", "点検")):
        return "confirmation_question"
    return "application_rule"


def _quality(claim: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if len(claim) < 18:
        reasons.append("too_short")
    if any(marker in claim for marker in HARD_REJECT_MARKERS):
        reasons.append("template_or_metadata")
    if not any(marker in claim for marker in ACTIONABLE_MARKERS):
        reasons.append("no_actionable_marker")
    if any(marker in claim for marker in GENERIC_MARKERS) and not any(marker in claim for marker in ("稟議", "残価", "再販", "撤退", "保守", "点検")):
        reasons.append("textbook_general")
    return ("actionable" if not reasons else "needs_review"), reasons


def _items_from_section(heading: str, content: str) -> list[str]:
    items: list[str] = []
    for line in content.splitlines():
        if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+[.)]\s+", line):
            cleaned = _clean_item(line)
            if cleaned:
                items.append(cleaned)
    if not items and heading == "Summary":
        for sentence in re.split(r"(?<=[。！？!?])\s*", content):
            cleaned = _clean_item(sentence)
            if cleaned:
                items.append(cleaned)
    return items


def extract_candidates_from_note(path: Path, vault: Path) -> list[dict[str, Any]]:
    if path.name == "Promoted Knowledge.md":
        return []
    text = _read_text(path)
    if not text:
        return []
    rel_path = path.relative_to(vault).as_posix()
    title_match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    rows: list[dict[str, Any]] = []
    for heading, content in _sections(text):
        if any(marker in heading for marker in ("テンプレート", "文面", "意思確認状")):
            continue
        if heading not in USEFUL_HEADINGS and not any(key in heading for key in ("稟議", "注意", "リスク", "確認", "条件")):
            continue
        for item in _items_from_section(heading, content):
            claim = _clean_item(item)
            asset_quality, quality_reasons = _quality(claim)
            if asset_quality != "actionable":
                continue
            seed = f"obsidian_knowledge|{rel_path}|{heading}|{claim}"
            candidate_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "id": candidate_id,
                    "candidate_type": _candidate_type(heading, claim),
                    "research_topic": "obsidian_knowledge",
                    "research_title": title,
                    "research_date": dt.date.today().isoformat(),
                    "claim": claim,
                    "effective_claim": claim,
                    "edited_claim": "",
                    "edit_count": 0,
                    "last_edited_at": "",
                    "source_section": heading,
                    "evidence_path": rel_path,
                    "review_status": "candidate",
                    "asset_quality": asset_quality,
                    "quality_reasons": quality_reasons,
                    "promotion_status": "not_promoted",
                    "use_count": 0,
                    "useful_count": 0,
                    "rejected_count": 0,
                    "neutral_count": 0,
                    "last_used_at": "",
                    "last_feedback_at": "",
                    "verified_status": "unverified",
                    "verification_note": "imported_from_obsidian_knowledge",
                    "requires_human_use_feedback": True,
                    "requires_result_verification": True,
                    "import_source": "obsidian_knowledge",
                    "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "use_policy": "Obsidian Knowledge由来の判断資産候補。案件レビューで使い、効いた/外したと結果検証してから昇格する。",
                }
            )
    return rows


def _existing_keys(path: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    claims: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ids, claims
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            if item.get("id"):
                ids.add(str(item["id"]))
            claim = _clean_item(str(item.get("edited_claim") or item.get("claim") or ""))
            if claim:
                claims.add(claim)
    return ids, claims


def _drop_existing_import_source(path: Path, import_source: str) -> int:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return 0
    kept: list[str] = []
    removed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if isinstance(item, dict) and str(item.get("import_source") or "") == import_source:
            removed += 1
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def import_knowledge_candidates(
    *,
    vault: Path,
    source_dirs: list[Path],
    output_jsonl: Path,
    replace_existing_import: bool = False,
) -> dict[str, Any]:
    removed_existing = _drop_existing_import_source(output_jsonl, "obsidian_knowledge") if replace_existing_import else 0
    existing_ids, existing_claims = _existing_keys(output_jsonl)
    scanned_notes = 0
    extracted = 0
    imported: list[dict[str, Any]] = []
    for rel_dir in source_dirs:
        root = vault / rel_dir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            scanned_notes += 1
            rows = extract_candidates_from_note(path, vault)
            extracted += len(rows)
            for row in rows:
                claim = _clean_item(str(row.get("claim") or ""))
                candidate_id = str(row.get("id") or "")
                if candidate_id in existing_ids or claim in existing_claims:
                    continue
                imported.append(row)
                existing_ids.add(candidate_id)
                existing_claims.add(claim)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if imported:
        with output_jsonl.open("a", encoding="utf-8") as f:
            for row in imported:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "scanned_notes": scanned_notes,
        "extracted": extracted,
        "imported": len(imported),
        "removed_existing": removed_existing,
        "imported_items": imported,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Obsidian Knowledge → Judgment Assets Import",
        "",
        f"- Scanned notes: {summary['scanned_notes']}",
        f"- Extracted candidates: {summary['extracted']}",
        f"- Removed existing obsidian_knowledge candidates: {summary.get('removed_existing', 0)}",
        f"- Imported new candidates: {summary['imported']}",
        "",
        "## Imported",
        "",
    ]
    items = summary.get("imported_items") or []
    if not items:
        lines.append("- No new candidates.")
    for item in items[:80]:
        lines.extend(
            [
                f"### {item['id']} / {item['candidate_type']}",
                f"- Claim: {item['claim']}",
                f"- Source: `{item['evidence_path']}` / {item['source_section']}",
                "",
            ]
        )
    if len(items) > 80:
        lines.append(f"- ...and {len(items) - 80} more")
    lines.append("")
    lines.append("All imported items remain `not_promoted` and require field feedback before active use.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=_vault_path())
    parser.add_argument("--source-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--report-md", type=Path, default=REPORTS_DIR / "obsidian_knowledge_judgment_asset_import_latest.md")
    parser.add_argument("--replace-existing-import", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dirs = args.source_dir or list(DEFAULT_SOURCE_DIRS)
    summary = import_knowledge_candidates(
        vault=args.vault.expanduser(),
        source_dirs=source_dirs,
        output_jsonl=args.output_jsonl,
        replace_existing_import=args.replace_existing_import,
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        "Obsidian Knowledge import: "
        f"scanned={summary['scanned_notes']} extracted={summary['extracted']} imported={summary['imported']}"
    )
    print(f"report: {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
