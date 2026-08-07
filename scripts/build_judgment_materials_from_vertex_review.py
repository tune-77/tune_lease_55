#!/usr/bin/env python3
"""Bridge Vertex Distilled notes awaiting human review into judgment materials preview.

This script is intentionally read-only with respect to Obsidian and sidecar-only
with respect to the app. It reads Research/Vertex Distilled notes whose frontmatter
declares review_status: needs_human_review and writes preview artifacts only:

- data/judgment_materials_preview_vertex_review.jsonl
- reports/judgment_materials_preview_vertex_review_latest.md

It does not connect to RAG, scoring, chat prompts, or Obsidian sync. It does not
touch Research/Auto Research notes, which have their own separate, stricter
promotion pipeline (build_autoresearch_judgment_asset_candidates.py).

Materials produced here are tagged source="vertex_distilled_review" and
source_role="vertex_review" (never "user"), so build_canonical_judgment_rules.py
can avoid auto-promoting concept groups that are backed only by Vertex-derived
claims without genuine user-confirmed evidence.
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime_paths import resolve_obsidian_vault  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "judgment_materials_preview_vertex_review.jsonl"
DEFAULT_VAULT = resolve_obsidian_vault()
SOURCE_DIR = Path("Research") / "Vertex Distilled"

SOURCE_LABEL = "vertex_distilled_review"
SOURCE_ROLE = "vertex_review"
MAX_CLAIMS_PER_NOTE = 10
RISK_HEADING_TERMS = ("反証", "リスク", "兆候")

FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
FIELD_PATTERN = re.compile(r'^([A-Za-z_]+):\s*"?([^"\n]*)"?\s*$', re.MULTILINE)
SECTION_PATTERN = re.compile(r"##\s*Answer Summary\s*\n(.*?)(?=\n##\s|\Z)", re.DOTALL)
SEGMENT_SPLIT_PATTERN = re.compile(r"#{2,6}\s*[^\n*]*|\*\s+")


def _vault_path() -> Path:
    raw = os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or str(DEFAULT_VAULT)
    return Path(raw).expanduser()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _clean_text(value: str, limit: int = 240) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*[-*+]\s+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for key, value in FIELD_PATTERN.findall(match.group(1)):
        fields[key] = value.strip()
    return fields


def _material_id(path_label: str, claim: str) -> str:
    raw = f"{SOURCE_LABEL}|{path_label}|{claim}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _material_type_for_heading(heading: str) -> str:
    if any(term in heading for term in RISK_HEADING_TERMS):
        return "risk_signal"
    return "judgment_rule"


def _extract_claims(body: str) -> list[tuple[str, str]]:
    """Split the dense Answer Summary body into (heading_label, claim) pairs."""
    match = SECTION_PATTERN.search(body)
    if not match:
        return []
    section = match.group(1)

    markers = list(SEGMENT_SPLIT_PATTERN.finditer(section))
    if not markers:
        cleaned = _clean_text(section)
        return [("", cleaned)] if 20 <= len(cleaned) <= 240 else []

    claims: list[tuple[str, str]] = []
    current_heading = ""
    leading = _clean_text(section[: markers[0].start()])
    if 20 <= len(leading) <= 240:
        claims.append((current_heading, leading))
    for idx, marker in enumerate(markers):
        marker_text = marker.group(0).strip()
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(section)
        segment = section[start:end]
        cleaned = _clean_text(segment)
        if marker_text.startswith("#"):
            current_heading = re.sub(r"^#+\s*", "", marker_text).strip()
            continue
        if 20 <= len(cleaned) <= 240:
            claims.append((current_heading, cleaned))
    return claims


def _confidence_from_grounding_score(raw: str) -> float:
    try:
        score = float(raw)
    except (TypeError, ValueError):
        score = 0.5
    return round(min(0.95, max(0.3, score)), 2)


def _domain(_claim: str) -> str:
    return "lease_screening"


def extract_materials(*, vault: Path) -> list[dict[str, Any]]:
    materials: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()
    source_dir = vault / SOURCE_DIR
    if not source_dir.exists():
        return materials

    for path in sorted(source_dir.glob("*.md")):
        text = _read_text(path)
        frontmatter = _parse_frontmatter(text)
        if frontmatter.get("review_status") != "needs_human_review":
            continue

        rel_path = path.relative_to(vault) if str(path).startswith(str(vault)) else path
        note_date = frontmatter.get("date") or dt.date.today().isoformat()
        topic = frontmatter.get("topic") or ""
        workflow_mode = frontmatter.get("workflow_mode") or ""
        confidence = _confidence_from_grounding_score(frontmatter.get("grounding_score", ""))

        claims = _extract_claims(text)[:MAX_CLAIMS_PER_NOTE]
        for heading, claim in claims:
            dedupe_key = (str(rel_path), claim)
            if dedupe_key in seen_claims:
                continue
            seen_claims.add(dedupe_key)
            material_type = _material_type_for_heading(heading)
            materials.append(
                {
                    "id": _material_id(str(rel_path), claim),
                    "date": note_date,
                    "source": SOURCE_LABEL,
                    "source_role": SOURCE_ROLE,
                    "material_type": material_type,
                    "domain": _domain(claim),
                    "claim": claim,
                    "use_when": "Vertex検索で得た根拠・判断候補を人がレビューして採否を判断するとき",
                    "risk_axis": [],
                    "confidence": confidence,
                    "evidence_path": str(rel_path),
                    "vertex_workflow_mode": workflow_mode,
                    "vertex_topic": topic,
                    "private": False,
                    "preview": True,
                }
            )
    return materials


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _markdown(materials: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item in materials:
        counts[item["material_type"]] = counts.get(item["material_type"], 0) + 1
    lines = [
        "# Judgment Materials Preview (Vertex Distilled / needs_human_review)",
        "",
        "## Summary",
        "",
        f"- Materials: {len(materials)}",
        f"- judgment_rule: {counts.get('judgment_rule', 0)}",
        f"- risk_signal: {counts.get('risk_signal', 0)}",
        "",
        "## Safety",
        "",
        "- Preview only. Not connected to RAG, chat prompts, scoring, or Obsidian sync.",
        "- Source: Research/Vertex Distilled notes with review_status: needs_human_review only.",
        "- Research/Auto Research notes are intentionally excluded (separate promotion pipeline).",
        "- source_role is always vertex_review, never user — build_canonical_judgment_rules.py "
        "must not auto-promote concept groups backed only by this source.",
        "",
        "## Materials",
        "",
    ]
    for item in materials:
        lines += [
            f"### {item['date']} / {item['material_type']} / confidence={item['confidence']}",
            "",
            f"- Claim: {item['claim']}",
            f"- Topic: {item.get('vertex_topic', '')}",
            f"- Workflow mode: {item.get('vertex_workflow_mode', '')}",
            f"- Evidence: `{item['evidence_path']}`",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(materials: list[dict[str, Any]]) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_md = REPORTS_DIR / "judgment_materials_preview_vertex_review_latest.md"
    md = _markdown(materials)
    latest_md.write_text(md, encoding="utf-8")
    return {"latest_markdown": str(latest_md)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bridge Vertex Distilled needs_human_review notes into judgment materials preview"
    )
    parser.add_argument("--vault", default="", help="Obsidian Vault path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSONL), help="JSONL output path")
    args = parser.parse_args()

    vault = Path(args.vault).expanduser() if args.vault else _vault_path()
    materials = extract_materials(vault=vault)
    output_path = Path(args.output)
    write_jsonl(output_path, materials)
    paths = write_report(materials)
    print(
        json.dumps(
            {
                "materials": len(materials),
                "output_jsonl": str(output_path),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
