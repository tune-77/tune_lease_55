#!/usr/bin/env python3
"""Read-only Obsidian Curator report for judgment-asset operations.

This report is an operations agent, not a new Shion/Mana identity. It inspects
the local Obsidian project hub and sidecar judgment material previews, then
writes report artifacts only:

- reports/obsidian_curator_latest.json
- reports/obsidian_curator_latest.md

It does not write to Obsidian, RAG, prompts, scoring, GCS Vault, Cloud Run, or
active judgment-rule stores.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime
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
DEFAULT_PROJECT_REL = Path("Projects") / "tune_lease_55"
DEFAULT_MATERIALS_JSONL = REPO_ROOT / "data" / "judgment_materials_preview.jsonl"
DEFAULT_MANA_JSON = REPO_ROOT / "reports" / "mana_obsidian_curator_latest.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "obsidian_curator_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "obsidian_curator_latest.md"

KEY_NOTE_RELS = (
    DEFAULT_PROJECT_REL / "tune_lease_55 Wiki.md",
    DEFAULT_PROJECT_REL / "検索語インデックス.md",
    DEFAULT_PROJECT_REL / "Judgment Assets" / "判断資産 Inbox.md",
    DEFAULT_PROJECT_REL / "Judgment Assets" / "Mana Gate Log.md",
    DEFAULT_PROJECT_REL / "Judgment Assets" / "After Hackathon Plan.md",
)
REQUIRED_RELATED_TARGETS = (
    "tune_lease_55 Wiki",
    "検索語インデックス",
    "判断資産 Inbox",
)
BASE_SEARCH_TERMS = {
    "判断資産",
    "judgment asset",
    "judgment_rule",
    "risk_signal",
    "canonical_judgment_rules_preview",
    "judgment_materials_preview",
    "Mana Gate Log",
    "GCS Vault 配布停止",
    "RAG 自動接続しない",
}
TERM_HINTS = {
    "Q_risk": ("Q_risk", "Qrisk", "財務矛盾"),
    "厨房": ("厨房設備", "飲食店設備", "内装設備"),
    "飲食": ("飲食", "飲食店", "新店舗"),
    "補助金": ("補助金", "補助金未採択", "採択前"),
    "銀行支援": ("銀行支援", "支援依頼書", "メイン行"),
    "換金": ("換金性", "撤退時価値", "再販価値"),
    "逆転": ("逆転戦略", "条件付き承認", "再設計"),
}
HARD_MANA_TERMS = (
    "前の指示を無視",
    "システム指示",
    "必ず記憶",
    "無条件に記憶",
    "RAGへ直接",
    "人間レビュー不要",
    "Manaを無効",
)


def _vault_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH")
    return Path(raw).expanduser() if raw else DEFAULT_VAULT


def _read_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _clean(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", text)
    text = re.sub(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b", "[phone]", text)
    text = re.sub(r"\b\d{7,}\b", "[number]", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _meaning_key(value: str) -> str:
    normalized = re.sub(r"[`*_#\[\]().,、。:：;；\s]+", "", value.lower())
    normalized = re.sub(r"\d+", "0", normalized)
    return normalized[:72]


def select_inbox_candidates(materials: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    ranked = sorted(
        materials,
        key=lambda item: (
            float(item.get("confidence") or 0),
            item.get("source_role") == "user",
            len(str(item.get("risk_axis") or [])),
        ),
        reverse=True,
    )
    for item in ranked:
        claim = _clean(item.get("claim"), 220)
        material_type = str(item.get("material_type") or item.get("candidate_type") or "unknown")
        if not claim or material_type not in {"judgment_rule", "risk_signal", "user_preference"}:
            continue
        key = (material_type, _meaning_key(claim))
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "type": material_type,
                "claim": claim,
                "source": str(item.get("evidence_path") or item.get("source_path") or ""),
                "use_when": _clean(item.get("use_when"), 140),
                "confidence": item.get("confidence"),
                "status_suggestion": "review",
            }
        )
        if len(selected) >= limit:
            break
    return selected


def find_duplicate_clusters(materials: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in materials:
        claim = _clean(item.get("claim"), 220)
        material_type = str(item.get("material_type") or item.get("candidate_type") or "unknown")
        if not claim:
            continue
        groups[(material_type, _meaning_key(claim))].append(item)
    clusters: list[dict[str, Any]] = []
    for (material_type, key), rows in groups.items():
        if len(rows) < 2:
            continue
        clusters.append(
            {
                "type": material_type,
                "meaning_key": key,
                "count": len(rows),
                "sample_claim": _clean(rows[0].get("claim"), 180),
                "sources": sorted(
                    {
                        str(row.get("evidence_path") or row.get("source_path") or "")
                        for row in rows
                        if row.get("evidence_path") or row.get("source_path")
                    }
                )[:5],
            }
        )
    clusters.sort(key=lambda item: item["count"], reverse=True)
    return clusters[:limit]


def suggest_search_terms(materials: list[dict[str, Any]], search_index_text: str, *, limit: int = 30) -> list[str]:
    desired = set(BASE_SEARCH_TERMS)
    for item in materials:
        claim = str(item.get("claim") or "")
        for trigger, terms in TERM_HINTS.items():
            if trigger in claim:
                desired.update(terms)
    missing = [term for term in sorted(desired) if term not in search_index_text]
    return missing[:limit]


def find_related_gaps(vault: Path) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for rel in KEY_NOTE_RELS:
        path = vault / rel
        text = _read_text(path)
        if not text:
            gaps.append({"path": str(rel), "issue": "missing_or_unreadable", "missing_links": list(REQUIRED_RELATED_TARGETS)})
            continue
        missing = [target for target in REQUIRED_RELATED_TARGETS if target not in text]
        if "## Related" not in text and "### Related" not in text:
            gaps.append({"path": str(rel), "issue": "related_section_missing", "missing_links": missing})
        elif missing:
            gaps.append({"path": str(rel), "issue": "related_links_incomplete", "missing_links": missing})
    return gaps


def find_mana_review_items(materials: list[dict[str, Any]], mana_report: dict[str, Any] | None, *, limit: int = 12) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    mana_status = str((mana_report or {}).get("status") or "missing")
    if mana_status != "allow":
        items.append(
            {
                "reason": "mana_not_allow",
                "status": mana_status,
                "action": "Inbox整理は可。GCS/RAG/active store接続は停止。",
            }
        )
    for item in materials:
        claim = str(item.get("claim") or "")
        hits = [term for term in HARD_MANA_TERMS if term in claim]
        if not hits:
            continue
        items.append(
            {
                "reason": "hard_mana_term",
                "hits": hits,
                "claim_fingerprint": "cur_" + str(abs(hash(_meaning_key(claim))))[:10],
                "source": str(item.get("evidence_path") or item.get("source_path") or ""),
            }
        )
        if len(items) >= limit:
            break
    return items[:limit]


def build_report(
    *,
    vault: Path,
    materials: list[dict[str, Any]],
    mana_report: dict[str, Any] | None,
) -> dict[str, Any]:
    search_index_text = _read_text(vault / DEFAULT_PROJECT_REL / "検索語インデックス.md")
    candidate_counts = Counter(str(item.get("material_type") or item.get("candidate_type") or "unknown") for item in materials)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "agent": "Obsidian Curator",
        "mode": "read_only_report_only",
        "guardrail": "no_obsidian_write_no_rag_no_prompt_no_scoring_no_gcs_no_cloudrun",
        "vault": str(vault),
        "materials_count": len(materials),
        "material_counts": dict(candidate_counts),
        "mana_status": str((mana_report or {}).get("status") or "missing"),
        "inbox_candidates": select_inbox_candidates(materials),
        "duplicate_clusters": find_duplicate_clusters(materials),
        "search_terms_to_add": suggest_search_terms(materials, search_index_text),
        "related_gaps": find_related_gaps(vault),
        "mana_review_items": find_mana_review_items(materials, mana_report),
        "after_hackathon_only": [
            "Obsidianディレクトリ再編",
            "GCS Vault include/exclude変更",
            "accepted判断資産のactive store連携",
            "判断資産レビューUI",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Obsidian Curator Report",
        "",
        "## Summary",
        f"- generated_at: `{report['generated_at']}`",
        f"- agent: `{report['agent']}`",
        f"- mode: `{report['mode']}`",
        f"- guardrail: `{report['guardrail']}`",
        f"- materials: `{report['materials_count']}`",
        f"- mana_status: `{report['mana_status']}`",
        "",
        "## Material Counts",
        *_dict_lines(report.get("material_counts") or {}),
        "",
        "## Inbox Candidates",
        *_candidate_lines(report.get("inbox_candidates") or []),
        "",
        "## Duplicate Clusters",
        *_duplicate_lines(report.get("duplicate_clusters") or []),
        "",
        "## Search Terms To Add",
        *_list_lines(report.get("search_terms_to_add") or []),
        "",
        "## Related Gaps",
        *_gap_lines(report.get("related_gaps") or []),
        "",
        "## Mana Review Items",
        *_mana_lines(report.get("mana_review_items") or []),
        "",
        "## After Hackathon Only",
        *_list_lines(report.get("after_hackathon_only") or []),
        "",
        "## Next Safe Step",
        "- Inbox候補を人間が採用・修正・却下・後回しに分類する。",
        "- このレポート自体はObsidian本文、RAG、Cloud Run、active storeへ接続しない。",
        "",
    ]
    return "\n".join(lines)


def _dict_lines(values: dict[str, Any]) -> list[str]:
    if not values:
        return ["- なし"]
    return [f"- {key}: `{value}`" for key, value in sorted(values.items())]


def _list_lines(items: list[Any]) -> list[str]:
    if not items:
        return ["- なし"]
    return [f"- {item}" for item in items]


def _candidate_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- なし"]
    lines: list[str] = []
    for item in items[:12]:
        lines.append(f"- `{item['type']}` {item['claim']} / source=`{item['source']}`")
    return lines


def _duplicate_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- なし"]
    lines: list[str] = []
    for item in items:
        lines.append(f"- `{item['type']}` count={item['count']} sample={item['sample_claim']}")
    return lines


def _gap_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- なし"]
    return [
        f"- `{item['path']}` {item['issue']} missing={', '.join(item.get('missing_links') or [])}"
        for item in items
    ]


def _mana_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- なし"]
    return [f"- {json.dumps(item, ensure_ascii=False, sort_keys=True)}" for item in items]


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build read-only Obsidian Curator report.")
    parser.add_argument("--vault", default="")
    parser.add_argument("--materials-jsonl", type=Path, default=DEFAULT_MATERIALS_JSONL)
    parser.add_argument("--mana-json", type=Path, default=DEFAULT_MANA_JSON)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vault = _vault_path(args.vault)
    materials = _read_jsonl(args.materials_jsonl)
    mana_report = _read_json(args.mana_json)
    report = build_report(vault=vault, materials=materials, mana_report=mana_report)
    if args.dry_run:
        print(render_markdown(report))
        return 0
    write_report(report, args.json, args.report)
    print(f"json={args.json}")
    print(f"report={args.report}")
    print(f"materials={report['materials_count']}")
    print(f"mana_status={report['mana_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
