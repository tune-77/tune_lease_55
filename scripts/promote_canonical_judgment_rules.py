#!/usr/bin/env python3
"""Promote accepted canonical judgment-rule previews into the active store.

This script is intentionally conservative:

- Only `accepted_preview` rules are promoted.
- Existing rules with the same stable id are updated, not duplicated.
- Previously promoted rules that are absent from the preview are kept.
- The output is a local JSON store; it does not write to Obsidian.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_INPUT = DATA_DIR / "canonical_judgment_rules_preview.json"
DEFAULT_OUTPUT = DATA_DIR / "canonical_judgment_rules.json"


def read_preview(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rules = payload.get("canonical_rules") if isinstance(payload, dict) else []
    return [item for item in rules if isinstance(item, dict)]


def read_store(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    rules = payload.get("rules") if isinstance(payload, dict) else []
    if not isinstance(rules, list):
        rules = []
    return {
        "schema_version": int(payload.get("schema_version") or 1) if isinstance(payload, dict) else 1,
        "rules": [item for item in rules if isinstance(item, dict)],
    }


def _promotable(rule: dict[str, Any]) -> bool:
    if rule.get("private") is True:
        return False
    if rule.get("preview") is not True:
        return False
    if rule.get("status") != "accepted_preview":
        return False
    if not str(rule.get("canonical_statement") or "").strip():
        return False
    return True


def _promoted_rule(rule: dict[str, Any], *, now: str, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    previous = previous or {}
    return {
        "id": str(rule.get("id") or previous.get("id") or ""),
        "status": "active",
        "source_status": str(rule.get("status") or "accepted_preview"),
        "material_type": str(rule.get("material_type") or previous.get("material_type") or "judgment_rule"),
        "material_types": list(rule.get("material_types") or previous.get("material_types") or [str(rule.get("material_type") or previous.get("material_type") or "judgment_rule")]),
        "domain": str(rule.get("domain") or previous.get("domain") or "lease_screening"),
        "domains": list(rule.get("domains") or previous.get("domains") or [str(rule.get("domain") or previous.get("domain") or "lease_screening")]),
        "concept": str(rule.get("concept") or previous.get("concept") or ""),
        "canonical_statement": str(rule.get("canonical_statement") or previous.get("canonical_statement") or "").strip(),
        "evidence_count": int(rule.get("evidence_count") or previous.get("evidence_count") or 0),
        "user_evidence_count": int(rule.get("user_evidence_count") or previous.get("user_evidence_count") or 0),
        "confidence": float(rule.get("confidence") or previous.get("confidence") or 0.7),
        "risk_axis": list(rule.get("risk_axis") or previous.get("risk_axis") or [])[:5],
        "sample_claims": list(rule.get("sample_claims") or previous.get("sample_claims") or [])[:8],
        "evidence_paths": list(rule.get("evidence_paths") or previous.get("evidence_paths") or [])[:12],
        "created_at": str(previous.get("created_at") or now),
        "updated_at": now,
        "promotion_source": "canonical_judgment_rules_preview",
        "private": False,
    }


def _merge_preview_rules(preview_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rule in preview_rules:
        if not _promotable(rule):
            continue
        key = (
            str(rule.get("concept") or ""),
            str(rule.get("canonical_statement") or "").strip(),
            "",
        )
        if key not in merged:
            merged[key] = dict(rule)
            merged[key]["material_types"] = [str(rule.get("material_type") or "judgment_rule")]
            merged[key]["domains"] = [str(rule.get("domain") or "lease_screening")]
            continue
        target = merged[key]
        target["evidence_count"] = int(target.get("evidence_count") or 0) + int(rule.get("evidence_count") or 0)
        target["user_evidence_count"] = int(target.get("user_evidence_count") or 0) + int(rule.get("user_evidence_count") or 0)
        target["confidence"] = max(float(target.get("confidence") or 0), float(rule.get("confidence") or 0))
        material_type = str(rule.get("material_type") or "judgment_rule")
        if material_type not in target["material_types"]:
            target["material_types"].append(material_type)
        domain = str(rule.get("domain") or "lease_screening")
        if domain not in target["domains"]:
            target["domains"].append(domain)
        for field, limit in (("risk_axis", 5), ("sample_claims", 8), ("evidence_paths", 12)):
            values = list(target.get(field) or [])
            for value in rule.get(field) or []:
                if value not in values:
                    values.append(value)
            target[field] = values[:limit]
    return list(merged.values())


def _semantic_key(rule: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(rule.get("concept") or ""),
        str(rule.get("canonical_statement") or "").strip(),
        "",
    )


def _merge_rule_dicts(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged["evidence_count"] = max(int(base.get("evidence_count") or 0), int(incoming.get("evidence_count") or 0))
    merged["user_evidence_count"] = max(int(base.get("user_evidence_count") or 0), int(incoming.get("user_evidence_count") or 0))
    merged["confidence"] = max(float(base.get("confidence") or 0), float(incoming.get("confidence") or 0))
    material_types = list(base.get("material_types") or [base.get("material_type") or "judgment_rule"])
    for material_type in list(incoming.get("material_types") or [incoming.get("material_type") or "judgment_rule"]):
        material_type = str(material_type)
        if material_type not in material_types:
            material_types.append(material_type)
    merged["material_types"] = material_types
    domains = list(base.get("domains") or [base.get("domain") or "lease_screening"])
    for domain in list(incoming.get("domains") or [incoming.get("domain") or "lease_screening"]):
        domain = str(domain)
        if domain not in domains:
            domains.append(domain)
    merged["domains"] = domains
    for field, limit in (("risk_axis", 5), ("sample_claims", 8), ("evidence_paths", 12)):
        values = list(base.get(field) or [])
        for value in incoming.get(field) or []:
            if value not in values:
                values.append(value)
        merged[field] = values[:limit]
    if str(incoming.get("updated_at") or "") > str(base.get("updated_at") or ""):
        merged["updated_at"] = incoming.get("updated_at")
    return merged


def promote_rules(preview_rules: list[dict[str, Any]], existing_store: dict[str, Any], *, now: str | None = None) -> dict[str, Any]:
    now = now or dt.datetime.now().isoformat(timespec="seconds")
    existing_rules = existing_store.get("rules") or []
    by_semantic: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rule in existing_rules:
        if not isinstance(rule, dict):
            continue
        key = _semantic_key(rule)
        if key in by_semantic:
            by_semantic[key] = _merge_rule_dicts(by_semantic[key], rule)
        else:
            by_semantic[key] = dict(rule)
    promoted_count = 0
    updated_count = 0
    skipped_count = 0

    promotable_rules = _merge_preview_rules(preview_rules)
    skipped_count = sum(1 for rule in preview_rules if not _promotable(rule))

    for rule in promotable_rules:
        rid = str(rule.get("id") or "")
        if not rid:
            skipped_count += 1
            continue
        key = _semantic_key(rule)
        previous = by_semantic.get(key)
        promoted = _promoted_rule(rule, now=now, previous=previous)
        by_semantic[key] = _merge_rule_dicts(previous, promoted) if previous else promoted
        if previous:
            updated_count += 1
        else:
            promoted_count += 1

    rules = sorted(
        by_semantic.values(),
        key=lambda item: (
            -int(item.get("evidence_count") or 0),
            -int(item.get("user_evidence_count") or 0),
            str(item.get("concept") or ""),
            str(item.get("material_type") or ""),
        ),
    )
    return {
        "schema_version": 1,
        "generated_at": now,
        "source": "canonical_judgment_rules_preview",
        "summary": {
            "active_rules": len(rules),
            "promoted": promoted_count,
            "updated": updated_count,
            "skipped": skipped_count,
        },
        "rules": rules,
    }


def write_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _markdown(store: dict[str, Any]) -> str:
    summary = store.get("summary") or {}
    lines = [
        "# Canonical Judgment Rules",
        "",
        "## Summary",
        "",
        f"- Active rules: {summary.get('active_rules', 0)}",
        f"- Promoted: {summary.get('promoted', 0)}",
        f"- Updated: {summary.get('updated', 0)}",
        f"- Skipped preview rules: {summary.get('skipped', 0)}",
        "",
        "## Safety",
        "",
        "- Only accepted_preview rules are promoted.",
        "- This is a local active store. Obsidian is not modified.",
        "- These rules can be included in the Shion memory index as judgment_memory.",
        "",
        "## Rules",
        "",
    ]
    for rule in store.get("rules") or []:
        axes = ", ".join(rule.get("risk_axis") or [])
        lines += [
            f"### {rule.get('concept')} / evidence={rule.get('evidence_count')} / user={rule.get('user_evidence_count')}",
            "",
            f"- Rule: {rule.get('canonical_statement')}",
            f"- Type: {rule.get('material_type')}",
            f"- Confidence: {rule.get('confidence')}",
            f"- Axis: {axes or 'n/a'}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(store: dict[str, Any], *, date: dt.date) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_key = date.isoformat().replace("-", "")
    md_path = REPORTS_DIR / f"canonical_judgment_rules_{date_key}.md"
    latest_md = REPORTS_DIR / "canonical_judgment_rules_latest.md"
    md = _markdown(store)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")
    return {"markdown": str(md_path), "latest_markdown": str(latest_md)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote accepted canonical judgment-rule previews")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    store = promote_rules(read_preview(input_path), read_store(output_path))
    write_store(output_path, store)
    paths = write_report(store, date=dt.date.fromisoformat(args.date))
    print(
        json.dumps(
            {
                "active_rules": store["summary"]["active_rules"],
                "promoted": store["summary"]["promoted"],
                "updated": store["summary"]["updated"],
                "skipped": store["summary"]["skipped"],
                "output": str(output_path),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
