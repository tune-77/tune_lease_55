#!/usr/bin/env python3
"""Promote human-adopted Obsidian Curator Inbox candidates into the judgment-rule preview.

This is a deliberately narrow bridge between two existing, already-reviewed
pipelines. It does not invent a new active-store write path:

    obsidian_curator_report.py (Inbox candidates, read-only)
        -> [human marks candidate_id as "adopt" in data/curator_inbox_decisions.jsonl]
        -> this script appends an `accepted_preview` rule to
           data/canonical_judgment_rules_preview.json
        -> [human runs the existing scripts/promote_canonical_judgment_rules.py]
        -> data/canonical_judgment_rules.json (active store)

Guardrails (mirrors mana_obsidian_curator.py / obsidian_curator_report.py):
    - Refuses to write anything unless the latest Mana report status is "allow".
    - Only acts on candidates a human explicitly marked "adopt" by candidate_id;
      nothing is adopted automatically from confidence/evidence-count heuristics.
    - Never writes to Obsidian, RAG, prompts, scoring, or the active store directly.
    - Re-running with the same decisions is idempotent (same id -> same rule, no dup).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURATOR_JSON = REPO_ROOT / "reports" / "obsidian_curator_latest.json"
DEFAULT_MANA_JSON = REPO_ROOT / "reports" / "mana_obsidian_curator_latest.json"
DEFAULT_DECISIONS_JSONL = REPO_ROOT / "data" / "curator_inbox_decisions.jsonl"
DEFAULT_PREVIEW_JSON = REPO_ROOT / "data" / "canonical_judgment_rules_preview.json"

GUARDRAIL = "human_adopt_required_mana_allow_required_no_active_store_write_no_obsidian_write"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
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


def _adopted_candidate_ids(decisions: list[dict[str, Any]]) -> set[str]:
    """最後に書かれた決定を有効とする（jsonl追記形式・上書きは末尾優先）。"""
    latest: dict[str, str] = {}
    for row in decisions:
        candidate_id = str(row.get("candidate_id") or "").strip()
        decision = str(row.get("decision") or "").strip().lower()
        if not candidate_id or decision not in {"adopt", "reject"}:
            continue
        latest[candidate_id] = decision
    return {candidate_id for candidate_id, decision in latest.items() if decision == "adopt"}


def _rule_id_for(candidate_id: str) -> str:
    return f"curator_{candidate_id}"


def _preview_rule(candidate: dict[str, Any], *, now: str) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    claim = str(candidate.get("claim") or "").strip()
    confidence = candidate.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else 0.7
    except (TypeError, ValueError):
        confidence = 0.7
    source = str(candidate.get("source") or "").strip()
    return {
        "id": _rule_id_for(candidate_id),
        "status": "accepted_preview",
        "preview": True,
        "private": False,
        "material_type": str(candidate.get("type") or "judgment_rule"),
        "domain": "lease_screening",
        "concept": "curator_inbox_adopted",
        "canonical_statement": claim,
        "evidence_count": 1,
        "user_evidence_count": 0,
        "confidence": max(0.0, min(1.0, confidence)),
        "risk_axis": [],
        "parent_ids": [],
        "derivation_reason": "obsidian_curator_report.py Inbox Candidates を人間が採用",
        "sample_claims": [claim] if claim else [],
        "evidence_paths": [source] if source else [],
        "source": "curator_inbox_adopted",
        "candidate_id": candidate_id,
        "created_at": now,
        "updated_at": now,
    }


def build_promotion(
    *,
    curator_report: dict[str, Any] | None,
    mana_report: dict[str, Any] | None,
    decisions: list[dict[str, Any]],
    existing_preview: dict[str, Any] | None,
    now: str | None = None,
) -> dict[str, Any]:
    now = now or dt.datetime.now().isoformat(timespec="seconds")
    mana_status = str((mana_report or {}).get("status") or "missing")

    if mana_status != "allow":
        return {
            "status": "blocked",
            "reason": "mana_not_allow",
            "mana_status": mana_status,
            "guardrail": GUARDRAIL,
            "promoted": [],
            "skipped": [],
            "preview": existing_preview,
        }

    if curator_report is None:
        return {
            "status": "blocked",
            "reason": "curator_report_missing",
            "mana_status": mana_status,
            "guardrail": GUARDRAIL,
            "promoted": [],
            "skipped": [],
            "preview": existing_preview,
        }

    adopted_ids = _adopted_candidate_ids(decisions)
    candidates_by_id = {
        str(item.get("candidate_id") or ""): item
        for item in curator_report.get("inbox_candidates") or []
        if item.get("candidate_id")
    }

    preview = existing_preview if isinstance(existing_preview, dict) else {}
    existing_rules = [r for r in (preview.get("canonical_rules") or []) if isinstance(r, dict)]
    by_id = {str(rule.get("id") or ""): rule for rule in existing_rules if rule.get("id")}

    promoted: list[str] = []
    skipped: list[dict[str, str]] = []
    for candidate_id in sorted(adopted_ids):
        candidate = candidates_by_id.get(candidate_id)
        if not candidate:
            skipped.append({"candidate_id": candidate_id, "reason": "candidate_not_in_latest_curator_report"})
            continue
        rule = _preview_rule(candidate, now=now)
        by_id[rule["id"]] = rule
        promoted.append(candidate_id)

    new_preview = {
        "schema_version": int(preview.get("schema_version") or 1),
        "generated_at": now,
        "canonical_rules": list(by_id.values()),
    }
    return {
        "status": "applied",
        "reason": "" if promoted else "no_new_adopted_candidates",
        "mana_status": mana_status,
        "guardrail": GUARDRAIL,
        "promoted": promoted,
        "skipped": skipped,
        "preview": new_preview,
    }


def write_preview(preview: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curator-json", type=Path, default=DEFAULT_CURATOR_JSON)
    parser.add_argument("--mana-json", type=Path, default=DEFAULT_MANA_JSON)
    parser.add_argument("--decisions-jsonl", type=Path, default=DEFAULT_DECISIONS_JSONL)
    parser.add_argument("--preview-json", type=Path, default=DEFAULT_PREVIEW_JSON)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = build_promotion(
        curator_report=_read_json(args.curator_json),
        mana_report=_read_json(args.mana_json),
        decisions=_read_jsonl(args.decisions_jsonl),
        existing_preview=_read_json(args.preview_json),
    )

    if result["status"] == "applied" and not args.dry_run:
        write_preview(result["preview"], args.preview_json)

    print(
        json.dumps(
            {
                "status": result["status"],
                "reason": result["reason"],
                "mana_status": result["mana_status"],
                "promoted": result["promoted"],
                "skipped": result["skipped"],
                "wrote": str(args.preview_json) if result["status"] == "applied" and not args.dry_run else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "applied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
