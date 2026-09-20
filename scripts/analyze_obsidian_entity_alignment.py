#!/usr/bin/env python3
"""Generate a read-only Obsidian entity-alignment report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obsidian_entity_alignment import (  # noqa: E402
    DEFAULT_INCLUDE_ROOTS,
    alignment_enabled,
    judge_pairs,
    load_notes,
    select_candidate_pairs,
)
from runtime_paths import resolve_obsidian_vault  # noqa: E402

DEFAULT_OUTPUT_JSON = ROOT / "reports" / "obsidian_entity_alignment_latest.json"
DEFAULT_OUTPUT_MD = ROOT / "reports" / "obsidian_entity_alignment_latest.md"


def _offline_rows(pairs) -> list[dict[str, Any]]:
    return [
        {
            "a": pair.a,
            "b": pair.b,
            "local_similarity": pair.similarity,
            "reasons": list(pair.reasons),
            "already_linked": pair.already_linked,
            "route": "unjudged",
        }
        for pair in pairs
    ]


def build_report(vault: Path, notes, pairs, judged, meta) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in judged:
        row = dict(item)
        row["note_a"] = {"path": notes[item["a"]].path, "title": notes[item["a"]].title}
        row["note_b"] = {"path": notes[item["b"]].path, "title": notes[item["b"]].title}
        rows.append(row)
    routes = Counter(str(row.get("route") or "unknown") for row in rows)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "guardrail": "read_only_no_vault_write",
        "vault": str(vault),
        "summary": {
            "notes_scanned": len(notes),
            "candidate_pairs": len(pairs),
            "routes": dict(sorted(routes.items())),
        },
        "typesafe": meta,
        "pairs": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Obsidian Entity Alignment Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Guardrail: `{report['guardrail']}`",
        f"- Notes scanned: {summary['notes_scanned']}",
        f"- Candidate pairs: {summary['candidate_pairs']}",
        f"- TypeSafe status: `{report['typesafe'].get('status', 'unknown')}`",
        "",
        "## Route Summary",
        "",
    ]
    if summary["routes"]:
        lines.extend(f"- `{route}`: {count}" for route, count in summary["routes"].items())
    else:
        lines.append("- No candidate pairs")
    lines.extend(["", "## Candidates", ""])
    for index, row in enumerate(report["pairs"], 1):
        a, b = row["note_a"], row["note_b"]
        lines.extend(
            [
                f"### {index}. {a['title']} ↔ {b['title']}",
                "",
                f"- Route: `{row['route']}`",
                f"- Local similarity: `{row['local_similarity']:.3f}`",
                f"- Signals: {', '.join(row.get('reasons') or []) or 'none'}",
                f"- Existing direct link: `{str(bool(row.get('already_linked'))).lower()}`",
                f"- Note A: `{a['path']}`",
                f"- Note B: `{b['path']}`",
            ]
        )
        if "score" in row:
            lines.extend(
                [
                    f"- Alignment score/confidence: `{row['score']:.3f}` / `{row['confidence']:.3f}`",
                    f"- Same subject: `{row['same_subject']:.3f}`",
                    f"- Same conclusion: `{row['same_conclusion']:.3f}`",
                    f"- Contradiction: `{row['contradiction']:.3f}`",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Safety",
            "",
            "- This report did not modify the Vault.",
            "- Local paths were not included in TypeSafe request state.",
            "- Review every proposed link or merge before changing notes.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=resolve_obsidian_vault())
    parser.add_argument("--include-root", action="append", dest="include_roots")
    parser.add_argument("--max-notes", type=int, default=1000)
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--min-similarity", type=float, default=0.18)
    parser.add_argument("--read-timeout", type=float, default=0.5, help="Per-note iCloud read timeout in seconds")
    parser.add_argument("--send", action="store_true", help="Call Jev; also requires TYPESAFE_OBSIDIAN_ALIGNMENT_ENABLED=1")
    parser.add_argument("--inspect", action="store_true", help="Print the exact compact note state that a live run would send")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = tuple(args.include_roots or DEFAULT_INCLUDE_ROOTS)
    notes = load_notes(
        args.vault,
        include_roots=roots,
        max_notes=max(1, args.max_notes),
        read_timeout_seconds=max(0.0, args.read_timeout),
    )
    pairs = select_candidate_pairs(
        notes,
        min_similarity=max(0.0, min(1.0, args.min_similarity)),
        max_pairs=max(1, args.max_pairs),
    )
    if args.inspect:
        preview = [
            {"note_a": notes[pair.a].public_state(), "note_b": notes[pair.b].public_state()}
            for pair in pairs
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        if not args.send:
            return 0

    if args.send:
        if not alignment_enabled():
            print(
                "Live alignment is disabled. Set TYPESAFE_OBSIDIAN_ALIGNMENT_ENABLED=1 and configure the server-side TypeSafe key.",
                file=sys.stderr,
            )
            return 2
        try:
            judged, meta = judge_pairs(notes, pairs)
        except Exception as exc:
            judged = _offline_rows(pairs)
            meta = {"status": "fallback", "error_type": type(exc).__name__}
    else:
        judged = _offline_rows(pairs)
        meta = {"status": "offline_candidates_only"}

    report = build_report(args.vault, notes, pairs, judged, meta)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"notes={len(notes)} candidates={len(pairs)} typesafe={meta['status']}")
    print(args.output_md)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
