#!/usr/bin/env python3
"""Vault-wide tag rename/merge tool, built on top of ``obsidian_taxonomy_audit``.

Deterministic and (by default) read-only: computes a canonical tag for every
duplicate cluster detected by ``build_taxonomy_audit`` (case / full-half width /
separator variants), optionally merged with an explicit ``--rename-map`` for
cases the audit can't decide on its own (fuzzy near-duplicates, synonyms).

Only the ``tags:`` frontmatter field is touched; the rest of each note
(including frontmatter formatting and body) is left byte-for-byte untouched.
Nothing is written to disk unless ``--apply`` is passed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import resolve_obsidian_vault  # noqa: E402
from scripts._obsidian_common import (  # noqa: E402
    FRONTMATTER_RE,
    iter_vault_markdown_files,
    note_tags,
    parse_note_frontmatter,
)
from scripts.obsidian_taxonomy_audit import build_taxonomy_audit  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "obsidian_tag_cleanup_latest.json"

_NEEDS_QUOTE_RE = re.compile(r"""[:#\[\]{}&*!|>'"%@,]|^\s|\s$""")


def compute_canonical_mapping(
    tag_clusters: list[dict[str, Any]],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """One old->canonical mapping per duplicate cluster, plus explicit overrides.

    Canonical choice: highest note-count label in the cluster, ties broken
    alphabetically. Overrides always win, including for tags outside any
    detected cluster (e.g. fuzzy pairs a human decided to merge).
    """
    mapping: dict[str, str] = {}
    for cluster in tag_clusters:
        labels = cluster["labels"]
        counts = cluster["counts"]
        canonical = sorted(labels, key=lambda label: (-counts[label], label))[0]
        for label in labels:
            if label != canonical:
                mapping[label] = canonical
    if overrides:
        mapping.update(overrides)
    return mapping


def _quote_if_needed(tag: str) -> str:
    if _NEEDS_QUOTE_RE.search(tag):
        return '"' + tag.replace('"', '\\"') + '"'
    return tag


def _render_tags_block(new_tags: list[str], *, style: str, indent: str) -> str:
    if style == "flow":
        return "tags: [" + ", ".join(_quote_if_needed(t) for t in new_tags) + "]"
    if style == "scalar" and len(new_tags) == 1:
        return "tags: " + _quote_if_needed(new_tags[0])
    lines = ["tags:"]
    for tag in new_tags:
        lines.append(f"{indent}- {_quote_if_needed(tag)}")
    return "\n".join(lines)


def _find_tags_field(frontmatter_body: str) -> tuple[int, int, str, str] | None:
    """Locate the ``tags:`` field's line span within a frontmatter body.

    Returns (start_line_idx, end_line_idx_exclusive, style, indent) or None
    if there is no ``tags:`` field to rewrite.
    """
    lines = frontmatter_body.split("\n")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("tags:"):
            continue
        rest = stripped[len("tags:") :].strip()
        if rest.startswith("["):
            return idx, idx + 1, "flow", "  "
        if rest:
            return idx, idx + 1, "scalar", "  "
        end = idx + 1
        indent = "  "
        while end < len(lines) and re.match(r"^(\s+)-\s*.*$", lines[end]):
            indent = re.match(r"^(\s+)-", lines[end]).group(1)
            end += 1
        return idx, end, "block", indent
    return None


def rewrite_tags_field(text: str, new_tags: list[str]) -> str | None:
    """Rewrite a note's frontmatter ``tags:`` field. Returns None if absent."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None
    body = match.group(1)
    field = _find_tags_field(body)
    if field is None:
        return None
    start, end, style, indent = field
    lines = body.split("\n")
    replacement = _render_tags_block(new_tags, style=style, indent=indent)
    new_body = "\n".join(lines[:start] + [replacement] + lines[end:])
    return text[: match.start(1)] + new_body + text[match.end(1) :]


def plan_tag_renames(vault: Path, mapping: dict[str, str]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for path in iter_vault_markdown_files(vault):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text:
            continue
        old_tags = note_tags(parse_note_frontmatter(text))
        if not old_tags:
            continue
        new_tags: list[str] = []
        for tag in old_tags:
            mapped = mapping.get(tag, tag)
            if mapped not in new_tags:
                new_tags.append(mapped)
        if new_tags == old_tags:
            continue
        changes.append(
            {
                "path": path.relative_to(vault).as_posix(),
                "before": old_tags,
                "after": new_tags,
            }
        )
    return changes


def apply_tag_renames(vault: Path, changes: list[dict[str, Any]]) -> int:
    written = 0
    for change in changes:
        path = vault / change["path"]
        text = path.read_text(encoding="utf-8", errors="ignore")
        new_text = rewrite_tags_field(text, change["after"])
        if new_text is None or new_text == text:
            continue
        path.write_text(new_text, encoding="utf-8")
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument(
        "--rename-map",
        type=Path,
        default=None,
        help='JSON file of {"old_tag": "new_tag"} overrides, applied on top of auto-detected clusters',
    )
    parser.add_argument("--apply", action="store_true", help="write changes to disk (default: dry-run)")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    vault = args.vault.expanduser() if args.vault else resolve_obsidian_vault()
    if not vault.exists():
        print(f"[ERROR] Vault not found: {vault}")
        return 1

    overrides: dict[str, str] = {}
    if args.rename_map:
        overrides = json.loads(args.rename_map.read_text(encoding="utf-8"))

    audit = build_taxonomy_audit(vault)
    mapping = compute_canonical_mapping(audit["tag_duplicate_clusters"], overrides)
    if not mapping:
        print("no duplicate tags detected (and no --rename-map given); nothing to do")
        return 0

    changes = plan_tag_renames(vault, mapping)
    report = {
        "vault": str(vault),
        "mapping": mapping,
        "applied": bool(args.apply),
        "changed_notes": changes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"mapping ({len(mapping)} tag(s)):")
    for old, new in sorted(mapping.items()):
        print(f"  {old} -> {new}")
    print(f"notes to change: {len(changes)}")
    for change in changes:
        print(f"  {change['path']}: {change['before']} -> {change['after']}")

    if args.apply:
        written = apply_tag_renames(vault, changes)
        print(f"written: {written}")
    else:
        print("dry-run only; re-run with --apply to write changes")
    print(f"saved: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
