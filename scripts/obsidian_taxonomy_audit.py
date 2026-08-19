#!/usr/bin/env python3
"""Vault-wide tag/folder duplicate audit and naming-rule proposal.

Deterministic and read-only: no LLM call, no web call, no Obsidian write.
Scans frontmatter ``tags`` and folder names across the vault, groups
near-duplicates (case, full/half-width, separator, singular/plural), and
proposes up to 5 naming rules to prevent new duplicates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import resolve_obsidian_vault  # noqa: E402
from scripts._obsidian_common import iter_vault_markdown_files, note_tags, parse_note_frontmatter  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "obsidian_taxonomy_audit_latest.json"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "obsidian_taxonomy_audit_latest.md"

FUZZY_THRESHOLD = 0.84


def _normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = re.sub(r"[\s_\-/]+", "", text)
    return text


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def collect_tags_and_folders(vault: Path) -> tuple[Counter[str], Counter[str]]:
    tag_counts: Counter[str] = Counter()
    folder_counts: Counter[str] = Counter()
    for path in iter_vault_markdown_files(vault):
        text = _read_text(path)
        if not text:
            continue
        frontmatter = parse_note_frontmatter(text)
        for tag in note_tags(frontmatter):
            tag_counts[tag] += 1
        rel = path.relative_to(vault)
        for part in rel.parts[:-1]:
            folder_counts[part] += 1
    return tag_counts, folder_counts


def _exact_normalized_clusters(counts: Counter[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for label in counts:
        groups[_normalize_label(label)].append(label)
    clusters = []
    for normalized, labels in groups.items():
        if len(labels) < 2:
            continue
        clusters.append(
            {
                "normalized": normalized,
                "labels": sorted(labels),
                "counts": {label: counts[label] for label in labels},
                "reason": "normalized_form_matches",
            }
        )
    return sorted(clusters, key=lambda item: item["normalized"])


def _fuzzy_clusters(counts: Counter[str], exact_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    already_grouped = {label for cluster in exact_clusters for label in cluster["labels"]}
    candidates = sorted(label for label in counts if label not in already_grouped)
    pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            norm_left, norm_right = _normalize_label(left), _normalize_label(right)
            if not norm_left or not norm_right:
                continue
            ratio = SequenceMatcher(None, norm_left, norm_right).ratio()
            if ratio < FUZZY_THRESHOLD:
                continue
            key = (left, right)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            pairs.append(
                {
                    "labels": [left, right],
                    "counts": {left: counts[left], right: counts[right]},
                    "similarity": round(ratio, 3),
                    "reason": "high_string_similarity",
                }
            )
    return sorted(pairs, key=lambda item: -item["similarity"])


def _has_case_only_variants(clusters: list[dict[str, Any]]) -> bool:
    for cluster in clusters:
        lowered = {label.lower() for label in cluster["labels"]}
        if len(lowered) == 1 and len(cluster["labels"]) > 1:
            return True
    return False


def _has_width_variants(clusters: list[dict[str, Any]]) -> bool:
    for cluster in clusters:
        widths = {unicodedata.normalize("NFKC", label) for label in cluster["labels"]}
        if len(widths) == 1 and len({label for label in cluster["labels"]}) > 1:
            return True
    return False


def _has_separator_variants(clusters: list[dict[str, Any]]) -> bool:
    for cluster in clusters:
        stripped = {re.sub(r"[\s_\-]", "", label.lower()) for label in cluster["labels"]}
        if len(stripped) == 1 and len(cluster["labels"]) > 1:
            return True
    return False


def _has_plural_variants(clusters: list[dict[str, Any]]) -> bool:
    for cluster in clusters:
        for label in cluster["labels"]:
            if label.endswith("s") and label[:-1] in cluster["labels"]:
                return True
    return False


def propose_naming_rules(
    tag_clusters: list[dict[str, Any]],
    folder_clusters: list[dict[str, Any]],
    tag_fuzzy: list[dict[str, Any]],
) -> list[str]:
    all_clusters = tag_clusters + folder_clusters
    rules: list[str] = []
    if _has_case_only_variants(all_clusters):
        rules.append("英数字のタグ・フォルダ名は小文字に統一する（例: AI ではなく ai）")
    if _has_width_variants(all_clusters):
        rules.append("全角英数字・記号は使わず半角に統一する")
    if _has_separator_variants(all_clusters):
        rules.append("単語の区切りはハイフン(-)に統一する（スペース・アンダースコアを避ける）")
    if _has_plural_variants(all_clusters):
        rules.append("タグは単数形で統一する（複数形にしない）")
    if tag_fuzzy:
        rules.append("新しいタグを作る前に既存タグ一覧を検索し、近い意味のタグがあれば使い回す")
    if not rules:
        rules.append("既存タグ・フォルダ名の一覧を作成し、新規追加前に重複がないか確認する運用を続ける")
    return rules[:5]


def build_taxonomy_audit(vault: Path) -> dict[str, Any]:
    tag_counts, folder_counts = collect_tags_and_folders(vault)
    tag_clusters = _exact_normalized_clusters(tag_counts)
    folder_clusters = _exact_normalized_clusters(folder_counts)
    tag_fuzzy = _fuzzy_clusters(tag_counts, tag_clusters)
    folder_fuzzy = _fuzzy_clusters(folder_counts, folder_clusters)

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "vault": str(vault),
        "totals": {
            "distinct_tags": len(tag_counts),
            "distinct_folders": len(folder_counts),
            "tag_notes_scanned": sum(tag_counts.values()),
        },
        "tag_duplicate_clusters": tag_clusters,
        "tag_fuzzy_duplicate_pairs": tag_fuzzy,
        "folder_duplicate_clusters": folder_clusters,
        "folder_fuzzy_duplicate_pairs": folder_fuzzy,
        "naming_rules": propose_naming_rules(tag_clusters, folder_clusters, tag_fuzzy),
        "top_tags": [{"tag": tag, "count": count} for tag, count in tag_counts.most_common(20)],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Obsidian Taxonomy Audit")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Distinct tags: {report['totals']['distinct_tags']}")
    lines.append(f"- Distinct folders: {report['totals']['distinct_folders']}")
    lines.append("")
    lines.append("## Tag Duplicate Clusters (normalized match)")
    if not report["tag_duplicate_clusters"]:
        lines.append("- なし")
    for cluster in report["tag_duplicate_clusters"]:
        labels = ", ".join(f"{label}({cluster['counts'][label]})" for label in cluster["labels"])
        lines.append(f"- {labels}")
    lines.append("")
    lines.append("## Tag Fuzzy Duplicate Pairs")
    if not report["tag_fuzzy_duplicate_pairs"]:
        lines.append("- なし")
    for pair in report["tag_fuzzy_duplicate_pairs"]:
        labels = " / ".join(pair["labels"])
        lines.append(f"- {labels} (similarity={pair['similarity']})")
    lines.append("")
    lines.append("## Folder Duplicate Clusters")
    if not report["folder_duplicate_clusters"]:
        lines.append("- なし")
    for cluster in report["folder_duplicate_clusters"]:
        labels = ", ".join(f"{label}({cluster['counts'][label]})" for label in cluster["labels"])
        lines.append(f"- {labels}")
    lines.append("")
    lines.append("## 命名ルール提案")
    for rule in report["naming_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("## Top Tags")
    for item in report["top_tags"]:
        lines.append(f"- {item['tag']}: {item['count']}")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vault = args.vault.expanduser() if args.vault else resolve_obsidian_vault()
    if not vault.exists():
        print(f"[ERROR] Vault not found: {vault}")
        return 1

    report = build_taxonomy_audit(vault)
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    write_outputs(report, output_json=args.output_json.expanduser(), output_md=args.output_md.expanduser())
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
