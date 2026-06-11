"""Utilities for reading and summarizing prompt feedback logs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "prompt_feedback_log.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _count_diff_lines(diff_text: str) -> dict[str, int]:
    added = removed = context = 0
    for line in (diff_text or "").splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
        elif line.strip():
            context += 1
    return {"added": added, "removed": removed, "context": context}


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    surface_counts = Counter(str(row.get("surface") or "unknown") for row in rows)
    pdca_count = sum(1 for row in rows if row.get("pdca_applied"))
    previous_diff_count = sum(1 for row in rows if (row.get("response_diff_from_previous") or "").strip())
    response_len = [int(row.get("response_len") or 0) for row in rows if row.get("response_len") is not None]
    prompt_base_len = [int(row.get("prompt_base_len") or 0) for row in rows if row.get("prompt_base_len") is not None]
    prompt_final_len = [int(row.get("prompt_final_len") or 0) for row in rows if row.get("prompt_final_len") is not None]
    prompt_diff_sizes = [_count_diff_lines(str(row.get("prompt_diff") or "")) for row in rows]
    response_diff_sizes = [_count_diff_lines(str(row.get("response_diff_from_previous") or "")) for row in rows if row.get("response_diff_from_previous")]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("surface") or "unknown")].append(row)

    by_surface: dict[str, dict[str, Any]] = {}
    for surface, items in grouped.items():
        by_surface[surface] = {
            "count": len(items),
            "pdca_rate": round(sum(1 for row in items if row.get("pdca_applied")) / len(items) * 100, 1),
            "avg_response_len": round(mean(int(row.get("response_len") or 0) for row in items), 1),
            "avg_prompt_diff_added": round(mean(_count_diff_lines(str(row.get("prompt_diff") or ""))["added"] for row in items), 1),
            "avg_prompt_diff_removed": round(mean(_count_diff_lines(str(row.get("prompt_diff") or ""))["removed"] for row in items), 1),
            "response_changed_rate": round(sum(1 for row in items if (row.get("response_diff_from_previous") or "").strip()) / len(items) * 100, 1),
        }

    largest_prompt_changes = sorted(
        rows,
        key=lambda row: (
            _count_diff_lines(str(row.get("prompt_diff") or ""))["added"]
            + _count_diff_lines(str(row.get("prompt_diff") or ""))["removed"]
        ),
        reverse=True,
    )[:5]
    largest_response_changes = sorted(
        [row for row in rows if (row.get("response_diff_from_previous") or "").strip()],
        key=lambda row: (
            _count_diff_lines(str(row.get("response_diff_from_previous") or ""))["added"]
            + _count_diff_lines(str(row.get("response_diff_from_previous") or ""))["removed"]
        ),
        reverse=True,
    )[:5]

    return {
        "total": total,
        "surface_counts": dict(surface_counts),
        "pdca_count": pdca_count,
        "pdca_rate": round(pdca_count / total * 100, 1) if total else 0.0,
        "previous_diff_count": previous_diff_count,
        "previous_diff_rate": round(previous_diff_count / total * 100, 1) if total else 0.0,
        "avg_response_len": round(mean(response_len), 1) if response_len else 0.0,
        "avg_prompt_base_len": round(mean(prompt_base_len), 1) if prompt_base_len else 0.0,
        "avg_prompt_final_len": round(mean(prompt_final_len), 1) if prompt_final_len else 0.0,
        "avg_prompt_diff_added": round(mean(d["added"] for d in prompt_diff_sizes), 1) if prompt_diff_sizes else 0.0,
        "avg_prompt_diff_removed": round(mean(d["removed"] for d in prompt_diff_sizes), 1) if prompt_diff_sizes else 0.0,
        "avg_prompt_diff_context": round(mean(d["context"] for d in prompt_diff_sizes), 1) if prompt_diff_sizes else 0.0,
        "avg_response_diff_added": round(mean(d["added"] for d in response_diff_sizes), 1) if response_diff_sizes else 0.0,
        "avg_response_diff_removed": round(mean(d["removed"] for d in response_diff_sizes), 1) if response_diff_sizes else 0.0,
        "avg_response_diff_context": round(mean(d["context"] for d in response_diff_sizes), 1) if response_diff_sizes else 0.0,
        "by_surface": by_surface,
        "largest_prompt_changes": [
            {
                "timestamp": row.get("timestamp", ""),
                "surface": row.get("surface", ""),
                "question": row.get("question", ""),
                "prompt_diff": row.get("prompt_diff", ""),
            }
            for row in largest_prompt_changes
        ],
        "largest_response_changes": [
            {
                "timestamp": row.get("timestamp", ""),
                "surface": row.get("surface", ""),
                "question": row.get("question", ""),
                "response_diff_from_previous": row.get("response_diff_from_previous", ""),
            }
            for row in largest_response_changes
        ],
    }


def render_markdown(summary: dict[str, Any], source: Path) -> str:
    lines: list[str] = []
    lines.append("# Prompt Feedback Summary")
    lines.append("")
    lines.append(f"- Source: `{source}`")
    lines.append(f"- Total entries: {summary['total']}")
    lines.append(f"- PDCA applied: {summary['pdca_count']} ({summary['pdca_rate']}%)")
    lines.append(f"- Previous-response diffs: {summary['previous_diff_count']} ({summary['previous_diff_rate']}%)")
    lines.append(f"- Avg response length: {summary['avg_response_len']}")
    lines.append(f"- Avg prompt length: base {summary['avg_prompt_base_len']} -> final {summary['avg_prompt_final_len']}")
    lines.append(f"- Avg prompt diff: +{summary['avg_prompt_diff_added']} / -{summary['avg_prompt_diff_removed']} / context {summary['avg_prompt_diff_context']}")
    lines.append(f"- Avg response diff: +{summary['avg_response_diff_added']} / -{summary['avg_response_diff_removed']} / context {summary['avg_response_diff_context']}")
    lines.append("")
    lines.append("## Surfaces")
    if not summary["by_surface"]:
        lines.append("- No entries")
    else:
        for surface, stats in sorted(summary["by_surface"].items(), key=lambda kv: (-kv[1]["count"], kv[0])):
            lines.append(
                f"- `{surface}`: {stats['count']}件, PDCA {stats['pdca_rate']}%, "
                f"response変化率 {stats['response_changed_rate']}%, avg len {stats['avg_response_len']}"
            )
    lines.append("")
    lines.append("## Largest Prompt Changes")
    if not summary["largest_prompt_changes"]:
        lines.append("- No entries")
    else:
        for row in summary["largest_prompt_changes"]:
            lines.append(f"- `{row['timestamp']}` / `{row['surface']}` / {row['question']}")
            lines.append("```diff")
            lines.append(row["prompt_diff"] or "(empty)")
            lines.append("```")
    lines.append("")
    lines.append("## Largest Response Changes")
    if not summary["largest_response_changes"]:
        lines.append("- No entries")
    else:
        for row in summary["largest_response_changes"]:
            lines.append(f"- `{row['timestamp']}` / `{row['surface']}` / {row['question']}")
            lines.append("```diff")
            lines.append(row["response_diff_from_previous"] or "(empty)")
            lines.append("```")
    return "\n".join(lines) + "\n"
