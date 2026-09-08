"""Utilities for reading and summarizing prompt feedback logs."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from runtime_paths import get_data_dir

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = get_data_dir() / "prompt_feedback_log.jsonl"
DEFAULT_HUMAN_FEEDBACK_PATH = get_data_dir() / "human_response_feedback.jsonl"

USEFULNESS_RATINGS = {"good", "thin", "bad"}
DEFAULT_REPORT_TIMEZONE = ZoneInfo("Asia/Tokyo")


def filter_rows_by_month(
    rows: list[dict[str, Any]],
    month: str,
    *,
    timestamp_keys: tuple[str, ...] = ("timestamp", "ts"),
    report_timezone: ZoneInfo = DEFAULT_REPORT_TIMEZONE,
) -> list[dict[str, Any]]:
    """Return rows belonging to YYYY-MM in the report's local timezone.

    Offset-aware timestamps (including Cloud Run's UTC values) are converted to
    the report timezone. Naive timestamps are treated as already being local so
    existing locally-written prompt feedback keeps its historical semantics.
    """
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("month must be YYYY-MM")
    matched: list[dict[str, Any]] = []
    for row in rows:
        timestamp = next((str(row.get(key) or "") for key in timestamp_keys if row.get(key)), "")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(report_timezone)
        if f"{parsed.year:04d}-{parsed.month:02d}" == month:
            matched.append(row)
    return matched


def build_human_feedback_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize explicit response buttons separately from identity feedback."""
    usefulness_rows = [row for row in rows if str(row.get("rating") or "") in USEFULNESS_RATINGS]
    rating_counts = Counter(str(row.get("rating") or "unknown") for row in rows)
    usefulness_counts = Counter(str(row.get("rating") or "") for row in usefulness_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usefulness_rows:
        grouped[str(row.get("route") or "unknown")].append(row)

    by_route: dict[str, dict[str, Any]] = {}
    for route, items in grouped.items():
        counts = Counter(str(row.get("rating") or "") for row in items)
        improvement_count = counts["thin"] + counts["bad"]
        by_route[route] = {
            "count": len(items),
            "good": counts["good"],
            "thin": counts["thin"],
            "bad": counts["bad"],
            "useful_rate": round(counts["good"] / len(items) * 100, 1),
            "improvement_rate": round(improvement_count / len(items) * 100, 1),
            "priority_score": counts["bad"] * 2 + counts["thin"],
        }

    priorities = [
        {
            "route": route,
            "priority_score": stats["priority_score"],
            "thin": stats["thin"],
            "bad": stats["bad"],
            "reason": (
                f"意図違い {stats['bad']}件を優先確認し、薄い回答 {stats['thin']}件の具体性を補う"
                if stats["bad"]
                else f"薄い回答 {stats['thin']}件の根拠・次の行動を補う"
            ),
        }
        for route, stats in by_route.items()
        if stats["priority_score"] > 0
    ]
    priorities.sort(key=lambda item: (-item["priority_score"], item["route"]))

    negative_examples: list[dict[str, str]] = []
    for row in reversed(usefulness_rows):
        rating = str(row.get("rating") or "")
        if rating not in {"thin", "bad"}:
            continue
        response_lines = str(row.get("response_start") or "").strip().splitlines()
        negative_examples.append({
            "timestamp": str(row.get("ts") or row.get("timestamp") or ""),
            "route": str(row.get("route") or "unknown"),
            "rating": rating,
            "question": str(row.get("message_preview") or "").strip()[:160],
            "response_start": response_lines[0][:180] if response_lines else "",
            "comment": str(row.get("comment") or "").strip()[:180],
        })
        if len(negative_examples) >= 5:
            break

    usefulness_total = len(usefulness_rows)
    improvement_count = usefulness_counts["thin"] + usefulness_counts["bad"]
    return {
        "total": len(rows),
        "rating_counts": dict(rating_counts),
        "usefulness_total": usefulness_total,
        "good_count": usefulness_counts["good"],
        "thin_count": usefulness_counts["thin"],
        "bad_count": usefulness_counts["bad"],
        "useful_rate": round(usefulness_counts["good"] / usefulness_total * 100, 1) if usefulness_total else 0.0,
        "improvement_count": improvement_count,
        "improvement_rate": round(improvement_count / usefulness_total * 100, 1) if usefulness_total else 0.0,
        "identity_feedback_total": rating_counts["shion_like"] + rating_counts["not_shion"],
        "by_route": by_route,
        "improvement_priorities": priorities,
        "negative_examples": negative_examples,
    }


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


def render_markdown(
    summary: dict[str, Any],
    source: Path,
    human_feedback: dict[str, Any] | None = None,
    human_source: Path | None = None,
) -> str:
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
    if human_feedback is not None:
        lines.append("## 回答ボタン評価")
        lines.append("")
        if human_source is not None:
            lines.append(f"- Source: `{human_source}`")
        lines.append(f"- 評価数: {human_feedback['usefulness_total']}件")
        lines.append(f"- 効いた: {human_feedback['good_count']}件 ({human_feedback['useful_rate']}%)")
        lines.append(f"- 微妙: {human_feedback['thin_count']}件")
        lines.append(f"- 違う: {human_feedback['bad_count']}件")
        lines.append(f"- 要改善率: {human_feedback['improvement_rate']}%")
        lines.append("")
        lines.append("### 画面別")
        if not human_feedback["by_route"]:
            lines.append("- ボタン評価はまだありません")
        else:
            for route, stats in sorted(
                human_feedback["by_route"].items(),
                key=lambda kv: (-kv[1]["priority_score"], -kv[1]["count"], kv[0]),
            ):
                lines.append(
                    f"- `{route}`: {stats['count']}件 / 効いた {stats['good']} / "
                    f"微妙 {stats['thin']} / 違う {stats['bad']} / 有用率 {stats['useful_rate']}%"
                )
        lines.append("")
        lines.append("### 改善優先順位")
        if not human_feedback["improvement_priorities"]:
            lines.append("- 要改善評価はまだありません")
        else:
            for index, item in enumerate(human_feedback["improvement_priorities"][:5], start=1):
                lines.append(f"{index}. `{item['route']}` — {item['reason']}")
        if human_feedback["negative_examples"]:
            lines.append("")
            lines.append("### 要改善回答の例")
            for item in human_feedback["negative_examples"]:
                label = "違う" if item["rating"] == "bad" else "微妙"
                lines.append(f"- `{item['route']}` / {label} / 質問: {item['question'] or '(記録なし)'}")
                if item["response_start"]:
                    lines.append(f"  - 回答冒頭: {item['response_start']}")
                if item["comment"]:
                    lines.append(f"  - コメント: {item['comment']}")
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
