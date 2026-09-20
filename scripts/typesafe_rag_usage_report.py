#!/usr/bin/env python3
"""Summarize Jev(TypeSafe) RAG filtering effectiveness from the usage log.

data/typesafe_rag_usage.jsonl（typesafe_rag_guard.py が
TYPESAFE_RAG_USAGE_LOG_PATH 設定時のみ追記）を読み、候補数に対して
どれだけ除外されたかを集計する。計測結果を読むだけの read-only レポート。
フィルタ閾値やスコアリングには一切触れない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "typesafe_rag_usage.jsonl"


def read_entries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            rows.append(entry)
    return rows


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    applied = [e for e in entries if e.get("status") == "applied"]
    fallback = [e for e in entries if e.get("status") == "fallback"]

    candidate_total = sum(int(e.get("candidate_count") or 0) for e in applied)
    accepted_total = sum(int(e.get("accepted_count") or 0) for e in applied)
    excluded_total = sum(int(e.get("excluded_count") or 0) for e in applied)
    input_tokens = sum(int((e.get("usage") or {}).get("input_tokens") or 0) for e in applied)
    output_tokens = sum(int((e.get("usage") or {}).get("output_tokens") or 0) for e in applied)

    error_counts: dict[str, int] = {}
    for entry in fallback:
        error_type = str(entry.get("error_type") or "unknown")
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

    return {
        "samples": len(entries),
        "applied_count": len(applied),
        "fallback_count": len(fallback),
        "candidate_total": candidate_total,
        "accepted_total": accepted_total,
        "excluded_total": excluded_total,
        "exclusion_rate": round(excluded_total / candidate_total, 3) if candidate_total else 0.0,
        "avg_candidate_count": round(candidate_total / len(applied), 2) if applied else 0.0,
        "avg_accepted_count": round(accepted_total / len(applied), 2) if applied else 0.0,
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "fallback_errors": sorted(error_counts.items(), key=lambda item: -item[1]),
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Jev (TypeSafe) RAG Usage",
        "",
        f"- Samples: {summary['samples']}",
        f"- Applied: {summary['applied_count']} / Fallback: {summary['fallback_count']}",
        f"- Candidates seen: {summary['candidate_total']}"
        f"（1回あたり平均 {summary['avg_candidate_count']}）",
        f"- Accepted: {summary['accepted_total']}"
        f"（1回あたり平均 {summary['avg_accepted_count']}）",
        f"- Excluded: {summary['excluded_total']}"
        f"（除外率 {summary['exclusion_rate'] * 100:.1f}%）",
        f"- TypeSafe usage tokens: input={summary['input_tokens_total']} "
        f"output={summary['output_tokens_total']}",
        "",
        "## Fallback errors",
        "",
        "| error_type | count |",
        "|---|---:|",
    ]
    lines.extend(f"| {error_type} | {count} |" for error_type, count in summary["fallback_errors"])
    if not summary["fallback_errors"]:
        lines.append("| (fallback なし) | 0 |")
    lines += [
        "",
        "除外率が低い場合は候補選定が既に良好なため、Jevの追加コストに見合っていない可能性がある。",
        "fallback_errors が多い場合は typesafe_rag_guard.py 側の到達性・タイムアウトを見直す。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--json", action="store_true", help="Markdown ではなく JSON で出力")
    args = parser.parse_args()

    entries = read_entries(Path(args.input))
    summary = summarize(entries)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render(summary))


if __name__ == "__main__":
    main()
