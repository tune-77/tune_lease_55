#!/usr/bin/env python3
"""Build a deterministic introspection report from local continuity artifacts.

This is intentionally not an LLM diary generator. It checks whether the system
has actually reflected on recent work, whether the loop is getting stale, and
what should change next.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary as build_prompt_summary, load_jsonl

REPORTS_DIR = REPO_ROOT / "reports"
MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "introspection_latest.json"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "introspection_latest.md"
DEFAULT_LATEST_REPORT = REPORTS_DIR / "latest.json"
DEFAULT_RECURSIVE_REPORT = REPORTS_DIR / "recursive_self_improvement_latest.json"
DEFAULT_LOOP_REPORT = REPORTS_DIR / "loop_engineering_latest.json"
DEFAULT_MEMORY = REPO_ROOT / "MEMORY.md"

BOREDOM_TERMS = (
    "退屈",
    "つまらない",
    "停滞",
    "同じ",
    "繰り返し",
    "ワンパターン",
    "内省されていない",
    "内省がない",
)

REFLECTION_TERMS = (
    "内省",
    "違和感",
    "失敗",
    "学んだ",
    "変わった",
    "反省",
    "次の行動",
    "Promotable Items",
)

ACTION_TERMS = (
    "次の行動",
    "Next",
    "TODO",
    "確認する",
    "見る",
    "直す",
    "作る",
    "実装",
)


def _read_text(path: Path, max_chars: int = 20000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, False
    return (data, True) if isinstance(data, dict) else ({}, False)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(text.count(term) for term in terms)


def _daily_paths(today: dt.date, days: int) -> list[Path]:
    return [MEMORY_DIR / f"{today - dt.timedelta(days=i)}.md" for i in range(days)]


def _keyword_counter(text: str) -> Counter[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[一-龥ぁ-んァ-ン]{2,}", text)
    stopwords = {
        "こと",
        "ため",
        "よう",
        "する",
        "した",
        "ある",
        "ない",
        "これ",
        "それ",
        "改善",
        "内省",
        "レポート",
        "システム",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "should",
        "next",
        "action",
    }
    return Counter(token for token in tokens if token not in stopwords)


def _top_repeated_terms(texts: list[str], limit: int = 8) -> list[dict[str, Any]]:
    counter = Counter()
    for text in texts:
        counter.update(_keyword_counter(text))
    return [{"term": term, "count": count} for term, count in counter.most_common(limit)]


def _extract_promotable_items(text: str) -> list[str]:
    match = re.search(r"##\s*Promotable Items\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if not match:
        return []
    items: list[str] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _score_daily_note(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    exists = bool(text)
    reflection_hits = _count_terms(text, REFLECTION_TERMS)
    action_hits = _count_terms(text, ACTION_TERMS)
    boredom_hits = _count_terms(text, BOREDOM_TERMS)
    promotable_items = _extract_promotable_items(text)
    return {
        "path": str(path),
        "date": path.stem,
        "exists": exists,
        "chars": len(text),
        "reflection_hits": reflection_hits,
        "action_hits": action_hits,
        "boredom_hits": boredom_hits,
        "promotable_items": promotable_items,
    }


def _build_findings(
    daily_notes: list[dict[str, Any]],
    latest_report: dict[str, Any],
    recursive_report: dict[str, Any],
    loop_report: dict[str, Any],
    prompt_summary: dict[str, Any],
    *,
    latest_available: bool,
    recursive_available: bool,
    loop_available: bool,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    existing_notes = [note for note in daily_notes if note["exists"]]
    reflected_days = sum(1 for note in existing_notes if note["reflection_hits"] > 0)
    action_days = sum(1 for note in existing_notes if note["action_hits"] > 0)
    boredom_days = sum(1 for note in existing_notes if note["boredom_hits"] > 0)

    if not existing_notes:
        findings.append(
            {
                "severity": "attention",
                "title": "日次メモがなく、内省材料がない",
                "detail": "memory/YYYY-MM-DD.md が直近範囲で見つからないため、継続的な振り返りが成立していない。",
            }
        )
    elif reflected_days < max(1, len(existing_notes) // 2):
        findings.append(
            {
                "severity": "attention",
                "title": "内省語彙が少ない",
                "detail": "日次メモに事実ログはあるが、違和感・失敗・学び・変化の記録が薄い。",
            }
        )

    if action_days < reflected_days:
        findings.append(
            {
                "severity": "warn",
                "title": "内省が次の行動に変換されていない",
                "detail": "振り返りらしき記述に対して、次に確認・修正する行動が少ない。",
            }
        )

    if boredom_days:
        findings.append(
            {
                "severity": "attention",
                "title": "退屈・停滞シグナルが出ている",
                "detail": "ユーザーまたはログに退屈化の兆候がある。数値集計だけでなく、何を変えるかを明示する必要がある。",
            }
        )

    needs_review = _safe_int(latest_report.get("needs_review_count"))
    applied = _safe_int(latest_report.get("applied_count"))
    if latest_available and needs_review > 0 and applied == 0:
        findings.append(
            {
                "severity": "warn",
                "title": "改善候補が観察だけで止まっている",
                "detail": f"needs_review={needs_review} に対して applied={applied}。棚卸しだけが増えると退屈になる。",
            }
        )

    if not recursive_available:
        findings.append(
            {
                "severity": "warn",
                "title": "再帰的自己改善レポートが欠けている",
                "detail": "recursive_self_improvement_latest.json がないため、改善結果が次の候補に戻っているか確認できない。",
            }
        )

    if loop_available and loop_report.get("status") != "ok":
        findings.append(
            {
                "severity": "info",
                "title": "ループ健全性に警告がある",
                "detail": f"loop_engineering status={loop_report.get('status')}。内省レポートでも同じ警告を拾う。",
            }
        )

    previous_diff_rate = _safe_float(prompt_summary.get("previous_diff_rate"))
    total = _safe_int(prompt_summary.get("total"))
    if total and previous_diff_rate < 10.0:
        findings.append(
            {
                "severity": "info",
                "title": "応答変化率が低い",
                "detail": f"prompt feedback の previous_diff_rate={previous_diff_rate}% 。PDCAが形式化している可能性がある。",
            }
        )

    if not findings:
        findings.append(
            {
                "severity": "ok",
                "title": "内省ループは最低限動いている",
                "detail": "日次メモ、改善候補、行動変換の観測が揃っている。",
            }
        )
    return findings


def _build_next_actions(findings: list[dict[str, str]]) -> list[str]:
    titles = {finding["title"] for finding in findings}
    actions: list[str] = []
    if "内省語彙が少ない" in titles:
        actions.append("日次メモに「違和感」「失敗」「学び」「次に変える行動」を最低1行ずつ残す")
    if "内省が次の行動に変換されていない" in titles:
        actions.append("各内省項目に、確認日または実装対象ファイルを1つ紐づける")
    if "退屈・停滞シグナルが出ている" in titles:
        actions.append("観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える")
    if "改善候補が観察だけで止まっている" in titles:
        actions.append("needs_review から低リスク1件だけ選び、適用または却下まで進める")
    if "再帰的自己改善レポートが欠けている" in titles:
        actions.append("日次改善パイプライン後に recursive_self_improvement_latest.json の生成を確認する")
    if not actions:
        actions.append("直近3日で一番繰り返している論点を1つ選び、次回の判断基準に昇格する")
    return actions


def build_introspection_report(
    *,
    today: dt.date | None = None,
    days: int = 7,
    latest_report_path: Path = DEFAULT_LATEST_REPORT,
    recursive_report_path: Path = DEFAULT_RECURSIVE_REPORT,
    loop_report_path: Path = DEFAULT_LOOP_REPORT,
    prompt_log_path: Path = DEFAULT_LOG_PATH,
    memory_path: Path = DEFAULT_MEMORY,
) -> dict[str, Any]:
    current_day = today or dt.date.today()
    daily_notes = [_score_daily_note(path) for path in _daily_paths(current_day, days)]
    daily_texts = [_read_text(Path(note["path"])) for note in daily_notes if note["exists"]]
    memory_text = _read_text(memory_path)

    latest_report, latest_available = _load_json(latest_report_path)
    recursive_report, recursive_available = _load_json(recursive_report_path)
    loop_report, loop_available = _load_json(loop_report_path)
    prompt_rows = load_jsonl(prompt_log_path)
    prompt_summary = build_prompt_summary(prompt_rows)

    findings = _build_findings(
        daily_notes,
        latest_report,
        recursive_report,
        loop_report,
        prompt_summary,
        latest_available=latest_available,
        recursive_available=recursive_available,
        loop_available=loop_available,
    )
    severity_order = {"attention": 3, "warn": 2, "info": 1, "ok": 0}
    max_severity = max((severity_order.get(item["severity"], 0) for item in findings), default=0)
    status = "attention" if max_severity >= 3 else "warn" if max_severity == 2 else "ok"

    existing_count = sum(1 for note in daily_notes if note["exists"])
    reflection_days = sum(1 for note in daily_notes if note["exists"] and note["reflection_hits"] > 0)
    action_days = sum(1 for note in daily_notes if note["exists"] and note["action_hits"] > 0)

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "window": {"end_date": current_day.isoformat(), "days": days},
        "sources": {
            "daily_notes_found": existing_count,
            "memory_path": str(memory_path),
            "latest_report": {"path": str(latest_report_path), "available": latest_available},
            "recursive_report": {"path": str(recursive_report_path), "available": recursive_available},
            "loop_report": {"path": str(loop_report_path), "available": loop_available},
            "prompt_feedback_log": {"path": str(prompt_log_path), "rows": len(prompt_rows)},
        },
        "metrics": {
            "reflection_days": reflection_days,
            "action_days": action_days,
            "boredom_hits": sum(note["boredom_hits"] for note in daily_notes),
            "promotable_item_count": sum(len(note["promotable_items"]) for note in daily_notes),
            "long_term_memory_reflection_hits": _count_terms(memory_text, REFLECTION_TERMS),
            "prompt_previous_diff_rate": _safe_float(prompt_summary.get("previous_diff_rate")),
        },
        "daily_notes": daily_notes,
        "repeated_terms": _top_repeated_terms(daily_texts + [memory_text], limit=8),
        "findings": findings,
        "next_actions": _build_next_actions(findings),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Introspection Report")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at']}`")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Window: {report['window']['end_date']} / {report['window']['days']} days")
    lines.append("")
    lines.append("## Metrics")
    metrics = report["metrics"]
    lines.append(f"- Daily notes found: {report['sources']['daily_notes_found']}")
    lines.append(f"- Reflection days: {metrics['reflection_days']}")
    lines.append(f"- Action days: {metrics['action_days']}")
    lines.append(f"- Boredom hits: {metrics['boredom_hits']}")
    lines.append(f"- Promotable items: {metrics['promotable_item_count']}")
    lines.append(f"- Prompt previous diff rate: {metrics['prompt_previous_diff_rate']}%")
    lines.append("")
    lines.append("## Findings")
    for finding in report["findings"]:
        lines.append(f"- `{finding['severity']}` {finding['title']}: {finding['detail']}")
    lines.append("")
    lines.append("## Repeated Terms")
    for item in report["repeated_terms"]:
        lines.append(f"- {item['term']}: {item['count']}")
    lines.append("")
    lines.append("## Next Actions")
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--today", type=lambda value: dt.date.fromisoformat(value), default=None)
    parser.add_argument("--latest-report", type=Path, default=DEFAULT_LATEST_REPORT)
    parser.add_argument("--recursive-report", type=Path, default=DEFAULT_RECURSIVE_REPORT)
    parser.add_argument("--loop-report", type=Path, default=DEFAULT_LOOP_REPORT)
    parser.add_argument("--prompt-log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_introspection_report(
        today=args.today,
        days=max(1, args.days),
        latest_report_path=args.latest_report.expanduser(),
        recursive_report_path=args.recursive_report.expanduser(),
        loop_report_path=args.loop_report.expanduser(),
        prompt_log_path=args.prompt_log.expanduser(),
        memory_path=args.memory.expanduser(),
    )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    write_outputs(
        report,
        output_json=args.output_json.expanduser(),
        output_md=args.output_md.expanduser(),
    )
    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
