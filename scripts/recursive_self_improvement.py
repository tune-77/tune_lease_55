#!/usr/bin/env python3
"""Build a recursive self-improvement report from existing improvement artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prompt_feedback_metrics import DEFAULT_LOG_PATH, build_summary as build_prompt_summary, load_jsonl

_PIPELINE_ROOT = _REPO_ROOT / ".agents" / "skills" / "auto-improvement-pipeline"
_PIPELINE_SCRIPTS_DIR = _REPO_ROOT / ".agents" / "skills" / "auto-improvement-pipeline" / "scripts"
if _PIPELINE_ROOT.exists():
    sys.path.insert(0, str(_PIPELINE_ROOT))
if _PIPELINE_SCRIPTS_DIR.exists():
    sys.path.insert(0, str(_PIPELINE_SCRIPTS_DIR))

try:
    from auto_fix_policy import classify_quick_fix, evaluate_auto_fix_policy
    from improvement_deduplicator import deduplicate_improvements
    from improvement_identity import canonical_key
    from implementation_ranker import rank_improvements
    import pipeline_ledger
except ImportError as exc:  # pragma: no cover - import wiring failure is fatal
    raise SystemExit(f"failed to import pipeline helpers: {exc}")

REPORTS_DIR = _REPO_ROOT / "reports"
DEFAULT_OUTPUT_JSON = REPORTS_DIR / "recursive_self_improvement_latest.json"
DEFAULT_OUTPUT_MD = REPORTS_DIR / "recursive_self_improvement_latest.md"

_OBSIDIAN_MARKED_ITEM_RE = re.compile(r"^\[(?:改善|TODO)\]\s*(.+)$")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get(key, "")).strip()
        for key in ("title", "description", "detail", "reason", "target_module")
        if str(item.get(key, "")).strip()
    ]
    return " ".join(parts)


def _candidate_source(item: dict[str, Any], source: str, state: str, index: int) -> dict[str, Any]:
    title = str(item.get("title") or "").strip()
    description = str(item.get("description") or item.get("detail") or "").strip()
    reason = str(item.get("reason") or "").strip()
    canonical = canonical_key(title, description)
    enriched: dict[str, Any] = {
        "id": str(item.get("id") or f"{source.upper()}-{index:03d}"),
        "title": title or description or f"{source} item {index}",
        "description": description,
        "reason": reason,
        "detail": str(item.get("detail") or "").strip(),
        "source": source,
        "source_state": state,
        "target_module": str(item.get("target_module") or "").strip(),
        "canonical_key": canonical,
        "source_text": _candidate_text(item),
    }
    if item.get("auto_fix_policy"):
        enriched["auto_fix_policy"] = item["auto_fix_policy"]
    return enriched


def _parse_obsidian_note(text: str, source_name: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _OBSIDIAN_MARKED_ITEM_RE.match(lines[i].strip())
        if not match:
            i += 1
            continue
        title = match.group(1).strip()
        reason = ""
        if i + 1 < len(lines) and lines[i + 1].startswith("理由："):
            reason = lines[i + 1][3:].strip()
            i += 1
        items.append(
            {
                "id": f"OBS-{len(items) + 1:03d}",
                "title": title,
                "description": reason,
                "reason": reason,
                "detail": reason,
                "source": source_name,
                "source_state": "candidate",
            }
        )
        i += 1
    return items


def _load_obsidian_notes(paths: list[Path]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            md_files = sorted(path.glob("*.md"))
            for md_file in md_files:
                try:
                    content = md_file.read_text(encoding="utf-8")
                except Exception:
                    continue
                notes.append({"title": md_file.stem, "body": content, "path": str(md_file)})
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        notes.append({"title": path.stem, "body": content, "path": str(path)})
    return notes


def _collect_raw_items(report: dict[str, Any], obsidian_items: list[dict[str, Any]], workspace_root: Path) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []

    for key, state in (("applied", "applied"), ("needs_review", "validated")):
        for index, item in enumerate(report.get(key) or [], 1):
            if not isinstance(item, dict):
                continue
            enriched = _candidate_source(item, key, state, index)
            _attach_quick_fix_target(enriched, workspace_root)
            enriched["auto_fix_policy"] = evaluate_auto_fix_policy(enriched, workspace_root)
            raw.append(enriched)

    for index, item in enumerate(obsidian_items, 1):
        if not isinstance(item, dict):
            continue
        enriched = _candidate_source(item, "obsidian", "candidate", index)
        _attach_quick_fix_target(enriched, workspace_root)
        enriched["auto_fix_policy"] = evaluate_auto_fix_policy(enriched, workspace_root)
        raw.append(enriched)

    return raw


def _attach_quick_fix_target(enriched: dict[str, Any], workspace_root: Path) -> None:
    """quick_fix と判定された候補に、推定された対象ファイルと quick_ui 分類を付与する。

    classify_quick_fix（auto_fix_policy）を候補生成段に配線し、抽象的な要望から
    推定した単一対象ファイルを候補へ明示的に載せる。これにより ranked_queue・
    Codexキュー・レポートが具体的な対象ファイルを保持でき、自動修正が実行可能になる。
    """
    verdict = classify_quick_fix(enriched, workspace_root)
    if not verdict.get("is_quick_fix"):
        return
    target = verdict.get("target_module")
    if target and not enriched.get("target_module"):
        enriched["target_module"] = target
    implementation = dict(enriched.get("implementation") or {})
    implementation.setdefault("category", "quick_ui")
    enriched["implementation"] = implementation


def _enrich_ranked_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked, recommended_order = rank_improvements(candidates)
    order_map = {str(item.get("canonical_key") or ""): item for item in recommended_order}
    for item in ranked:
        canonical = str(item.get("canonical_key") or canonical_key(str(item.get("title") or ""), str(item.get("description") or "")))
        order = order_map.get(canonical, {}).get("order")
        item["recommended_order"] = order if isinstance(order, int) else None
        item["priority_score"] = order_map.get(canonical, {}).get("priority_score", item.get("implementation", {}).get("priority_score", 0))
    return ranked


def _build_suppression_reason(status: str, duplicate_count: int, processed: bool) -> str:
    reasons: list[str] = []
    if processed:
        reasons.append(f"ledger={status}")
    if duplicate_count > 0:
        reasons.append(f"duplicate_count={duplicate_count}")
    return ", ".join(reasons) if reasons else "noise"


def build_recursive_self_improvement(
    report: dict[str, Any],
    *,
    prompt_feedback_log: list[dict[str, Any]] | None = None,
    obsidian_notes: list[dict[str, Any]] | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Build the recursive improvement bundle without mutating external state."""
    root = workspace_root or _REPO_ROOT
    obsidian_items = []
    for note in obsidian_notes or []:
        if not isinstance(note, dict):
            continue
        body = str(note.get("body") or note.get("content") or note.get("text") or "").strip()
        if body:
            source_name = str(note.get("title") or note.get("path") or note.get("source") or "obsidian_note")
            obsidian_items.extend(_parse_obsidian_note(body, source_name))

    raw_items = _collect_raw_items(report, obsidian_items, root)
    deduped, grouped = deduplicate_improvements(raw_items)
    ranked_candidates = _enrich_ranked_candidates(deduped)

    canonical_candidates: list[dict[str, Any]] = []
    ranked_queue: list[dict[str, Any]] = []
    suppressions: list[dict[str, Any]] = []
    ledger_events: list[dict[str, Any]] = []
    reused_count = 0
    repeat_count = 0
    queued_count = 0

    for item in ranked_candidates:
        policy = evaluate_auto_fix_policy(item, root)
        item["auto_fix_policy"] = policy
        key = str(item.get("canonical_key") or "")
        processed, ledger_status = pipeline_ledger.is_processed(key)
        duplicate_count = int(item.get("duplicate_count") or 0)
        if duplicate_count > 0:
            repeat_count += 1
        if processed:
            reused_count += 1

        if processed and ledger_status:
            state = "suppressed"
            reason = _build_suppression_reason(ledger_status, duplicate_count, True)
            # 台帳で既に決着済み（applied/deleted/rejected）の抑制は健全な重複排除。
            # 一方 needs_review/parked/suppressed のクールダウン抑制は「滞留(churn)」で、
            # ループが前へ進んでいないサイン。両者を区別して noise の誤読を防ぐ。
            healthy = ledger_status.startswith(("applied", "deleted", "rejected"))
            suppressions.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "canonical_key": key,
                    "status": state,
                    "reason": reason,
                    "healthy": healthy,
                }
            )
        elif item.get("source_state") == "applied":
            state = "applied"
        elif policy.get("auto_fix_allowed"):
            state = "validated"
            queued_count += 1
            ranked_queue.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "canonical_key": key,
                    "target_module": item.get("target_module")
                    or policy.get("inferred_target_module")
                    or "",
                    "category": item.get("implementation", {}).get("category", ""),
                    "effort": item.get("implementation", {}).get("effort", 0),
                    "risk": item.get("implementation", {}).get("risk", 0),
                    "impact": item.get("implementation", {}).get("impact", 0),
                    "priority_score": item.get("implementation", {}).get("priority_score", 0),
                    "recommended_order": item.get("recommended_order"),
                    "auto_fix_policy": policy,
                    "source_state": item.get("source_state"),
                }
            )
        else:
            state = "needs_review"

        item["state"] = state
        canonical_candidates.append(item)
        ledger_events.append(
            {
                "key": key,
                "status": state,
                "canonical_key": key,
                "title": item.get("title"),
                "reason": policy.get("reason") or item.get("reason") or "",
                "source": item.get("source"),
            }
        )

    prompt_rows = prompt_feedback_log or []
    prompt_summary = build_prompt_summary(prompt_rows)
    total = len(canonical_candidates)
    suppressed_healthy = sum(1 for s in suppressions if s.get("healthy"))
    suppressed_churn = len(suppressions) - suppressed_healthy
    measurement_summary = {
        "pdca_rate": prompt_summary.get("pdca_rate", 0.0),
        "response_changed_rate": prompt_summary.get("previous_diff_rate", 0.0),
        "repeat_issue_rate": round(repeat_count / total * 100, 1) if total else 0.0,
        "reuse_rate": round(reused_count / total * 100, 1) if total else 0.0,
        # noise_rate は後方互換のため「全抑制/総数」を維持。健全な重複排除も含むため、
        # ループ滞留の判定には churn_rate（クールダウン滞留のみ）を使うこと。
        "noise_rate": round(len(suppressions) / total * 100, 1) if total else 0.0,
        "churn_rate": round(suppressed_churn / total * 100, 1) if total else 0.0,
        "suppressed_healthy_count": suppressed_healthy,
        "suppressed_churn_count": suppressed_churn,
        "prompt_total": prompt_summary.get("total", 0),
        "prompt_previous_diff_count": prompt_summary.get("previous_diff_count", 0),
    }

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "date": dt.date.today().isoformat(),
        "source_report": report.get("source", ""),
        "source_report_date": report.get("date") or report.get("generated_at", ""),
        "canonical_candidate_count": total,
        "ranked_queue_count": len(ranked_queue),
        "suppressed_count": len(suppressions),
        "grouped_improvements": grouped,
        "canonical_candidates": canonical_candidates,
        "ranked_queue": ranked_queue,
        "suppressions": suppressions,
        "ledger_events": ledger_events,
        "measurement_summary": measurement_summary,
        "prompt_feedback_summary": prompt_summary,
        "input_counts": {
            "report_applied": len(report.get("applied") or []),
            "report_needs_review": len(report.get("needs_review") or []),
            "obsidian_notes": len(obsidian_items),
            "prompt_feedback_rows": len(prompt_rows),
        },
    }


def render_markdown(bundle: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Recursive Self-Improvement Report")
    lines.append("")
    lines.append(f"- Generated at: `{bundle['generated_at']}`")
    lines.append(f"- Source report: `{bundle.get('source_report') or ''}`")
    lines.append(f"- Canonical candidates: {bundle['canonical_candidate_count']}")
    lines.append(f"- Ranked queue: {bundle['ranked_queue_count']}")
    lines.append(f"- Suppressed: {bundle['suppressed_count']}")
    lines.append("")
    lines.append("## Measurement")
    measurement = bundle["measurement_summary"]
    lines.append(f"- PDCA rate: {measurement['pdca_rate']}%")
    lines.append(f"- Response changed rate: {measurement['response_changed_rate']}%")
    lines.append(f"- Repeat issue rate: {measurement['repeat_issue_rate']}%")
    lines.append(f"- Reuse rate: {measurement['reuse_rate']}%")
    lines.append(f"- Noise rate: {measurement['noise_rate']}%")
    lines.append(
        f"- Churn rate: {measurement.get('churn_rate', 0.0)}% "
        f"(healthy dedup: {measurement.get('suppressed_healthy_count', 0)}, "
        f"churn: {measurement.get('suppressed_churn_count', 0)})"
    )
    lines.append("")
    lines.append("## Ranked Queue")
    if not bundle["ranked_queue"]:
        lines.append("- No auto-fix candidates")
    else:
        for item in bundle["ranked_queue"][:10]:
            lines.append(
                f"- `{item.get('id')}` `{item.get('title')}` "
                f"(order={item.get('recommended_order')}, score={item.get('priority_score')})"
            )
    lines.append("")
    lines.append("## Suppressions")
    if not bundle["suppressions"]:
        lines.append("- No suppressed items")
    else:
        for item in bundle["suppressions"][:10]:
            lines.append(f"- `{item.get('id')}` `{item.get('title')}` / {item.get('reason')}")
    return "\n".join(lines) + "\n"


def record_ledger_events(events: list[dict[str, Any]]) -> int:
    """ledger_events を台帳へ記録する。記録した件数を返す。

    重要: status=="suppressed" のイベントは記録しない。
    suppressed は「既に台帳にある候補をクールダウンで抑制した」という一時的な観測結果
    であって新しい決定ではない。これを毎回記録し直すと、pipeline_ledger.is_processed が
    見る最新エントリの recorded_at が日々更新され、needs_review/suppressed の30日
    クールダウンが永久にリセットされて恒久抑制に陥る（applied=0 の空回りの原因）。
    決定を表す状態（applied/validated/needs_review/rejected 等）のみ記録する。
    """
    recorded = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        status = str(event.get("status") or "")
        if status == "suppressed":
            continue
        pipeline_ledger.record(
            str(event.get("key") or event.get("canonical_key") or ""),
            status,
            str(event.get("title") or ""),
            reason=str(event.get("reason") or ""),
            canonical_key=str(event.get("canonical_key") or ""),
        )
        recorded += 1
    return recorded


def write_recursive_outputs(
    bundle: dict[str, Any],
    *,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    latest_json: Path | None = None,
    latest_md: Path | None = None,
    augment_report: Path | None = None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    _dump_json(output_json, bundle)
    output_md.write_text(render_markdown(bundle), encoding="utf-8")

    if latest_json is not None:
        _dump_json(latest_json, bundle)
    if latest_md is not None:
        latest_md.write_text(render_markdown(bundle), encoding="utf-8")

    if augment_report is not None and augment_report.exists():
        report = _load_json(augment_report)
        report["recursive_self_improvement"] = {
            "generated_at": bundle["generated_at"],
            "path": str(output_json),
            "canonical_candidate_count": bundle["canonical_candidate_count"],
            "ranked_queue_count": bundle["ranked_queue_count"],
            "suppressed_count": bundle["suppressed_count"],
            "measurement_summary": bundle["measurement_summary"],
        }
        _dump_json(augment_report, report)


def _load_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        candidates = sorted(REPORTS_DIR.glob("improvement_report_*.json"))
        if not candidates:
            return {}
        path = candidates[-1]
    report = _load_json(path)
    if report:
        report.setdefault("source", str(path))
    return report


CHAT_INTAKE_PATH = _REPO_ROOT / "data" / "chat_quick_fix_intake.jsonl"
CHAT_INTAKE_EXECUTED_PATH = _REPO_ROOT / "data" / "chat_quick_fix_executed.json"


def _load_chat_quick_fix_executed_ids(path: Path = CHAT_INTAKE_EXECUTED_PATH) -> set[str]:
    """execute_chat_quick_fix.start_execution がバックグラウンド即時実行した候補IDの集合。

    ここに含まれるIDは、日次バッチ実行前にすでに execute_codex_queue 相当が完了
    （成功/失敗いずれも）しているため、needs_review へ再度乗せると二重実行になる。
    """
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(x) for x in data} if isinstance(data, list) else set()
    except (OSError, json.JSONDecodeError):
        return set()


def load_chat_quick_fix_intake(
    path: Path | None = None, executed_ids_path: Path | None = None
) -> list[dict[str, Any]]:
    """チャットで紫苑が起票した quick_fix 提案を needs_review 候補として読み込む。

    propose_quick_fix ツール（lease_intelligence_tools）が追記する JSONL を、
    自律改善パイプラインの候補源へ取り込むための入口。欠損時は空リスト。
    ledger のクールダウンで重複起票は自動的に抑制されるため、追記形式のまま扱う。
    ただし execute_chat_quick_fix によりすでに即時実行済みのIDは除外する
    （二重実行防止。実行結果は codex_queue_result_*_chat.json 側にある）。
    """
    intake_path = path or CHAT_INTAKE_PATH
    if not intake_path.exists():
        return []
    executed_ids = _load_chat_quick_fix_executed_ids(executed_ids_path or CHAT_INTAKE_EXECUTED_PATH)
    items: list[dict[str, Any]] = []
    for line in intake_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        title = str(record.get("title") or "").strip()
        if not title:
            continue
        if str(record.get("id") or "") in executed_ids:
            continue
        items.append({
            "id": str(record.get("id") or ""),
            "title": title,
            "description": str(record.get("description") or "").strip(),
            "target_module": str(record.get("target_module") or "").strip(),
            # 元の source（例: shion_promise）を保持し、改善ログUIで出所を辿れるようにする。
            # 未指定のチャット quick_fix は従来どおり chat_quick_fix。
            "source": str(record.get("source") or "chat_quick_fix"),
        })
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None, help="Improvement report JSON path.")
    parser.add_argument("--prompt-log", type=Path, default=DEFAULT_LOG_PATH, help="Prompt feedback JSONL log path.")
    parser.add_argument(
        "--obsidian-note",
        type=Path,
        action="append",
        default=[],
        help="Obsidian markdown file or directory to include as improvement candidates.",
    )
    parser.add_argument("--workspace-root", type=Path, default=_REPO_ROOT, help="Workspace root for policy checks.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Recursive report JSON path.")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD, help="Recursive report markdown path.")
    parser.add_argument("--latest-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Latest recursive JSON path.")
    parser.add_argument("--latest-md", type=Path, default=DEFAULT_OUTPUT_MD, help="Latest recursive markdown path.")
    parser.add_argument(
        "--augment-report",
        type=Path,
        default=None,
        help="Optional report path to annotate with recursive metadata.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs or ledger entries.")
    args = parser.parse_args()

    report = _load_report(args.report)
    chat_intake = load_chat_quick_fix_intake()
    if chat_intake:
        report.setdefault("needs_review", [])
        report["needs_review"].extend(chat_intake)
    prompt_rows = load_jsonl(args.prompt_log.expanduser())
    obsidian_notes = _load_obsidian_notes([path.expanduser() for path in args.obsidian_note])
    bundle = build_recursive_self_improvement(
        report,
        prompt_feedback_log=prompt_rows,
        obsidian_notes=obsidian_notes,
        workspace_root=args.workspace_root.expanduser(),
    )

    if args.dry_run:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0

    record_ledger_events(bundle.get("ledger_events", []))

    write_recursive_outputs(
        bundle,
        output_json=args.output_json.expanduser(),
        output_md=args.output_md.expanduser(),
        latest_json=args.latest_json.expanduser() if args.latest_json else None,
        latest_md=args.latest_md.expanduser() if args.latest_md else None,
        augment_report=args.augment_report.expanduser() if args.augment_report else None,
    )

    print(f"saved: {args.output_json.expanduser()}")
    print(f"saved: {args.output_md.expanduser()}")
    if args.latest_json:
        print(f"latest: {args.latest_json.expanduser()}")
    if args.latest_md:
        print(f"latest: {args.latest_md.expanduser()}")
    if args.augment_report:
        print(f"augmented: {args.augment_report.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
