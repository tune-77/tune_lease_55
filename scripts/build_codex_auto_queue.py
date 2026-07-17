#!/usr/bin/env python3
"""Build a small Codex auto-improvement queue from the latest report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any


LEDGER_RULES_PATH = "api/rule_engine/ledger_rules.json"
PIPELINE_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "auto-improvement-pipeline" / "scripts"
if str(PIPELINE_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SCRIPTS_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from auto_fix_policy import evaluate_auto_fix_policy  # type: ignore[import]
except ImportError:
    evaluate_auto_fix_policy = None  # type: ignore[assignment]

from shion_triage import (  # noqa: E402
    TRIAGE_MODES,
    is_approved_today,
    load_triage_latest,
    resolve_triage_mode,
    triage_record_for_item,
)

BLOCKED_KEYWORDS = [
    "db",
    "api連携",
    "帝国データバンク",
    "モデル",
    "ocr",
    "ポートフォリオ",
    "公平性",
    "バイアス",
    "migration",
    "kubernetes",
    "スコアリング",
    "認証",
    "セキュリティ",
]

SAFE_TITLE_KEYWORDS = [
    "表示",
    "文言",
    "ホーム",
    "入力",
    "質問",
    "マウスオーバー",
    "初期設定",
    "整理",
    "参照",
    "音声入力",
    "背景",
    "フローティングUI",
    "補助金",
]

EXECUTION_STATUS_FILE = "codex_auto_execution_status.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dynamic_blocked_keywords(root: Path, days: int = 7) -> list[str]:
    """直近 days 日の codex_queue_result_*.json から3回以上失敗したアイテムのタイトルキーワードを返す."""
    cutoff = (dt.date.today() - dt.timedelta(days=days)).strftime("%Y%m%d")
    fail_counts: dict[str, int] = {}
    for path in sorted((root / "reports").glob("codex_queue_result_*.json")):
        m = re.search(r"codex_queue_result_(\d{8})", path.name)
        if m and m.group(1) < cutoff:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data.get("results") or []:
            if not isinstance(item, dict):
                continue
            if item.get("exit_code", 0) != 0:
                title = str(item.get("title") or "")
                if title:
                    fail_counts[title] = fail_counts.get(title, 0) + 1

    dynamic: list[str] = []
    for title, count in fail_counts.items():
        if count < 3:
            continue
        segs = re.split(r"[のをがにではともやへから・ 　]+", title)
        for seg in segs:
            kw = seg.strip()
            if len(kw) >= 2 and kw not in BLOCKED_KEYWORDS and kw not in dynamic:
                dynamic.append(kw)
    return dynamic


def _batch_applied_targets_today(root: Path) -> frozenset[str]:
    """今日 batch_apply が適用したルールの target ファイルパス・basename の集合を返す。

    ledger_rules.json の applied_at フィールド（UTC タイムスタンプ）が
    今日の日付で始まるルールを対象とする。
    batch_apply.py は専用ログを出力しないため、ledger_rules.json を唯一の記録源とする。
    """
    ledger_path = root / LEDGER_RULES_PATH
    if not ledger_path.exists():
        return frozenset()
    try:
        rules: list[dict[str, Any]] = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception:
        return frozenset()
    today = dt.date.today().isoformat()  # "YYYY-MM-DD"
    targets: set[str] = set()
    for rule in rules:
        applied_at = str(rule.get("applied_at") or "")
        if not applied_at.startswith(today):
            continue
        target = str(rule.get("target") or "").strip()
        if target:
            targets.add(target)           # 相対パス全体 (例: api/routers/ocr.py)
            targets.add(Path(target).name)  # ファイル名のみ (例: ocr.py)
    return frozenset(targets)


def is_batch_apply_touched(
    item: dict[str, Any], applied_targets: frozenset[str]
) -> tuple[bool, str]:
    """今日の batch_apply が触れたファイルに関係する REV かどうかを判定する。"""
    if not applied_targets:
        return False, ""
    text = item_text(item)
    for target in sorted(applied_targets):  # ソートで決定的な出力
        if target.lower() in text:
            return True, f"batch_apply 適用済みファイル: {target}"
    return False, ""


def latest_report_path(root: Path) -> Path:
    reports = sorted((root / "reports").glob("improvement_report_*.json"))
    if not reports:
        raise SystemExit("No reports/improvement_report_*.json files found.")
    return reports[-1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_execution_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    try:
        data = load_json(path)
    except Exception:
        return {"items": {}}
    if not isinstance(data.get("items"), dict):
        data["items"] = {}
    return data


def quota_blocked_items(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = status.get("items") or {}
    blocked: dict[str, dict[str, Any]] = {}
    for rev_id, record in items.items():
        if not isinstance(record, dict):
            continue
        if record.get("status") == "blocked_by_quota":
            blocked[str(rev_id)] = record
    return blocked


def item_text(item: dict[str, Any]) -> str:
    policy = item.get("auto_fix_policy") or {}
    return " ".join(
        str(value or "")
        for value in (
            item.get("id"),
            item.get("title"),
            item.get("reason"),
            item.get("detail"),
            item.get("description"),
            policy.get("reason"),
        )
    ).lower()


def is_blocked(item: dict[str, Any], extra_keywords: list[str] | None = None) -> tuple[bool, str]:
    policy = item.get("auto_fix_policy") or {}
    risk = str(policy.get("risk") or "").lower()
    max_files = policy.get("max_files")
    text = item_text(item)
    all_keywords = BLOCKED_KEYWORDS + (extra_keywords or [])
    hits = [keyword for keyword in all_keywords if keyword.lower() in text]
    if risk == "high":
        return True, "risk=high"
    if max_files == 0:
        return True, "max_files=0"
    if hits:
        return True, "blocked_keyword: " + ", ".join(hits[:5])
    return False, ""


def is_codex_safe(item: dict[str, Any]) -> bool:
    policy = item.get("auto_fix_policy") or {}
    if policy.get("auto_fix_allowed") is not True:
        return False
    risk = str(policy.get("risk") or "").lower()
    max_files = policy.get("max_files")
    title = str(item.get("title") or "")
    if risk != "low":
        return False
    if max_files is not None and max_files > 1:
        return False
    return any(keyword in title for keyword in SAFE_TITLE_KEYWORDS)


def refresh_auto_fix_policy(item: dict[str, Any], root: Path) -> dict[str, Any]:
    """古いレポートでも、現在の auto_fix_policy で安全候補を再判定する."""
    if evaluate_auto_fix_policy is None:
        return item

    refreshed = dict(item)
    policy = evaluate_auto_fix_policy(refreshed, root)
    refreshed["auto_fix_policy"] = policy
    inferred_target = policy.get("inferred_target_module")
    if inferred_target and not refreshed.get("target_module"):
        refreshed["target_module"] = inferred_target
    return refreshed


def queue_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    order = item.get("recommended_order")
    if not isinstance(order, int):
        order = item.get("_source_index")
    if not isinstance(order, int):
        order = 9999
    return (order, str(item.get("id") or ""))


def queue_item(item: dict[str, Any]) -> dict[str, Any]:
    policy = item.get("auto_fix_policy") or {}
    title = str(item.get("title") or "")
    target_module = item.get("target_module") or policy.get("inferred_target_module") or ""
    return {
        "id": item.get("id"),
        "title": title,
        "reason": item.get("reason") or policy.get("reason") or "",
        "detail": item.get("detail") or item.get("description") or "",
        "target_module": target_module,
        "risk": policy.get("risk") or "",
        "max_files": policy.get("max_files"),
        "required_checks": policy.get("required_checks") or ["py_compile", "targeted_test"],
        "mode": "codex_dry_run_first",
        "prompt": f"{item.get('id')} {title} を {target_module or '対象ファイル推定'} に小さく実装し、テスト後に差分を報告してください。data/models/.claude/state は触らないでください。",
        "triage_decision": str(item.get("triage_decision") or ""),
        "user_approved": bool(item.get("user_approved")),
    }


def _triage_sort_key(item: dict[str, Any], record: dict | None) -> tuple[int, int, str]:
    """トリアージ反映後の並び順（P2-1）。今日やる(承認済み) > 今日やる > その他。"""
    if is_approved_today(record):
        rank = 0
    elif record and str(record.get("decision") or "") == "today":
        rank = 1
    else:
        rank = 2
    base = queue_sort_key(item)
    return (rank, base[0], base[1])


def _apply_triage_to_safe(
    safe_sorted: list[dict[str, Any]],
    triage: dict[str, dict],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """トリアージを適用した (queued, excluded_by_discard, promoted_ids) を返す。"""
    kept: list[tuple[dict[str, Any], dict | None]] = []
    excluded: list[dict[str, Any]] = []
    for item in safe_sorted:
        record = triage_record_for_item(triage, item)
        if record and str(record.get("decision") or "") == "discard":
            excluded.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "reason": "triage_discard",
                    "classified_by": record.get("classified_by") or "",
                }
            )
            continue
        kept.append((item, record))
    kept.sort(key=lambda pair: _triage_sort_key(pair[0], pair[1]))
    queued: list[dict[str, Any]] = []
    promoted_ids: list[str] = []
    for item, record in kept[:limit]:
        annotated = dict(item)
        if record:
            annotated["triage_decision"] = record.get("decision") or ""
            annotated["user_approved"] = is_approved_today(record)
            if str(record.get("decision") or "") == "today":
                promoted_ids.append(str(item.get("id") or ""))
        queued.append(annotated)
    return queued, excluded, promoted_ids


def build_queue(
    report: dict[str, Any],
    limit: int,
    execution_status: dict[str, Any] | None = None,
    batch_applied_targets: frozenset[str] = frozenset(),
    dynamic_blocked_keywords: list[str] | None = None,
    triage: dict[str, dict] | None = None,
    triage_mode: str = "shadow",
) -> dict[str, Any]:
    needs_review = [item for item in report.get("needs_review") or [] if isinstance(item, dict)]
    quota_blocked = quota_blocked_items(execution_status or {})
    safe: list[dict[str, Any]] = []
    maybe: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    quota_hold: list[dict[str, Any]] = []
    batch_apply_blocked: list[dict[str, Any]] = []

    for index, item in enumerate(needs_review):
        item = refresh_auto_fix_policy(item, repo_root())
        item["_source_index"] = index
        rev_id = str(item.get("id") or "")
        if rev_id in quota_blocked:
            quota_hold.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "reason": "blocked_by_quota",
                    "last_attempted_at": quota_blocked[rev_id].get("updated_at", ""),
                    "detail": quota_blocked[rev_id].get("detail", ""),
                }
            )
            continue
        # batch_apply が今日触れたファイルを扱う REV はキューに乗せない（二重適用防止）
        touched, touch_reason = is_batch_apply_touched(item, batch_applied_targets)
        if touched:
            batch_apply_blocked.append(
                {"id": item.get("id"), "title": item.get("title"), "reason": touch_reason}
            )
            continue
        is_manual, reason = is_blocked(item, dynamic_blocked_keywords)
        if is_manual:
            blocked.append({"id": item.get("id"), "title": item.get("title"), "reason": reason})
        elif is_codex_safe(item):
            safe.append(item)
        else:
            maybe.append(item)

    safe_sorted = sorted(safe, key=queue_sort_key)
    baseline_queued = safe_sorted[:limit]

    # ── Phase 2: トリアージ反映（P2-0 シャドー / P2-1 ライブ / P2-4 オフ）──────
    if triage_mode not in TRIAGE_MODES:
        triage_mode = "shadow"
    triage = triage or {}
    triage_info: dict[str, Any] = {"mode": triage_mode, "decisions_loaded": len(triage), "applied": False}
    queued = baseline_queued
    if triage_mode != "off" and triage:
        triage_queued, excluded_by_discard, promoted_ids = _apply_triage_to_safe(safe_sorted, triage, limit)
        baseline_ids = [str(item.get("id") or "") for item in baseline_queued]
        with_triage_ids = [str(item.get("id") or "") for item in triage_queued]
        triage_info.update(
            {
                "excluded_by_discard": excluded_by_discard,
                "promoted_today_ids": promoted_ids,
                "baseline_ids": baseline_ids,
                "with_triage_ids": with_triage_ids,
                "diverges": baseline_ids != with_triage_ids,
            }
        )
        if triage_mode == "live":
            # 実キューをトリアージ反映後の並びに切り替える。切り戻しは
            # SHION_TRIAGE_QUEUE_MODE=shadow / off で即時（P2-4）
            triage_info["applied"] = True
            queued = triage_queued

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_date": report.get("date"),
        "source_generated_at": report.get("generated_at"),
        "limit": limit,
        "needs_review_count": len(needs_review),
        "codex_auto_safe_count": len(safe),
        "codex_auto_maybe_count": len(maybe),
        "manual_or_blocked_count": len(blocked),
        "blocked_by_quota_count": len(quota_hold),
        "batch_apply_blocked_count": len(batch_apply_blocked),
        "queued_count": len(queued),
        "status": "READY" if queued else "EMPTY",
        "items": [queue_item(item) for item in queued],
        "skipped_safe_ids": [item.get("id") for item in safe_sorted[limit:]],
        "blocked_by_quota": quota_hold,
        "manual_or_blocked": blocked,
        "batch_apply_blocked": batch_apply_blocked,
        "triage": triage_info,
    }


def update_latest(latest_path: Path, queue_path: Path, queue: dict[str, Any]) -> None:
    if latest_path.exists():
        latest = load_json(latest_path)
    else:
        latest = {}
    latest["codex_auto_queue"] = {
        "status": queue.get("status"),
        "path": str(queue_path),
        "queued_count": queue.get("queued_count", 0),
        "safe_count": queue.get("codex_auto_safe_count", 0),
        "maybe_count": queue.get("codex_auto_maybe_count", 0),
        "manual_or_blocked_count": queue.get("manual_or_blocked_count", 0),
        "blocked_by_quota_count": queue.get("blocked_by_quota_count", 0),
        "batch_apply_blocked_count": queue.get("batch_apply_blocked_count", 0),
        "limit": queue.get("limit"),
        "generated_at": queue.get("generated_at"),
    }
    latest["codex_auto_queue_count"] = queue.get("queued_count", 0)
    latest["codex_auto_safe_count"] = queue.get("codex_auto_safe_count", 0)
    dump_json(latest_path, latest)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None, help="Improvement report JSON.")
    parser.add_argument("--latest", type=Path, default=root / "reports" / "latest.json")
    parser.add_argument("--output", type=Path, default=None, help="Queue JSON path.")
    parser.add_argument("--status-file", type=Path, default=root / "reports" / EXECUTION_STATUS_FILE)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--triage-mode",
        choices=list(TRIAGE_MODES),
        default=None,
        help="トリアージ反映モード。省略時は環境変数 SHION_TRIAGE_QUEUE_MODE → 既定 shadow",
    )
    args = parser.parse_args()

    report_path = args.report or latest_report_path(root)
    report = load_json(report_path)
    execution_status = load_execution_status(args.status_file)
    report_date = str(report.get("date") or dt.date.today().isoformat()).replace("-", "")
    output_path = args.output or root / "reports" / f"codex_auto_queue_{report_date}.json"

    batch_applied_targets = _batch_applied_targets_today(root)
    if batch_applied_targets:
        print(
            f"[build_codex_auto_queue] batch_apply guard: {len(batch_applied_targets)} target(s) today → "
            + ", ".join(sorted(batch_applied_targets)[:5])
        )

    dynamic_keywords = _load_dynamic_blocked_keywords(root)
    if dynamic_keywords:
        print(
            f"[build_codex_auto_queue] dynamic blocked keywords ({len(dynamic_keywords)}): "
            + ", ".join(dynamic_keywords[:5])
        )

    triage_mode = resolve_triage_mode(args.triage_mode)
    triage = load_triage_latest(root) if triage_mode != "off" else {}
    print(f"[build_codex_auto_queue] triage mode={triage_mode}, decisions={len(triage)}")

    queue = build_queue(
        report,
        max(0, args.limit),
        execution_status,
        batch_applied_targets,
        dynamic_keywords,
        triage=triage,
        triage_mode=triage_mode,
    )
    queue["source_report"] = str(report_path)
    queue["execution_status_file"] = str(args.status_file)

    triage_info = queue.get("triage") or {}
    if triage_info.get("diverges"):
        print(
            "[build_codex_auto_queue] triage比較: baseline="
            + ",".join(triage_info.get("baseline_ids") or [])
            + " / with_triage="
            + ",".join(triage_info.get("with_triage_ids") or [])
            + (" (適用済み)" if triage_info.get("applied") else " (シャドー・実キューは従来のまま)")
        )

    if args.dry_run:
        print(json.dumps(queue, ensure_ascii=False, indent=2))
        return

    dump_json(output_path, queue)
    update_latest(args.latest, output_path, queue)
    print(
        "Codex auto queue: "
        f"{queue['queued_count']} queued / {queue['codex_auto_safe_count']} safe "
        f"({output_path})"
    )


if __name__ == "__main__":
    main()
