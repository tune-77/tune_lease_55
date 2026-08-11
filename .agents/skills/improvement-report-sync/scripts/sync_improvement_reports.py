#!/usr/bin/env python3
"""Sync applied improvement IDs into the latest improvement reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


KNOWN_APPLIED_TITLE_PATTERNS = [
    "担当者の最終判断を記録する登録ボタンがない",
    "AIチャット履歴の表示を制御する機能がない",
    "対話AIへの接続エラー",
    "企業登録後、結果登録画面に直前に登録した企業が表示されない",
    "ダッシュボードにニュースが表示されていない",
    "リース審査専門Wikiが参照するVault名",
    "紫苑」が過去の会話内容を記憶していない",
]

KNOWN_PARKED_TITLE_PATTERNS = [
    "同期とDB監査",
    "Q_riskは既存数式",
    "スコアリング外",
    "60-80帯",
    "40-60帯",
    "モデルドリフト",
    "キャリブレーション",
    "PSI/CSI",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def pipeline_skill_dir(root: Path) -> Path:
    return root / ".agents" / "skills" / "auto-improvement-pipeline"


def latest_report_path(root: Path) -> Path:
    reports = sorted((root / "reports").glob("improvement_report_*.json"))
    if not reports:
        print("エラー: reports/improvement_report_*.json ファイルが見つかりません", file=sys.stderr)
        raise SystemExit(1)
    return reports[-1]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, IOError) as e:
        print(f"エラー: JSON ファイルの読み込みに失敗（パス: {path}）: {e}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"エラー: JSON ファイルの構文エラー（パス: {path}）: {e}", file=sys.stderr)
        raise


def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def id_map(items: Iterable[dict]) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        if item_id and item_id not in mapped:
            mapped[item_id] = item
    return mapped


def list_value(data: dict, key: str) -> list[dict]:
    value = data.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def first_list_value(data: dict, *keys: str) -> list[dict]:
    for key in keys:
        value = list_value(data, key)
        if value:
            return value
    return []


def normalize_text(value: str) -> str:
    return re.sub(r"[\s　・/（）()【】\[\]「」:：,，.。_-]+", "", value.lower())


def normalize_status(value: str) -> str:
    parts = str(value or "").split()
    return parts[0].strip().lower() if parts else ""


TERMINAL_LEDGER_STATUSES = {
    "applied",
    "approved",
    "deleted",
    "rejected",
    "deferred",
    "parked",
    "suppressed",
    "rule_registered",
    "apply_failed",
}


PARKED_LEDGER_STATUSES = {"parked", "deferred", "deleted", "rejected", "suppressed"}


def is_terminal_status(status: str) -> bool:
    return normalize_status(status) in TERMINAL_LEDGER_STATUSES


def prefer_ledger_entry(current: dict | None, candidate: dict) -> dict:
    """Keep human/resolution decisions ahead of later automatic needs_review rows."""
    if current is None:
        return candidate
    current_terminal = is_terminal_status(str(current.get("status") or ""))
    candidate_terminal = is_terminal_status(str(candidate.get("status") or ""))
    if candidate_terminal and not current_terminal:
        return candidate
    if current_terminal and not candidate_terminal:
        return current
    return candidate


def load_ledger_latest(ledger_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not ledger_path.exists():
        return latest

    try:
        content = ledger_path.read_text(encoding="utf-8")
    except (OSError, IOError) as e:
        print(f"警告: ledger ファイルの読み込みに失敗（パス: {ledger_path}）: {e}", file=sys.stderr)
        return latest

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        status = normalize_status(entry.get("status", ""))
        if not status:
            continue
        aliases = [
            entry.get("key"),
            entry.get("canonical_key"),
            entry.get("rev_id"),
        ]
        for alias in aliases:
            alias_text = str(alias or "").strip()
            if alias_text:
                latest[alias_text] = prefer_ledger_entry(latest.get(alias_text), entry)
    return latest


def load_canonical_key_func(root: Path):
    scripts_dir = pipeline_skill_dir(root) / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from improvement_identity import canonical_key  # type: ignore
    except Exception as e:
        print(f"警告: improvement_identity モジュールのロード失敗（canonical_key は利用不可）: {e}", file=sys.stderr)
        return None
    return canonical_key


def grouped_aliases(report: dict, latest: dict) -> dict[str, set[str]]:
    aliases_by_id: dict[str, set[str]] = {}
    for source in (report, latest):
        for group in source.get("grouped_improvements") or []:
            ids = {
                str(group.get("canonical_id") or ""),
                *[str(item_id or "") for item_id in group.get("ids") or []],
                *[str(item_id or "") for item_id in group.get("original_ids") or []],
                *[str(item_id or "") for item_id in group.get("duplicate_ids") or []],
            }
            aliases = {
                str(group.get("group_key") or ""),
                str(group.get("canonical_key") or ""),
                str(group.get("canonical_id") or ""),
                str(group.get("group_id") or ""),
                *ids,
            }
            aliases = {alias for alias in aliases if alias}
            for item_id in ids:
                if item_id:
                    aliases_by_id.setdefault(item_id, set()).update(aliases)
    return aliases_by_id


def item_aliases(item: dict, aliases_by_id: dict[str, set[str]], canonical_key_func) -> set[str]:
    item_id = str(item.get("id") or "")
    aliases = {
        item_id,
        str(item.get("canonical_key") or ""),
        str(item.get("group_key") or ""),
        str(item.get("rev_id") or ""),
    }
    if item_id:
        aliases.update(aliases_by_id.get(item_id, set()))

    if canonical_key_func is not None:
        title = str(item.get("title") or "")
        descriptions = [
            str(item.get("detail") or ""),
            str(item.get("description") or ""),
            str(item.get("reason") or ""),
            "",
        ]
        for description in descriptions:
            if title or description:
                aliases.add(str(canonical_key_func(title, description)))

    return {alias for alias in aliases if alias}


def infer_status_ids_from_ledger(
    report: dict,
    latest: dict,
    ledger_path: Path,
    canonical_key_func=None,
) -> tuple[list[str], list[str]]:
    ledger_latest = load_ledger_latest(ledger_path)
    if not ledger_latest:
        return [], []

    aliases_by_id = grouped_aliases(report, latest)
    candidate_by_id: dict[str, dict] = {}
    for item in list(report.get("needs_review") or []) + list(latest.get("needs_review") or []):
        item_id = str(item.get("id") or "")
        if item_id:
            candidate_by_id.setdefault(item_id, item)

    applied_ids: list[str] = []
    parked_ids: list[str] = []
    for item_id, item in candidate_by_id.items():
        statuses = {
            normalize_status(ledger_latest[alias].get("status", ""))
            for alias in item_aliases(item, aliases_by_id, canonical_key_func)
            if alias in ledger_latest
        }
        if "applied" in statuses:
            applied_ids.append(item_id)
        elif statuses & PARKED_LEDGER_STATUSES:
            parked_ids.append(item_id)

    return applied_ids, parked_ids


def title_matches(item: dict, patterns: Iterable[str]) -> bool:
    text = normalize_text(" ".join([
        str(item.get("title") or ""),
        str(item.get("detail") or ""),
        str(item.get("reason") or ""),
    ]))
    return any(normalize_text(pattern) in text for pattern in patterns)


def applied_entry(item: dict) -> dict:
    return {
        "id": item.get("id"),
        "file": item.get("file"),
        "title": item.get("title"),
        "pr_url": item.get("pr_url"),
    }


def status_entry(item: dict, reason: str = "") -> dict:
    entry = applied_entry(item)
    if reason:
        entry["reason"] = reason
    return entry


def sync_report(
    report: dict,
    applied_ids: list[str],
    applied_titles: list[str],
    parked_ids: list[str],
    parked_titles: list[str],
) -> tuple[dict, list[str], list[str], list[str]]:
    applied_ids_set = set(applied_ids)
    parked_ids_set = set(parked_ids)
    needs_review = list_value(report, "needs_review")
    applied = first_list_value(report, "applied", "applied_improvements")
    parked = first_list_value(report, "parked", "parked_improvements")

    needs_map = id_map(needs_review)
    applied_map = id_map(applied)
    moved: list[str] = []
    parked_moved: list[str] = []
    skipped: list[str] = []

    for item_id in applied_ids:
        if item_id in applied_map:
            continue
        item = needs_map.pop(item_id, None)
        if item is None:
            skipped.append(item_id)
            continue
        moved.append(item_id)
        applied.append(applied_entry(item))

    for item in list(needs_map.values()):
        item_id = str(item.get("id") or "")
        if item_id in applied_ids_set or item_id in parked_ids_set:
            continue
        if title_matches(item, applied_titles):
            needs_map.pop(item_id, None)
            moved.append(item_id)
            applied.append(applied_entry(item))

    for item_id in parked_ids:
        item = needs_map.pop(item_id, None)
        if item is None:
            skipped.append(item_id)
            continue
        parked_moved.append(item_id)
        parked.append(status_entry(item, "実装タスクではなく継続監視・研究テーマとして保留"))

    for item in list(needs_map.values()):
        item_id = str(item.get("id") or "")
        if title_matches(item, parked_titles):
            needs_map.pop(item_id, None)
            parked_moved.append(item_id)
            parked.append(status_entry(item, "実装タスクではなく継続監視・研究テーマとして保留"))

    removed_ids = applied_ids_set | parked_ids_set | set(moved) | set(parked_moved)
    report["needs_review"] = [item for item in needs_review if str(item.get("id") or "") not in removed_ids]
    report["applied"] = applied
    if parked:
        report["parked"] = parked
    summary = report.get("summary") or {}
    summary["applied_count"] = len(applied)
    summary["needs_review_count"] = len(report["needs_review"])
    if parked:
        summary["parked_count"] = len(parked)
    report["summary"] = summary
    return report, moved, parked_moved, skipped


def sync_latest(
    latest: dict,
    applied_ids: list[str],
    parked_ids: list[str],
) -> dict:
    """`report`（sync_report の出力）が確定させた applied_ids / parked_ids を
    latest.json のスキーマへそのまま投影する。

    以前はここでも独自に title パターンマッチを再実行しており、sync_report
    との判定が食い違う不具合（REV-019 が applied と parked の両方に計上され
    る）が起きた。判定ロジックは sync_report 側のみに一本化し、ここは
    「report で決まったことを latest.json の形に書き写すだけ」にする。
    渡される applied_ids / parked_ids は sync_report の出力 (report["applied"],
    report["parked"]) 由来のため、呼び出し時点で互いに排他であることが
    構造的に保証されている。
    """
    applied_ids_set = set(applied_ids)
    parked_ids_set = set(parked_ids)
    needs_review = list(latest.get("needs_review") or [])
    applied_improvements = list(latest.get("applied_improvements") or [])
    parked_improvements = list(latest.get("parked_improvements") or [])
    items = list(latest.get("items") or [])

    needs_map = id_map(needs_review)
    applied_map = id_map(applied_improvements)
    parked_map = id_map(parked_improvements)
    item_map = id_map(items)

    for item_id in applied_ids:
        if item_id in applied_map:
            continue
        item = item_map.get(item_id) or needs_map.get(item_id)
        if item is None:
            continue

        applied_improvements.append(applied_entry(item))

        for existing in items:
            if str(existing.get("id") or "") == item_id:
                existing["status"] = "APPLIED"
                existing["reason"] = "改善済み登録済み"

    for item_id in parked_ids:
        if item_id in applied_ids_set or item_id in parked_map:
            continue
        item = item_map.get(item_id) or needs_map.get(item_id)
        if item is None:
            continue
        parked_improvements.append(status_entry(item, "実装タスクではなく継続監視・研究テーマとして保留"))

    removed_ids = applied_ids_set | parked_ids_set
    latest["needs_review"] = [item for item in needs_review if str(item.get("id") or "") not in removed_ids]
    latest["applied_improvements"] = applied_improvements
    if parked_improvements:
        latest["parked_improvements"] = parked_improvements

    if items:
        latest["approved"] = sum(1 for item in items if item.get("status") in {"APPROVED", "AUTO_FIX_CANDIDATE"})
        latest["auto_fix_candidates"] = sum(1 for item in items if item.get("status") == "AUTO_FIX_CANDIDATE")
        latest["needs_review"] = [item for item in latest["needs_review"] if str(item.get("id") or "") not in removed_ids]
        latest["parked"] = sum(1 for item in items if item.get("status") == "PARKED")
        latest["rejected"] = sum(1 for item in items if item.get("status") == "REJECTED")
        latest["applied"] = sum(1 for item in items if item.get("status") == "APPLIED")
    else:
        latest["applied"] = len(applied_improvements)

    latest["applied_count"] = len(applied_improvements)
    latest["needs_review_count"] = len(latest.get("needs_review") or [])
    if parked_improvements:
        latest["parked_count"] = len(parked_improvements)
    latest["status"] = "COMPLETED" if latest["applied_count"] > 0 else "NO_APPLIED"
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Target improvement report JSON",
    )
    parser.add_argument(
        "--latest",
        type=Path,
        default=repo_root() / "reports" / "latest.json",
        help="Latest summary JSON to keep in sync",
    )
    parser.add_argument(
        "--applied",
        action="append",
        default=[],
        help="Improvement ID to mark as applied. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--applied-title",
        action="append",
        default=[],
        help="Title/detail substring to mark as applied. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--parked",
        action="append",
        default=[],
        help="Improvement ID to move out of needs_review as a parked monitoring/research item.",
    )
    parser.add_argument(
        "--parked-title",
        action="append",
        default=[],
        help="Title/detail substring to move out of needs_review as parked. Repeat for multiple patterns.",
    )
    parser.add_argument(
        "--from-report",
        action="store_true",
        help="Infer applied IDs from the report's applied/applied_improvements entries.",
    )
    parser.add_argument(
        "--include-known-cleanup",
        action="store_true",
        help="Apply known Shion cleanup title patterns for already-fixed or strategic-monitoring items.",
    )
    parser.add_argument(
        "--from-ledger",
        action="store_true",
        help="Infer applied/parked IDs from the runtime improvement ledger.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl",
        help="Runtime improvement ledger JSONL path used by --from-ledger.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned changes without writing files.")
    args = parser.parse_args()

    root = repo_root()
    report_path = args.report or latest_report_path(root)

    try:
        report = load_json(report_path)
    except (SystemExit, OSError, IOError, json.JSONDecodeError) as e:
        print(f"エラー: レポートファイルの読み込みに失敗しました: {e}", file=sys.stderr)
        raise SystemExit(1)

    try:
        latest = load_json(args.latest) if args.latest.exists() else {}
    except (OSError, IOError, json.JSONDecodeError) as e:
        print(f"警告: latest ファイルの読み込みに失敗（空のオブジェクトで継続）: {e}", file=sys.stderr)
        latest = {}

    applied_ids = list(args.applied)
    if args.from_report:
        applied_ids = [
            str(item.get("id") or "")
            for item in first_list_value(report, "applied", "applied_improvements")
            if str(item.get("id") or "")
        ]

    applied_titles = list(args.applied_title)
    parked_titles = list(args.parked_title)
    if args.include_known_cleanup:
        applied_titles.extend(KNOWN_APPLIED_TITLE_PATTERNS)
        parked_titles.extend(KNOWN_PARKED_TITLE_PATTERNS)

    parked_ids = list(args.parked)
    if args.from_ledger:
        ledger_path = args.ledger.expanduser()
        if not ledger_path.exists():
            print(f"警告: ledger ファイルが見つかりません（パス: {ledger_path}）、--from-ledger をスキップします", file=sys.stderr)
        else:
            try:
                canonical_key_func = load_canonical_key_func(root)
                ledger_applied_ids, ledger_parked_ids = infer_status_ids_from_ledger(
                    report,
                    latest,
                    ledger_path,
                    canonical_key_func=canonical_key_func,
                )
                applied_ids.extend(item_id for item_id in ledger_applied_ids if item_id not in applied_ids)
                parked_ids.extend(item_id for item_id in ledger_parked_ids if item_id not in parked_ids)
            except Exception as e:
                print(f"警告: ledger から apply/parked 情報を読み込み中にエラー（スキップ）: {e}", file=sys.stderr)

    if not applied_ids and not applied_titles and not parked_ids and not parked_titles:
        print("適用済み改善項目なし（スキップ）")
        return

    updated_report, moved, parked_moved, skipped = sync_report(
        report,
        applied_ids,
        applied_titles,
        parked_ids,
        parked_titles,
    )

    # latest.json は report（上で確定した分類）を単に投影するだけにする。
    # applied_ids/parked_ids/*_titles を latest 側でも独立に再解決すると、
    # 判定ロジックが2箇所に分岐して食い違う（REV-019 の二重計上事故の原因）。
    final_applied_ids = [
        str(item.get("id") or "") for item in updated_report.get("applied", []) if item.get("id")
    ]
    final_parked_ids = [
        str(item.get("id") or "") for item in updated_report.get("parked", []) if item.get("id")
    ]
    updated_latest = sync_latest(
        latest,
        final_applied_ids,
        final_parked_ids,
    ) if latest else {}

    if args.dry_run:
        print(json.dumps({
            "report": str(report_path),
            "latest": str(args.latest),
            "moved": moved,
            "parked": parked_moved,
            "skipped": skipped,
            "applied_count": updated_report.get("summary", {}).get("applied_count", 0),
            "needs_review_count": updated_report.get("summary", {}).get("needs_review_count", 0),
            "parked_count": updated_report.get("summary", {}).get("parked_count", 0),
        }, ensure_ascii=False, indent=2))
        return

    try:
        dump_json(report_path, updated_report)
    except (OSError, IOError) as e:
        print(f"エラー: レポートファイルの書き込みに失敗（パス: {report_path}）: {e}", file=sys.stderr)
        raise

    try:
        if updated_latest:
            dump_json(args.latest, updated_latest)
    except (OSError, IOError) as e:
        print(f"警告: latest ファイルの書き込みに失敗（パス: {args.latest}）: {e}", file=sys.stderr)

    print(f"Updated report: {report_path}")
    print(f"Updated latest: {args.latest}")
    print(f"Moved: {', '.join(moved) if moved else '(none)'}")
    print(f"Parked: {', '.join(parked_moved) if parked_moved else '(none)'}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
