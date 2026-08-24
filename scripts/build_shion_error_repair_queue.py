#!/usr/bin/env python3
"""Build a tiny Codex queue for low-risk Shion error repair.

This consumes `analyze_error_logs.py` entries from `api/rule_engine/ledger_rules.json`.
Only clearly small, file-scoped runtime/compile errors are allowed into the queue;
scoring, DB, auth, external API, infra, and ambiguous errors stay in review.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_codex_auto_queue import (  # noqa: E402
    is_blocked,
    is_codex_safe,
    queue_item,
    refresh_auto_fix_policy,
    repo_root,
)

LEDGER_PATH = Path("api/rule_engine/ledger_rules.json")
STATE_FILE_NAME = "shion_error_repair_queue_state.json"

_SAFE_ERROR_RE = re.compile(
    r"\b(NameError|ImportError|ModuleNotFoundError|AttributeError|SyntaxError|IndentationError)\b"
)
_FILE_RE = re.compile(r"[\w./-]+\.(?:py|tsx|ts|jsx|js|md|json)")
_DANGEROUS_PARTS = {"data", "models", "migrations", "alembic", ".github", "launchd"}
_SAFE_REPAIR_PREFIXES = (
    "frontend/src/app/",
    "frontend/src/components/",
    "frontend/src/lib/",
)
_DANGEROUS_NAME_RE = re.compile(
    r"(score|scoring|auth|security|credential|secret|db|database|migration|"
    r"lease_logic|category_config|coefficient|model)",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"queued_ids": []}
    try:
        data = load_json(path)
    except Exception:
        return {"queued_ids": []}
    if not isinstance(data, dict):
        return {"queued_ids": []}
    if not isinstance(data.get("queued_ids"), list):
        data["queued_ids"] = []
    return data


def _error_text(entry: dict[str, Any]) -> str:
    return " ".join(
        str(entry.get(key) or "")
        for key in ("rev_id", "description", "error_pattern", "target", "affected_files")
    )


def _single_safe_file(entry: dict[str, Any], root: Path) -> tuple[str | None, str]:
    candidates: list[str] = []
    for value in entry.get("affected_files") or []:
        if isinstance(value, str):
            candidates.append(value)
    candidates.extend(_FILE_RE.findall(_error_text(entry)))

    normalized: list[str] = []
    for raw in candidates:
        path = Path(raw)
        if path.is_absolute():
            try:
                path = path.relative_to(root)
            except ValueError:
                return None, f"workspace外のファイル参照: {raw}"
        rel = path.as_posix().lstrip("./")
        if not rel:
            continue
        if any(part in _DANGEROUS_PARTS for part in Path(rel).parts):
            return None, f"重要パス配下のため手動確認: {rel}"
        if _DANGEROUS_NAME_RE.search(rel):
            return None, f"審査ロジック/認証/DB/モデル系のため手動確認: {rel}"
        if not rel.startswith(_SAFE_REPAIR_PREFIXES):
            return None, f"UI表示系の許可パス外のため手動確認: {rel}"
        if rel not in normalized:
            normalized.append(rel)

    if not normalized:
        return None, "対象ファイル未特定"
    if len(normalized) > 1:
        return None, f"複数ファイル参照のため手動確認: {len(normalized)} files"
    if not (root / normalized[0]).exists():
        return None, f"対象ファイルが存在しない: {normalized[0]}"
    return normalized[0], ""


def classify_error_entry(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Return a queue-ready item verdict for one ledger error entry."""
    if entry.get("category") != "error_log_fix" or entry.get("source") != "analyze_error_logs":
        return {"is_safe": False, "reason": "エラーログ由来の候補ではない"}

    text = _error_text(entry)
    if not _SAFE_ERROR_RE.search(text):
        return {"is_safe": False, "reason": "軽微エラー種別として明示されていない"}

    target, target_reason = _single_safe_file(entry, root)
    if not target:
        return {"is_safe": False, "reason": target_reason}

    rev_id = str(entry.get("rev_id") or "")
    pattern = str(entry.get("error_pattern") or entry.get("description") or "")[:160]
    item = {
        "id": rev_id,
        "title": "表示/実行時の軽微エラー修復",
        "description": pattern,
        "detail": f"検出エラー: {pattern}" if pattern else "",
        "reason": (
            "紫苑の軽微エラー一次対応。import漏れ・名前解決・構文崩れなど、"
            "単一ファイル内で直せる範囲だけを確認する。"
        ),
        "target_module": target,
        "implementation": {"category": "runtime_error_repair"},
    }
    item = refresh_auto_fix_policy(item, root)
    blocked, block_reason = is_blocked(item)
    if blocked:
        return {"is_safe": False, "reason": block_reason, "item": item}
    if not is_codex_safe(item):
        return {"is_safe": False, "reason": "codex-safe条件を満たさない", "item": item}
    return {"is_safe": True, "reason": "軽微エラーとして自動修復キュー対象", "item": item}


def build_queue(
    ledger: list[dict[str, Any]],
    already_queued_ids: set[str],
    limit: int,
    root: Path,
) -> dict[str, Any]:
    safe: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []

    for entry in ledger:
        rev_id = str(entry.get("rev_id") or "")
        if not rev_id or rev_id in already_queued_ids:
            continue
        verdict = classify_error_entry(entry, root)
        if verdict.get("is_safe") and isinstance(verdict.get("item"), dict):
            safe.append(verdict["item"])
        elif entry.get("category") == "error_log_fix":
            manual.append(
                {
                    "id": rev_id,
                    "title": str(entry.get("description") or "")[:120],
                    "reason": verdict.get("reason") or "手動確認",
                }
            )

    queued = safe[: max(0, limit)]
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "limit": limit,
        "error_repair_safe_count": len(safe),
        "manual_or_blocked_count": len(manual),
        "queued_count": len(queued),
        "status": "READY" if queued else "EMPTY",
        "items": [queue_item(item) for item in queued],
        "skipped_safe_ids": [item.get("id") for item in safe[max(0, limit):]],
        "manual_or_blocked": manual,
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=root / LEDGER_PATH)
    parser.add_argument("--state-file", type=Path, default=root / "reports" / STATE_FILE_NAME)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ledger = load_json(args.ledger) if args.ledger.exists() else []
    if not isinstance(ledger, list):
        raise SystemExit("ledger must be a JSON array")
    state = load_state(args.state_file)
    already_queued_ids = {str(x) for x in state.get("queued_ids") or []}

    queue = build_queue(ledger, already_queued_ids, args.limit, root)
    date_tag = dt.date.today().strftime("%Y%m%d")
    output_path = args.output or root / "reports" / f"shion_error_repair_queue_{date_tag}.json"

    if args.dry_run:
        print(json.dumps(queue, ensure_ascii=False, indent=2))
        return

    try:
        queue["success_state_file"] = args.state_file.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise SystemExit("state file must be inside the repository")

    dump_json(output_path, queue)
    print(
        "Shion error repair queue: "
        f"{queue['queued_count']} queued / {queue['error_repair_safe_count']} safe "
        f"({output_path})"
    )


if __name__ == "__main__":
    main()
