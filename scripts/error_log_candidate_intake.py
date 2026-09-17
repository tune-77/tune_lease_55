#!/usr/bin/env python3
"""analyze_error_logs.py が ledger_rules.json に書いた候補を needs_review へ橋渡しする。

analyze_error_logs.py 自体は変更しない。このファイルは読み取り専用の橋渡し役で、
一度浮上させた rev_id を data/error_log_candidates_surfaced.json に記録し、
同じエラーパターンを重複して needs_review に積み上げないようにする。
target_module は常に空文字とし、auto_fix_policy.evaluate_auto_fix_policy() が
必ず手動確認（needs_review）に倒れるようにする。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LEDGER_FILE = _REPO_ROOT / "api" / "rule_engine" / "ledger_rules.json"
SURFACED_PATH = _REPO_ROOT / "data" / "error_log_candidates_surfaced.json"


def _load_surfaced_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    return {str(rev_id) for rev_id in data}


def _save_surfaced_ids(path: Path, rev_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(rev_ids), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_error_log_candidates(
    ledger_path: Path | None = None, surfaced_path: Path | None = None
) -> list[dict[str, Any]]:
    ledger_file = ledger_path or LEDGER_FILE
    surfaced_file = surfaced_path or SURFACED_PATH

    if not ledger_file.exists():
        return []
    try:
        ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(ledger, list):
        return []

    surfaced = _load_surfaced_ids(surfaced_file)
    newly_surfaced: set[str] = set()
    items: list[dict[str, Any]] = []

    for entry in ledger:
        if not isinstance(entry, dict):
            continue
        if entry.get("category") != "error_log_fix":
            continue
        if entry.get("source") != "analyze_error_logs":
            continue
        if entry.get("status") != "pending_review":
            continue
        rev_id = str(entry.get("rev_id") or "")
        if not rev_id or rev_id in surfaced:
            continue

        description = str(entry.get("description") or "").strip()
        if not description:
            continue

        newly_surfaced.add(rev_id)
        items.append({
            "id": f"ERRLOG-{rev_id}",
            "title": description,
            "description": description,
            "target_module": "",
            "source": "analyze_error_logs",
        })

    if newly_surfaced:
        _save_surfaced_ids(surfaced_file, surfaced | newly_surfaced)

    return items
