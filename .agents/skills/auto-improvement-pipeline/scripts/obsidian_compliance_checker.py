"""Obsidian連携ルールの整合性を確認するAgent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_FORBIDDEN_PATTERNS = [
    re.compile(r"\.rglob\(\s*[\"']\*\.md[\"']\s*\)"),
    re.compile(r"\.glob\(\s*[\"']\*\*/\*\.md[\"']\s*\)"),
]

_ALLOWED_FILES = {
    "mobile_app/obsidian_bridge.py",
    "scripts/extract_obsidian_improvements.py",
}

_SCAN_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx"}
_SCAN_DIRS = ["api", "mobile_app", "frontend/src", "components", "scripts"]


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _scan_for_direct_vault_search(workspace_root: Path) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []

    for dirname in _SCAN_DIRS:
        base = workspace_root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
                continue
            rel_path = _rel(path, workspace_root)
            if rel_path in _ALLOWED_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for lineno, line in enumerate(text.splitlines(), 1):
                if any(pattern.search(line) for pattern in _FORBIDDEN_PATTERNS):
                    violations.append({
                        "file": rel_path,
                        "line": lineno,
                        "rule": "no_direct_vault_markdown_scan",
                        "message": "Obsidian検索は obsidian_query.py / obsidian_ai_context.py / mobile_app/obsidian_bridge.py の共通経路を使う",
                    })

    return violations


def _requires_obsidian_route(item: dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("title", "description", "detail", "reason")
        if item.get(key)
    ).lower()
    return any(
        keyword in text
        for keyword in [
            "obsidian", "ai chat", "ナレッジ", "faq", "知識", "業界情報",
            "ニュース", "検索", "情報参照", "リース情報",
        ]
    )


def check_obsidian_compliance(
    improvements: list[dict[str, Any]],
    workspace_root: str | Path,
) -> dict[str, Any]:
    root = Path(workspace_root)
    violations = _scan_for_direct_vault_search(root)

    route_sensitive_ids = [
        item.get("id")
        for item in improvements
        if _requires_obsidian_route(item)
    ]

    notes: list[str] = []
    if route_sensitive_ids:
        notes.append(
            "Obsidian/AI Chat関連の改善は共通経路"
            "（obsidian_query.py, obsidian_ai_context.py, mobile_app/obsidian_bridge.py）"
            "を通す必要があります。"
        )

    status = "ok"
    if violations:
        status = "warning"

    return {
        "status": status,
        "violations": violations,
        "route_sensitive_ids": route_sensitive_ids,
        "required_route": [
            "obsidian_query.py",
            "obsidian_ai_context.py",
            "mobile_app/obsidian_bridge.py",
        ],
        "notes": notes,
    }


if __name__ == "__main__":
    import json

    result = check_obsidian_compliance([], Path.cwd())
    print(json.dumps(result, ensure_ascii=False, indent=2))
