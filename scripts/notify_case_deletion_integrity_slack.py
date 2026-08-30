#!/usr/bin/env python3
"""Notify Slack only when the case-deletion integrity audit is abnormal."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.send_daily_improvement_slack import (
    _is_plausible_slack_webhook,
    _load_webhook,
    send_slack,
)

DEFAULT_REPORT = REPO_ROOT / "reports" / "case_deletion_integrity_latest.json"
DEFAULT_STATE = REPO_ROOT / "data" / "slack_case_deletion_integrity_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid report json: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"report must be a JSON object: {path}")
    return payload


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def alert_fingerprint(report: dict[str, Any]) -> str:
    canonical = {
        "status": report.get("status"),
        "issues": [
            {
                key: value
                for key, value in issue.items()
                if key not in {"age_minutes"}
            }
            for issue in report.get("issues") or []
            if isinstance(issue, dict)
        ],
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def build_slack_payload(report: dict[str, Any]) -> dict[str, str]:
    status = str(report.get("status") or "FAIL")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    issues = [item for item in report.get("issues") or [] if isinstance(item, dict)]
    icon = "🚨" if status == "FAIL" else "⚠️"
    lines = [
        f"{icon} *案件削除監査の整合性異常: {status}*",
        "",
        (
            f"• events: `{summary.get('events', 0)}` / "
            f"errors: `{summary.get('errors', 0)}` / warnings: `{summary.get('warnings', 0)}`"
        ),
    ]
    for issue in issues[:5]:
        event = f" / `{issue.get('event_id')}`" if issue.get("event_id") else ""
        lines.append(f"• `{issue.get('severity', 'error')}` {issue.get('code', 'unknown')}{event}")
    if len(issues) > 5:
        lines.append(f"• 他 {len(issues) - 5} 件")
    lines.extend(
        [
            "",
            "確認: `reports/case_deletion_integrity_latest.md`",
            "_読み取り専用監査です。自動修復・削除・復元は行っていません。_",
        ]
    )
    return {"text": "\n".join(lines)}


def notify_if_abnormal(
    report: dict[str, Any],
    *,
    state_path: Path,
    webhook_url: str,
    dry_run: bool = False,
    sender: Callable[[str, dict[str, Any]], tuple[bool, str]] = send_slack,
) -> tuple[str, str]:
    status = str(report.get("status") or "FAIL").upper()
    state = _read_state(state_path)
    if status == "PASS":
        if not dry_run:
            _write_state(
                state_path,
                {
                    "last_checked_at": datetime.now().isoformat(timespec="seconds"),
                    "last_status": "PASS",
                    "active_fingerprint": "",
                    "last_sent_at": state.get("last_sent_at", ""),
                },
            )
        return "healthy_skip", "PASS: Slack notification not sent"

    fingerprint = alert_fingerprint(report)
    if state.get("active_fingerprint") == fingerprint:
        return "duplicate_skip", f"{status}: identical alert already sent"

    payload = build_slack_payload(report)
    if dry_run:
        return "dry_run", json.dumps(payload, ensure_ascii=False)
    if not webhook_url or not _is_plausible_slack_webhook(webhook_url):
        return "webhook_missing", "SLACK_WEBHOOK_URL is missing or invalid"

    ok, detail = sender(webhook_url, payload)
    if not ok:
        return "send_failed", detail
    _write_state(
        state_path,
        {
            "last_checked_at": datetime.now().isoformat(timespec="seconds"),
            "last_status": status,
            "active_fingerprint": fingerprint,
            "last_sent_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return "sent", detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--webhook", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    report = _read_json(args.report)
    webhook_url = "" if args.dry_run or str(report.get("status") or "").upper() == "PASS" else _load_webhook(args.webhook)
    result, detail = notify_if_abnormal(
        report,
        state_path=args.state,
        webhook_url=webhook_url,
        dry_run=args.dry_run,
    )
    print(f"case deletion Slack notification: {result} — {detail}")
    return 1 if result == "send_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
