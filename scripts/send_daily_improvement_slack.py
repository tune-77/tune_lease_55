"""Send the daily improvement report summary to Slack.

This is a narrow notification bridge for the morning improvement pipeline.
It reads the already-generated report and posts a concise summary through an
Incoming Webhook. It does not modify improvement status or promote items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "reports" / "latest.json"
DEFAULT_MANA_REPORT = REPO_ROOT / "reports" / "mana_obsidian_curator_latest.json"
DEFAULT_SCREENING_TERMS_REPORT = REPO_ROOT / "reports" / "screening_terms_audit_latest.json"
DEFAULT_STATE = REPO_ROOT / "data" / "slack_daily_improvement_state.json"
DEFAULT_TIMEOUT = 15


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"report not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid report json: {path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"report must be a JSON object: {path}")
    return data


def _read_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _report_hash(report: dict[str, Any]) -> str:
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _combined_hash(*values: dict[str, Any]) -> str:
    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _load_webhook(explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    env = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if env:
        return env
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return ""
    for line in secrets_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped.startswith("SLACK_WEBHOOK_URL"):
            continue
        _key, _, raw_value = stripped.partition("=")
        return raw_value.strip().strip('"').strip("'")
    return ""


def _clean_text(value: Any, limit: int = 140) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _items(report: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = report.get(key) or []
    return [item for item in value if isinstance(item, dict)]


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _mana_lines(mana_report: dict[str, Any] | None) -> list[str]:
    if not mana_report:
        return ["• status: `missing` / Manaレポート未生成"]

    status = _clean_text(mana_report.get("status") or "unknown", 24)
    inputs = mana_report.get("inputs") if isinstance(mana_report.get("inputs"), dict) else {}
    candidate_count = inputs.get("candidate_count", "-")
    useful_count = inputs.get("useful_candidate_count", "-")
    lines = [
        f"• status: `{status}` / candidates: `{candidate_count}` / useful: `{useful_count}`"
    ]

    findings = [item for item in mana_report.get("findings") or [] if isinstance(item, dict)]
    if not findings:
        lines.append("• findings: なし")
        return lines

    for finding in findings[:3]:
        code = _clean_text(finding.get("code") or "", 42)
        level = _clean_text(finding.get("level") or "", 12)
        message = _clean_text(finding.get("message") or "", 100)
        prefix = f"{code}: " if code else ""
        lines.append(f"• `{level}` {prefix}{message}")
    if len(findings) > 3:
        lines.append(f"• 他 {len(findings) - 3} 件")
    return lines


def _screening_terms_lines(terms_report: dict[str, Any] | None) -> list[str]:
    if not terms_report:
        return ["• status: `missing` / 審査用語監査レポート未生成"]

    status = _clean_text(terms_report.get("status") or "unknown", 24)
    counts = terms_report.get("counts") if isinstance(terms_report.get("counts"), dict) else {}
    warn = counts.get("warn", 0)
    review = counts.get("review", 0)
    ok = counts.get("ok", 0)
    report_path = REPO_ROOT / "reports" / "screening_terms_audit_latest.md"
    return [
        f"• status: `{status}` / warn: `{warn}` / review: `{review}` / ok: `{ok}`",
        f"• report: `{report_path.relative_to(REPO_ROOT)}`",
    ]


def build_message(
    report: dict[str, Any],
    *,
    report_date: str,
    mana_report: dict[str, Any] | None = None,
    screening_terms_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    applied = _items(report, "applied_improvements")
    needs_review = _items(report, "needs_review")
    failed = _items(report, "failed_improvements")
    applied_count = int(report.get("applied_count") or len(applied))
    needs_review_count = int(report.get("needs_review_count") or len(needs_review))
    failed_count = int(report.get("failed_count") or len(failed))

    top_review = needs_review[:5]
    review_lines = []
    for item in top_review:
        rev_id = _clean_text(item.get("id") or item.get("rev_id") or "", 24)
        title = _clean_text(item.get("title") or item.get("detail") or "無題", 120)
        risk = ""
        policy = item.get("auto_fix_policy")
        if isinstance(policy, dict) and policy.get("risk"):
            risk = f" / risk={policy.get('risk')}"
        prefix = f"{rev_id}: " if rev_id else ""
        review_lines.append(f"• {prefix}{title}{risk}")

    if not review_lines:
        review_lines.append("• 要レビュー項目なし")

    commit = report.get("commit_result") if isinstance(report.get("commit_result"), dict) else {}
    commit_msg = _clean_text(commit.get("message") or "commit情報なし", 120)

    text = "\n".join(
        [
            f"*日次改善レポート* `{report_date}`",
            f"• applied: `{applied_count}` / needs_review: `{needs_review_count}` / failed: `{failed_count}`",
            f"• commit: {commit_msg}",
            "",
            "*要レビュー上位*",
            *review_lines,
            "",
            "*Mana判定*",
            *_mana_lines(mana_report),
            "",
            "*審査用語監査*",
            *_screening_terms_lines(screening_terms_report),
            "",
            "_自動投稿: run_daily_improvement_pipeline / Slack通知のみ。改善状態は変更していません。_",
        ]
    )
    return {"text": text}


def send_slack(webhook_url: str, payload: dict[str, Any], *, timeout: int = DEFAULT_TIMEOUT) -> tuple[bool, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            text = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return False, str(exc)
    if status == 200 and text.strip() == "ok":
        return True, "ok"
    return False, f"HTTP {status}: {text}"


def should_skip(state: dict[str, Any], *, report_date: str, digest: str, force: bool) -> bool:
    if force:
        return False
    return state.get("last_sent_date") == report_date and state.get("last_report_hash") == digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Send daily improvement report summary to Slack.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mana-report", type=Path, default=DEFAULT_MANA_REPORT)
    parser.add_argument("--screening-terms-report", type=Path, default=DEFAULT_SCREENING_TERMS_REPORT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--webhook", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report = _read_json(args.report)
    mana_report = _read_optional_json(args.mana_report)
    screening_terms_report = _read_optional_json(args.screening_terms_report)
    digest = _combined_hash(report, mana_report or {}, screening_terms_report or {})
    payload = build_message(
        report,
        report_date=args.date,
        mana_report=mana_report,
        screening_terms_report=screening_terms_report,
    )

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    webhook_url = _load_webhook(args.webhook)
    if not webhook_url:
        print("SLACK_WEBHOOK_URL is not set; skipping Slack improvement report.")
        return 0
    if not webhook_url.startswith("https://hooks.slack.com/"):
        print("SLACK_WEBHOOK_URL must start with https://hooks.slack.com/; skipping.")
        return 0

    state = _read_state(args.state)
    if should_skip(state, report_date=args.date, digest=digest, force=args.force):
        print(f"Slack improvement report already sent for {args.date}; skipping.")
        return 0

    ok, detail = send_slack(webhook_url, payload)
    if not ok:
        print(f"Slack improvement report failed: {detail}", file=sys.stderr)
        return 1

    _write_state(
        args.state,
        {
            "last_sent_at": datetime.now().isoformat(timespec="seconds"),
            "last_sent_date": args.date,
            "last_report_hash": digest,
            "last_report": str(args.report),
        },
    )
    print(f"Slack improvement report sent for {args.date}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
