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
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_REPORT = REPO_ROOT / "reports" / "latest.json"
DEFAULT_MANA_REPORT = REPO_ROOT / "reports" / "mana_obsidian_curator_latest.json"
DEFAULT_SCREENING_TERMS_REPORT = REPO_ROOT / "reports" / "screening_terms_audit_latest.json"
DEFAULT_JUDGMENT_ASSET_GROWTH_REPORT = REPO_ROOT / "reports" / "judgment_asset_growth_latest.json"
DEFAULT_JUDGMENT_ASSET_FIELD_REVIEW = REPO_ROOT / "reports" / "judgment_asset_field_review_latest.json"
DEFAULT_ACTION_LEDGER_REPORT = REPO_ROOT / "reports" / "agent_action_ledger_latest.json"
DEFAULT_REFLECTION_JOURNAL_REPORT = REPO_ROOT / "reports" / "obsidian_reflection_journal_latest.json"
DEFAULT_SHION_OBSIDIAN_CURATOR_DAILY = REPO_ROOT / "reports" / "shion_obsidian_curator_daily_latest.json"
DEFAULT_STATE = REPO_ROOT / "data" / "slack_daily_improvement_state.json"
DEFAULT_TIMEOUT = 15
# 本文ハッシュ方式へ切り替えた版数。旧方式（生JSON全体からのハッシュ）で保存された
# state は digest_version が無いため、当日中にこのバージョンをまたいでパイプラインが
# 再実行されるとハッシュ方式の違いだけで不一致になり、内容が同じでも重複送信して
# しまう（Codexレビュー指摘）。それを避けるため、バージョン不一致時は日付一致のみで
# 「送信済み」とみなす。
DIGEST_VERSION = 2


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


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _is_plausible_slack_webhook(webhook_url: str) -> bool:
    """Return False for obvious placeholders/truncated Slack webhook URLs."""
    parsed = urlparse(webhook_url)
    if parsed.scheme != "https" or parsed.netloc != "hooks.slack.com":
        return False
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 4 or parts[0] != "services":
        return False
    team, channel, secret = parts[1:]
    return (
        team.startswith("T")
        and channel.startswith("B")
        and len(team) >= 8
        and len(channel) >= 8
        and len(secret) >= 20
    )


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


def _mana_action_fallback(findings: list[dict[str, Any]]) -> str:
    codes = {str(item.get("code") or "") for item in findings}
    if "memory_insight_reports_warning" in codes:
        return (
            "memory insight sidecar更新: "
            "scripts/build_obsidian_memory_insight_report.py / "
            "scripts/build_shion_memory_promotion_queue.py を再実行して36h以内に戻す"
        )
    if "private_reflection_not_meaningful" in codes or "reflection_handoff_incomplete" in codes:
        return "Private ReflectionをUser要求・誤読・次回行動が分かる形で再生成する"
    if "reflection_delta_missing" in codes:
        return "scripts/build_shion_reflection_delta.py を再実行する"
    if "monitor_report_missing" in codes:
        return "scripts/monitor_obsidian_environment.py を再実行する"
    return ""


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
    action_summary = _clean_text(mana_report.get("action_summary") or _mana_action_fallback(findings), 160)
    if action_summary:
        lines.append(f"• next: {action_summary}")

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


def _judgment_asset_growth_lines(growth_report: dict[str, Any] | None) -> list[str]:
    if not growth_report:
        return ["• status: `missing` / 判断資産成長レポート未生成"]

    latest = growth_report.get("latest") if isinstance(growth_report.get("latest"), dict) else growth_report
    score = latest.get("score", "-")
    components = latest.get("components") if isinstance(latest.get("components"), dict) else {}
    field_validation = components.get("field_validation", "-")
    feedback = latest.get("field_feedback") if isinstance(latest.get("field_feedback"), dict) else {}
    totals = feedback.get("totals") if isinstance(feedback.get("totals"), dict) else {}
    unused = feedback.get("unused_active_rules", "-")
    report_path = REPO_ROOT / "reports" / "judgment_asset_growth_latest.md"
    return [
        f"• score: `{score}` / field_validation: `{field_validation}`",
        (
            "• feedback: "
            f"used=`{totals.get('used', 0)}` / helped=`{totals.get('helped', 0)}` / "
            f"challenged=`{totals.get('challenged', 0)}` / rejected=`{totals.get('rejected', 0)}` / "
            f"unused_active=`{unused}`"
        ),
        f"• report: `{report_path.relative_to(REPO_ROOT)}`",
    ]


def _judgment_asset_field_review_lines(field_review: dict[str, Any] | None) -> list[str]:
    if not field_review:
        return ["• field_review: `missing` / 判断資産棚卸しレポート未生成"]

    summary = field_review.get("summary") if isinstance(field_review.get("summary"), dict) else {}
    report_path = REPO_ROOT / "reports" / "judgment_asset_field_review_latest.md"
    return [
        (
            "• field_review: "
            f"grow=`{summary.get('grow', 0)}` / review=`{summary.get('review', 0)}` / "
            f"sleeping=`{summary.get('sleeping', 0)}` / hold=`{summary.get('hold', 0)}`"
        ),
        f"• field_report: `{report_path.relative_to(REPO_ROOT)}`",
    ]


_SHION_ACTION_LABELS = {
    "daily_report": "日次報告",
    "system_watch": "システム監視",
    "improvement_classified": "改善候補の分類",
    "codex_request_drafted": "Codex依頼文の生成",
    "user_approval_requested": "承認依頼",
    "user_decision_recorded": "User判断の記録",
    "implementation_observed": "実装結果の観測",
    "followup_reported": "翌日報告への反映",
}


def _action_ledger_lines(action_ledger_report: dict[str, Any] | None) -> list[str]:
    """紫苑 Agent Action Ledger（backlog §9.2）の日次サマリを1〜数行にする。監査用の可視化のみ。"""
    if not action_ledger_report:
        return ["• status: `missing` / Agent Action Ledgerレポート未生成"]

    total = int(action_ledger_report.get("total") or 0)
    days = int(action_ledger_report.get("days") or 7)
    pending = int(action_ledger_report.get("pending_approval_count") or 0)
    lines = [f"• 直近{days}日: `{total}` 件 / 承認待ち `{pending}` 件"]

    by_action = action_ledger_report.get("by_action") if isinstance(action_ledger_report.get("by_action"), dict) else {}
    top_actions = sorted(
        ((k, v) for k, v in by_action.items() if v), key=lambda kv: kv[1], reverse=True
    )[:3]
    if top_actions:
        parts = [f"{_SHION_ACTION_LABELS.get(k, k)} {v}" for k, v in top_actions]
        lines.append("• 内訳: " + " / ".join(parts))
    return lines


# 会話外でも問題が届くよう、日次レポートに「システム監視」節を載せる。
# 閾値はチャットの自発報告（api/main.py）と揃える。
_STALE_REPORT_DAYS = 3
_PENDING_BACKLOG_THRESHOLD = 25


def _system_monitor_lines(now: datetime | None = None) -> list[str]:
    """自己改善レポートの鮮度低下・未完了タスク滞留を検出して行にする（無ければ異常なし）。"""
    import re as _re

    now = now or datetime.now()
    problems: list[str] = []

    report_md = REPO_ROOT / "reports" / "recursive_self_improvement_latest.md"
    if report_md.exists():
        try:
            m = _re.search(r"Generated at:\s*`([^`]+)`", report_md.read_text(encoding="utf-8"))
            if m:
                age = (now - datetime.fromisoformat(m.group(1).strip())).days
                if age >= _STALE_REPORT_DAYS:
                    problems.append(f"• ⚠️ 自己改善レポートが `{age}` 日更新なし（改善パイプライン停止の可能性）")
        except (ValueError, OSError):
            pass

    tasks_path = REPO_ROOT / "data" / "shion_pending_tasks.json"
    if tasks_path.exists():
        try:
            from lease_intelligence_pending import is_pending_open

            data = json.loads(tasks_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                open_count = sum(1 for t in data if is_pending_open(t))
                if open_count >= _PENDING_BACKLOG_THRESHOLD:
                    problems.append(f"• ⚠️ 未完了調査タスクが `{open_count}` 件滞留（改善ログで追跡中）")
        except (ValueError, OSError):
            pass

    aborted_line = _codex_queue_abort_line()
    if aborted_line:
        problems.append(aborted_line)

    return problems if problems else ["• 異常なし"]


def _codex_queue_abort_line() -> str:
    """execute_codex_queue.py の連続失敗ガード（backlog §9 前提ガード）が発動していれば警告行を返す。

    guards.aborted_by_consecutive_failures は次に人が結果JSONを開くまで気づけないため、
    日次Slackレポートの「システム監視」節で能動的に知らせる。
    """
    candidates = sorted((REPO_ROOT / "reports").glob("codex_queue_result_*.json"))
    if not candidates:
        return ""
    try:
        result = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    guards = result.get("guards") if isinstance(result.get("guards"), dict) else {}
    if not guards.get("aborted_by_consecutive_failures"):
        return ""
    carried_over = guards.get("carried_over") or []
    return (
        f"• ⚠️ Codex自動実行キューが連続失敗のため停止しました"
        f"（{candidates[-1].name} / 持ち越し {len(carried_over)} 件。原因調査が必要です）"
    )


def _shion_self_proposal_lines(report: dict[str, Any]) -> list[str]:
    section = report.get("shion_self_proposals")
    if not isinstance(section, dict):
        return ["• なし"]

    count = int(section.get("count") or 0)
    items = section.get("items") if isinstance(section.get("items"), list) else []
    if count <= 0 and not items:
        return ["• なし"]

    lines = [f"• 合計 `{count}` 件（通常のneeds_reviewには含めない）"]
    by_layer = section.get("counts_by_layer") if isinstance(section.get("counts_by_layer"), dict) else {}
    if by_layer:
        layer_labels = section.get("layer_labels") if isinstance(section.get("layer_labels"), dict) else {}
        parts = []
        for key in ("usage_based", "feedback_based", "system_audit_based"):
            value = int(by_layer.get(key) or 0)
            label = _clean_text(layer_labels.get(key) or key, 24)
            parts.append(f"{label} {value}")
        lines.append(f"• 根拠層: {' / '.join(parts)}")
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        kind = _clean_text(item.get("kind") or "自己提案", 24)
        layer = _clean_text(item.get("evidence_layer_label") or item.get("evidence_layer") or "", 24)
        title = _clean_text(item.get("title") or item.get("topic") or "無題", 90)
        priority = _clean_text(item.get("priority") or "", 16)
        suffix = f" / priority={priority}" if priority else ""
        prefix = f"[{kind} / {layer}]" if layer else f"[{kind}]"
        lines.append(f"• {prefix} {title}{suffix}")
        hypothesis = _clean_text(item.get("hypothesis") or "", 120)
        success_metric = _clean_text(item.get("success_metric") or "", 100)
        if hypothesis:
            lines.append(f"  仮説: {hypothesis}")
        if success_metric:
            lines.append(f"  成功指標: {success_metric}")
    return lines


def _reflection_journal_lines(report: dict[str, Any] | None) -> list[str]:
    """obsidian_reflection_journal.py（内省モード拡張・system-improvement-reflection）の当日分。"""
    if not report:
        return ["• status: `missing` / レポート未生成（Vertex AI未設定またはVault未接続の可能性）"]

    status = str(report.get("status") or "unknown")
    if status != "written":
        labels = {
            "skipped_no_theme_radar_report": "テーマレーダー未生成",
            "skipped_no_selected_theme": "Vaultにテーマ材料なし",
            "skipped_already_exists": "本日分は生成済み（下記は前回実行分）",
            "skipped_mana_hold": "Mana判定がholdのためVault書き込みを伴う内省ノート生成をスキップ",
            "skipped_mana_block": "Mana判定がblockのためVault書き込みを伴う内省ノート生成をスキップ",
            "skipped_mana_missing": "Mana判定レポート未生成のためVault書き込みを伴う内省ノート生成をスキップ",
        }
        lines = [f"• status: `{status}` / {labels.get(status, '内省ノート未生成')}"]
        action = _clean_text(report.get("mana_action_summary") or report.get("reason") or "", 160)
        if action:
            lines.append(f"• reason: {action}")
        return lines

    theme = _clean_text(report.get("theme") or "", 60)
    angle = _clean_text(report.get("angle") or "", 160)
    lines = [f"• テーマ: {theme}", f"• 切り口: {angle}"]
    problem = _clean_text(report.get("problem") or "", 200)
    insight = _clean_text(report.get("insight") or "", 200)
    solution = _clean_text(report.get("solution") or "", 200)
    if problem:
        lines.append(f"• 問題: {problem}")
    if insight:
        lines.append(f"• 気づき: {insight}")
    if solution:
        lines.append(f"• 解決: {solution}")
    return lines


def _shion_obsidian_curator_lines(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["• status: `missing` / 日次Curator診断なし"]
    if str(report.get("mode") or "") != "read_only_daily_obsidian_curator":
        return ["• status: `invalid` / Curator診断形式が想定外"]
    health = report.get("health") if isinstance(report.get("health"), dict) else {}
    summary = health.get("summary") if isinstance(health.get("summary"), dict) else {}
    buckets = summary.get("graph_buckets") if isinstance(summary.get("graph_buckets"), dict) else {}
    lines = [
        f"• isolated_used: `{buckets.get('isolated_but_used', 0)}` / degree0: `{summary.get('degree0_count', 0)}`",
    ]
    for item in (report.get("top_actions") or [])[:2]:
        if not isinstance(item, dict):
            continue
        target = _clean_text(item.get("target") or "unknown", 80)
        reason = _clean_text(item.get("reason") or "", 80)
        theme = _clean_text(item.get("theme") or "", 40)
        prefix = f"[{theme}] " if theme else ""
        lines.append(f"• {prefix}{target}: {reason}")
    return lines


def build_message(
    report: dict[str, Any],
    *,
    report_date: str,
    mana_report: dict[str, Any] | None = None,
    screening_terms_report: dict[str, Any] | None = None,
    judgment_asset_growth_report: dict[str, Any] | None = None,
    judgment_asset_field_review: dict[str, Any] | None = None,
    action_ledger_report: dict[str, Any] | None = None,
    reflection_journal_report: dict[str, Any] | None = None,
    shion_obsidian_curator_daily: dict[str, Any] | None = None,
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
            "*紫苑の自己提案*",
            *_shion_self_proposal_lines(report),
            "",
            "*Mana判定*",
            *_mana_lines(mana_report),
            "",
            "*審査用語監査*",
            *_screening_terms_lines(screening_terms_report),
            "",
            "*判断資産 実戦検証*",
            *_judgment_asset_growth_lines(judgment_asset_growth_report),
            *_judgment_asset_field_review_lines(judgment_asset_field_review),
            "",
            "*システム監視*",
            *_system_monitor_lines(),
            "",
            "*紫苑の行動ログ*",
            *_action_ledger_lines(action_ledger_report),
            "",
            "*内省モード: 今日の気づき*",
            *_reflection_journal_lines(reflection_journal_report),
            "",
            "*Obsidian Curator*",
            *_shion_obsidian_curator_lines(shion_obsidian_curator_daily),
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
    if state.get("last_sent_date") != report_date:
        return False
    if state.get("digest_version") != DIGEST_VERSION:
        # ハッシュ方式が変わった直後は新旧のダイジェストを比較できない。
        # 同じ日付に既に送信済みという事実だけで重複送信とみなす。
        return True
    return state.get("last_report_hash") == digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Send daily improvement report summary to Slack.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mana-report", type=Path, default=DEFAULT_MANA_REPORT)
    parser.add_argument("--screening-terms-report", type=Path, default=DEFAULT_SCREENING_TERMS_REPORT)
    parser.add_argument("--judgment-asset-growth-report", type=Path, default=DEFAULT_JUDGMENT_ASSET_GROWTH_REPORT)
    parser.add_argument("--judgment-asset-field-review", type=Path, default=DEFAULT_JUDGMENT_ASSET_FIELD_REVIEW)
    parser.add_argument("--action-ledger-report", type=Path, default=DEFAULT_ACTION_LEDGER_REPORT)
    parser.add_argument("--reflection-journal-report", type=Path, default=DEFAULT_REFLECTION_JOURNAL_REPORT)
    parser.add_argument("--shion-obsidian-curator-daily", type=Path, default=DEFAULT_SHION_OBSIDIAN_CURATOR_DAILY)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--webhook", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report = _read_json(args.report)
    mana_report = _read_optional_json(args.mana_report)
    screening_terms_report = _read_optional_json(args.screening_terms_report)
    judgment_asset_growth_report = _read_optional_json(args.judgment_asset_growth_report)
    judgment_asset_field_review = _read_optional_json(args.judgment_asset_field_review)
    action_ledger_report = _read_optional_json(args.action_ledger_report)
    reflection_journal_report = _read_optional_json(args.reflection_journal_report)
    shion_obsidian_curator_daily = _read_optional_json(args.shion_obsidian_curator_daily)
    payload = build_message(
        report,
        report_date=args.date,
        mana_report=mana_report,
        screening_terms_report=screening_terms_report,
        judgment_asset_growth_report=judgment_asset_growth_report,
        judgment_asset_field_review=judgment_asset_field_review,
        action_ledger_report=action_ledger_report,
        reflection_journal_report=reflection_journal_report,
        shion_obsidian_curator_daily=shion_obsidian_curator_daily,
    )
    # 実際にSlackへ送る本文でハッシュを取る。report/各サブレポートの生JSONには
    # attach_shion_self_proposals_to_report.py が毎回書く attached_at 等の実行時刻
    # フィールドが含まれ、本文が同じでも同日の再実行のたびにハッシュが変わって
    # 重複送信防止(should_skip)が機能しない不具合があった。
    digest = _text_hash(payload["text"])

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
    if not _is_plausible_slack_webhook(webhook_url):
        print("SLACK_WEBHOOK_URL looks incomplete or invalid; skipping Slack improvement report.")
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
            "digest_version": DIGEST_VERSION,
            "last_report": str(args.report),
        },
    )
    print(f"Slack improvement report sent for {args.date}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
