#!/usr/bin/env python3
"""
scripts/notify_pipeline_alerts.py

台帳（ledger_rules.json）に記録された改善パイプライン自体の障害検出
（category=pipeline_fix・承認待ち）を Slack に push 通知する。

従来は改善ログ画面を開かないと気づけなかった「自動適用の土台が
止まっている」障害を、検出当日に知らせる。同じ rev_id は一度しか
通知しない（通知済みリスト: data/pipeline_alert_notified.json）。

Webhook URL の取得順: 環境変数 SLACK_WEBHOOK_URL → .streamlit/secrets.toml
（Slackトークンのハードコード禁止: .claude/rules/security.md）

使い方:
  python scripts/notify_pipeline_alerts.py --dry-run
  python scripts/notify_pipeline_alerts.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "api" / "rule_engine" / "ledger_rules.json"
STATE_PATH = REPO_ROOT / "data" / "pipeline_alert_notified.json"
# 監視先行率KPI用: Slack通知の発報時刻を追記記録する（analyze_shion_pm_quality が読む）
NOTIFY_LOG_PATH = REPO_ROOT / "data" / "pipeline_alert_notify_log.jsonl"


def append_notify_log(rev_ids: list[str]) -> None:
    import datetime as _dt

    try:
        NOTIFY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NOTIFY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "rev_ids": rev_ids},
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass  # ログ失敗で通知本体を止めない


def get_webhook_url() -> str:
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if url:
        return url
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            return str(data.get("SLACK_WEBHOOK_URL", "")).strip()
        except Exception:
            return ""
    return ""


def load_alerts() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        rules = json.load(f)
    return [
        r for r in rules
        if isinstance(r, dict) and r.get("category") == "pipeline_fix" and r.get("pending_review")
    ]


def load_notified() -> set[str]:
    if not STATE_PATH.exists():
        return set()
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {str(x) for x in data} if isinstance(data, list) else set()
    except (json.JSONDecodeError, OSError):
        return set()


def save_notified(notified: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(notified), f, ensure_ascii=False, indent=2)
        f.write("\n")


def send_slack(webhook_url: str, alerts: list[dict]) -> bool:
    lines = ["⚠️ *改善パイプライン障害検出*", ""]
    for alert in alerts:
        lines.append(f"• `{alert.get('rev_id')}`: {alert.get('description', '')}")
    lines.append("")
    lines.append("パイプラインが失敗している間、承認済みルールの自動適用や朝の改善レポート生成が止まっている可能性があります。改善ログ画面で確認してください。")
    payload = json.dumps({"text": "\n".join(lines)}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"⚠️ Slack送信エラー: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="パイプライン障害検出のSlack通知")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="送信せずに結果を表示")
    mode.add_argument("--apply", action="store_true", help="実際に通知する")
    args = parser.parse_args()

    alerts = load_alerts()
    notified = load_notified()
    new_alerts = [a for a in alerts if str(a.get("rev_id")) not in notified]

    label = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"📣 notify_pipeline_alerts — {label}")
    print(f"   検出 {len(alerts)} 件 / 未通知 {len(new_alerts)} 件")
    for alert in new_alerts:
        print(f"   🚨 {alert.get('rev_id')}: {str(alert.get('description', ''))[:80]}")

    if not new_alerts or args.dry_run:
        return 0

    webhook_url = get_webhook_url()
    if not webhook_url.startswith("https://hooks.slack.com/"):
        print("   ⚠️ SLACK_WEBHOOK_URL が未設定のため通知をスキップします")
        return 0

    if send_slack(webhook_url, new_alerts):
        notified.update(str(a.get("rev_id")) for a in new_alerts)
        save_notified(notified)
        append_notify_log([str(a.get("rev_id")) for a in new_alerts])
        print(f"   ✅ Slackに通知しました（{len(new_alerts)}件）")
        print(f"      通知済み記録: {STATE_PATH}")
    else:
        print("   ❌ Slack通知に失敗しました（次回実行時に再送します）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
