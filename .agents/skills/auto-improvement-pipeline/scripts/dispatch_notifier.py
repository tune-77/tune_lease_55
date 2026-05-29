"""承認待ちPRをpending_approvals.jsonlに記録し、Dispatch通知文を生成する."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PENDING_LOG_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "pending_approvals.jsonl"


def _ensure_log_dir() -> None:
    _PENDING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _build_dispatch_message(pr_number: int | None, pr_url: str, title: str) -> str:
    """Dispatch に送る通知文テンプレートを生成する（実際の送信は行わない）."""
    pr_label = f"PR #{pr_number}" if pr_number else "PR"
    return (
        f"承認待ちPRがあります：{pr_label}「{title}」 {pr_url} "
        f"　マージする場合は「{pr_label} マージして」と返信してください"
    )


def record_pending_approval(
    pr_number: int | None,
    pr_url: str,
    title: str,
    size: str = "approval",
) -> dict:
    """
    承認待ちPR情報を pending_approvals.jsonl に追記する。

    Returns:
        追記したレコード dict（dispatch_message フィールドを含む）
    """
    _ensure_log_dir()

    dispatch_message = _build_dispatch_message(pr_number, pr_url, title)
    record = {
        "pr_number": pr_number,
        "pr_url": pr_url,
        "title": title,
        "size": size,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "dispatch_message": dispatch_message,
        "notified": False,
    }

    try:
        with _PENDING_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("pending_approval 記録: PR #%s '%s'", pr_number, title)
        logger.info("Dispatch 通知文: %s", dispatch_message)
    except OSError as e:
        logger.warning("pending_approvals.jsonl 書き込みエラー: %s", e)

    return record


def notify_pending_approval(
    pr_number: int | None,
    pr_url: str,
    title: str,
    size: str = "approval",
) -> dict:
    """
    承認待ちPRを pending_approvals.jsonl に記録し、Dispatch通知文を生成する。

    approval の場合はコード生成・実装は行わない。
    Dispatch への実際の送信も行わない（通知文のみ生成してログに記録する）。

    Returns:
        {"recorded": bool, "dispatch_message": str, "record": dict}
    """
    record = record_pending_approval(pr_number, pr_url, title, size)

    return {
        "recorded": True,
        "dispatch_message": record.get("dispatch_message", ""),
        "record": record,
    }


def list_pending_approvals(unnotified_only: bool = False) -> list[dict]:
    """pending_approvals.jsonl から承認待ちリストを返す."""
    if not _PENDING_LOG_PATH.exists():
        return []
    records: list[dict] = []
    try:
        for line in _PENDING_LOG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if unnotified_only and rec.get("notified"):
                    continue
                records.append(rec)
            except json.JSONDecodeError:
                continue
    except OSError as e:
        logger.warning("pending_approvals.jsonl 読み込みエラー: %s", e)
    return records


if __name__ == "__main__":
    result = notify_pending_approval(
        pr_number=999,
        pr_url="https://github.com/tune-77/tune_lease_55/pull/999",
        title="テスト承認待ちPR",
        size="approval",
    )
    print(f"recorded={result['recorded']}")
    print(f"dispatch_message={result['dispatch_message']}")
    print(f"log path: {_PENDING_LOG_PATH}")
