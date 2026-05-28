"""承認待ちPRをpending_approvals.jsonlに記録する."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_PENDING_LOG_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "pending_approvals.jsonl"


def _ensure_log_dir() -> None:
    _PENDING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def record_pending_approval(
    pr_number: int | None,
    pr_url: str,
    title: str,
    size: str = "approval",
) -> dict:
    """
    承認待ちPR情報を pending_approvals.jsonl に追記する。

    Returns:
        追記したレコード dict
    """
    _ensure_log_dir()

    record = {
        "pr_number": pr_number,
        "pr_url": pr_url,
        "title": title,
        "size": size,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "notified": False,
    }

    try:
        with _PENDING_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("pending_approval 記録: PR #%s '%s'", pr_number, title)
    except OSError as e:
        logger.warning("pending_approvals.jsonl 書き込みエラー: %s", e)

    return record


def send_dispatch_message(pr_number: int | None, pr_url: str, title: str) -> bool:
    """
    claude CLI を使って Dispatch にメッセージを送る。
    claude CLI が存在しない場合は False を返す（pending_approvals.jsonl への記録は別途行う）。

    Returns:
        True: 送信成功, False: スキップ
    """
    claude_path = _find_claude_bin()
    if not claude_path:
        logger.info("claude CLI が見つからないため Dispatch 通知をスキップ")
        return False

    pr_label = f"PR #{pr_number}" if pr_number else "PR"
    message = (
        f"承認待ちPRがあります：{pr_label}「{title}」 {pr_url} "
        f"　マージする場合は「{pr_label} マージして」と返信してください"
    )

    try:
        result = subprocess.run(
            [claude_path, "--print", message],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Dispatch 通知送信成功: %s", pr_label)
            return True
        else:
            logger.warning("Dispatch 通知失敗 (rc=%d): %s", result.returncode, result.stderr[:200])
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning("Dispatch 通知例外: %s", e)
        return False


def notify_pending_approval(
    pr_number: int | None,
    pr_url: str,
    title: str,
    size: str = "approval",
) -> dict:
    """
    承認待ちPRを記録し、可能なら Dispatch にも通知する。

    Returns:
        {"recorded": bool, "dispatched": bool, "record": dict}
    """
    record = record_pending_approval(pr_number, pr_url, title, size)
    dispatched = send_dispatch_message(pr_number, pr_url, title)

    if dispatched:
        # notified フラグを更新（最終行を書き直す）
        record["notified"] = True
        _update_last_record_notified()

    return {
        "recorded": True,
        "dispatched": dispatched,
        "record": record,
    }


def _update_last_record_notified() -> None:
    """pending_approvals.jsonl の最終行の notified を True に更新する."""
    try:
        if not _PENDING_LOG_PATH.exists():
            return
        lines = _PENDING_LOG_PATH.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        last = json.loads(lines[-1])
        last["notified"] = True
        lines[-1] = json.dumps(last, ensure_ascii=False)
        _PENDING_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning("notified フラグ更新失敗: %s", e)


def _find_claude_bin() -> str | None:
    """claude CLI のパスを which で取得する."""
    try:
        result = subprocess.run(
            ["which", "claude"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception:
        return None


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
    # 簡易動作確認
    result = notify_pending_approval(
        pr_number=999,
        pr_url="https://github.com/tune-77/tune_lease_55/pull/999",
        title="テスト承認待ちPR",
        size="approval",
    )
    print(f"recorded={result['recorded']}, dispatched={result['dispatched']}")
    print(f"log path: {_PENDING_LOG_PATH}")
