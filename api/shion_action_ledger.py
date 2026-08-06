"""紫苑 Agent Action Ledger（追記専用・監査用）。

planning/post_hackathon_shion_backlog.md §9.2 の仕様に基づく最小実装。
自律実行の権限を与えるものではなく、紫苑が「何を見て・何を判断し・
何を提案し・何をCodexへ渡したか」を後から追える監査ログを提供する。

個人情報・顧客情報・依頼文全文は保存しない（要約と参照IDのみ）。
呼び出し側は失敗してもパイプラインを止めないこと（本モジュールは例外を
送出し得るが、呼び出し側で握りつぶす方針は各スクリプト側に委ねる）。
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from runtime_paths import get_data_path

SCHEMA_VERSION = 1
LEDGER_PATH = Path(get_data_path("shion_action_ledger.jsonl"))
_LOCK = threading.RLock()

VALID_ACTIONS = {
    "daily_report",
    "system_watch",
    "improvement_classified",
    "codex_request_drafted",
    "user_approval_requested",
    "user_decision_recorded",
    "implementation_observed",
    "followup_reported",
}
VALID_RISK_LEVELS = {"low", "medium", "high"}
ActionType = Literal[
    "daily_report",
    "system_watch",
    "improvement_classified",
    "codex_request_drafted",
    "user_approval_requested",
    "user_decision_recorded",
    "implementation_observed",
    "followup_reported",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    return text[:limit].rstrip()


def log_action(
    action: ActionType,
    *,
    summary: str,
    mode: str = "improvement_pm",
    observed_sources: list[str] | None = None,
    risk_level: str = "low",
    requires_user_approval: bool = False,
    user_approved: bool | None = None,
    target: str | None = None,
    result: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """紫苑の行動を1件、追記専用JSONLへ記録する。

    要約・参照IDのみを保存し、個人情報や依頼文全文は保存しない前提。
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action: {action}")
    if risk_level not in VALID_RISK_LEVELS:
        raise ValueError(f"invalid risk_level: {risk_level}")

    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _now_iso(),
        "agent": "shion",
        "mode": _clean_text(mode, limit=40) or "improvement_pm",
        "action": action,
        "observed_sources": [_clean_text(s, limit=120) for s in (observed_sources or []) if _clean_text(s, limit=120)][:20],
        "summary": _clean_text(summary, limit=280),
        "risk_level": risk_level,
        "requires_user_approval": bool(requires_user_approval),
        "user_approved": user_approved,
        "target": _clean_text(target, limit=200) if target else None,
        "result": _clean_text(result, limit=120),
    }

    target_path = path or LEDGER_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, target_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


def read_actions(*, path: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """記録済みの行動を時系列順（古い→新しい）で返す。破損行はスキップする。"""
    target_path = path or LEDGER_PATH
    if not target_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with _LOCK:
        for line in target_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("action") in VALID_ACTIONS:
                entries.append(item)
    if limit is not None and limit >= 0:
        entries = entries[-limit:]
    return entries
