#!/usr/bin/env python3
"""チャット起票（chat_quick_fix_intake）の即時バックグラウンド実行。

propose_quick_fix が起票した intake レコードのうち、既存の codex-safe 判定
（build_codex_auto_queue.is_blocked / is_codex_safe、auto_fix_policy が単一の真実源）を
満たすものだけを、待たせずにバックグラウンドで即時実行する。実行そのものは
execute_codex_queue.py のガード（キルスイッチ・日次実行上限）と run_item() を
そのまま再利用し、新しい実行ロジックは書かない。

実行済みIDは data/chat_quick_fix_executed.json に記録し、翌日の日次パイプライン
（recursive_self_improvement.load_chat_quick_fix_intake）が同じ候補を二重に
拾わないようにする。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

_SKILL_SCRIPTS = _REPO_ROOT / ".agents" / "skills" / "auto-improvement-pipeline" / "scripts"
if _SKILL_SCRIPTS.is_dir() and str(_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SKILL_SCRIPTS))

from build_codex_auto_queue import is_blocked, is_codex_safe, queue_item, refresh_auto_fix_policy
from execute_codex_queue import (
    _get_gemini_api_key,
    _guard_daily_limit,
    _guard_disabled,
    count_executed_today,
    dump_json,
    load_json,
    record_status,
    run_item,
)

_STATE_LOCK = threading.Lock()


def _immediate_execution_enabled() -> bool:
    """既定はOFF。明示的に CHAT_QUICK_FIX_IMMEDIATE_EXECUTION=1 を設定した環境でのみ、

    propose_quick_fix から本物の claude --print サブプロセスが起動しうる即時実行を許可する。
    未設定のまま（CI・ローカルテスト・通常のチャット利用を含む）では常に起票のみで完結し、
    従来どおり日次パイプラインが拾う。誤って本番以外でエージェントを起動しないための必須ガード。
    """
    return str(os.environ.get("CHAT_QUICK_FIX_IMMEDIATE_EXECUTION") or "").strip() in {"1", "true", "yes"}


def _executed_ids_path(root: Path) -> Path:
    return root / "data" / "chat_quick_fix_executed.json"


def load_executed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(x) for x in data} if isinstance(data, list) else set()
    except (OSError, json.JSONDecodeError):
        return set()


def _mark_executed(rec_id: str, path: Path) -> None:
    with _STATE_LOCK:
        ids = load_executed_ids(path)
        ids.add(rec_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(ids), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_in_background(item: dict[str, Any], root: Path, gemini_api_key: str) -> None:
    """バックグラウンドスレッドで1件実行する。チャット応答はブロックしない。"""
    rev_id = str(item.get("id") or "")
    try:
        record_status(root, rev_id, "running")
        entry = run_item(item, gemini_api_key=gemini_api_key)

        date_tag = dt.date.today().strftime("%Y%m%d")
        result_path = root / "reports" / f"codex_queue_result_{date_tag}_chat.json"
        report = (
            load_json(result_path)
            if result_path.exists()
            else {
                "executed_at": dt.datetime.now().isoformat(timespec="seconds"),
                "queue_file": "chat_quick_fix",
                "results": [],
            }
        )
        report.setdefault("results", []).append(entry)
        dump_json(result_path, report)

        status = "completed_pending_review" if entry.get("exit_code") == 0 else "failed"
        detail = str(entry.get("stdout") or entry.get("stderr") or "")[:200]
        record_status(root, rev_id, status, detail=detail)
    finally:
        _mark_executed(rev_id, _executed_ids_path(root))


def start_execution(record: dict[str, Any], root: Path = _REPO_ROOT) -> dict[str, Any]:
    """intake レコード1件を判定し、安全ならバックグラウンド実行を開始する。

    実行を開始しない場合は理由を返す（起票自体は残り、次回の日次パイプラインが拾う）。
    """
    if not _immediate_execution_enabled():
        return {
            "execution": "queued_for_batch",
            "reason": "即時実行は無効です（CHAT_QUICK_FIX_IMMEDIATE_EXECUTION が未設定）",
        }

    rec_id = str(record.get("id") or "")
    if not rec_id:
        return {"execution": "skipped", "reason": "idが空です"}
    if rec_id in load_executed_ids(_executed_ids_path(root)):
        return {"execution": "skipped", "reason": "既に実行済みです"}

    item = refresh_auto_fix_policy(dict(record), root)

    blocked, block_reason = is_blocked(item)
    if blocked:
        return {"execution": "queued_for_batch", "reason": f"安全判定で保留: {block_reason}"}
    if not is_codex_safe(item):
        return {
            "execution": "queued_for_batch",
            "reason": "codex-safe条件を満たさないため、次回の日次パイプラインで人間レビューに回ります",
        }
    if _guard_disabled():
        return {"execution": "queued_for_batch", "reason": "CODEX_QUEUE_DISABLED によりキュー実行が停止中です"}

    date_tag = dt.date.today().strftime("%Y%m%d")
    if count_executed_today(root, date_tag) >= _guard_daily_limit():
        return {"execution": "queued_for_batch", "reason": "本日の自動修正実行枠の上限に達しています"}

    queued = queue_item(item)
    gemini_api_key = _get_gemini_api_key(root)
    thread = threading.Thread(
        target=_run_in_background,
        args=(queued, root, gemini_api_key),
        name=f"chat-quick-fix-{rec_id}",
        daemon=True,
    )
    thread.start()
    return {"execution": "started", "id": rec_id}
