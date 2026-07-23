"""Track investigation promises Shion makes and execute them on the next turn.
Also writes countermeasures to the improvement dispatch queue for the daily pipeline.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

PENDING_PATH = get_data_path("shion_pending_tasks.json")
DISPATCH_QUEUE_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "dispatch_queue.jsonl"

# 約束は「調べます」等の丁寧表現でほぼ毎ターン記録される一方、消化（done化）は
# 同じ対話の次ターン先頭でしか起きない。会話が続かなかった約束は pending のまま
# 取り残されて件数が無限に膨らむ（「未完了タスク約70件」の主因）。放置された
# pending を一定日数で expired に落として open 件数を有界にする。
STALE_PENDING_DAYS = 14
# done/expired 履歴が際限なく積み上がるとファイルが肥大化するため、最新分のみ保持する。
MAX_HISTORY = 200

_COUNTERMEASURE_RE = re.compile(
    r"\*{0,2}③\s*対応策\*{0,2}[^\n]*\n(.*?)(?=\n\*{0,2}[①-⑩]|\Z)",
    re.DOTALL,
)

# 「紫苑がこれから調べる」という未来の自己コミットだけを検出する。従来は語幹だけ
# （確認し／調べ）や過去形（確認しました／調べました＝既に完了）、ユーザーへの依頼
# （ご確認ください／確認してください）まで拾ってしまい、丁寧表現でほぼ毎ターン
# pending が生成され件数が膨張していた。以下の絞り込みで過剰マッチを抑える。
#   - 未来・意志の丁寧形（…ます）に限定。過去形（…ました）は末尾が食い違い一致しない
#   - 依頼形（…ください）は「ます」で終わらないため一致しない
#   - 語幹だけ（確認し・調べ 等）の裸マッチを廃止
_PROMISE_PATTERNS = [
    r"(?:お)?調べ(?:し|してみ|てみ|させていただき)?ます",   # 調べます/お調べします/調べてみます
    r"(?:ご)?確認(?:し|いたし|してみ|させていただき)ます",   # 確認します/ご確認いたします/確認してみます
    r"検索(?:し|してみ|いたし)ます",
    r"(?:ご)?調査(?:し|いたし|してみ|させていただき)ます",
    r"(?:ご)?報告(?:し|いたし|させていただき)ます",
    r"お知らせ(?:し|いたし)ます",
    r"(?:後で|後ほど)(?:ご)?(?:確認|調査|調べ|報告)",
    r"次回(?:まで)?(?:に)?(?:ご)?(?:確認|調査|調べ|報告)",
    r"改めて(?:ご)?(?:報告|確認|調べ|調査)",
]
_PROMISE_RE = re.compile("|".join(_PROMISE_PATTERNS))


def _load() -> list[dict]:
    try:
        with open(PENDING_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(tasks: list[dict]) -> None:
    try:
        with open(PENDING_PATH, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# チャット→自律改善パイプラインの取り込み口。propose_quick_fix と同じファイルで、
# recursive_self_improvement.load_chat_quick_fix_intake が needs_review 候補として拾う。
CHAT_INTAKE_PATH = get_data_path("chat_quick_fix_intake.jsonl")


def dispatch_promise_to_improvement_log(topic: str) -> str | None:
    """調査約束の topic を改善ログ（自律改善パイプラインの取り込み口）へ起票する。

    紫苑が「調べます／確認します」と約束したまま立ち消えるのを防ぐため、topic を
    chat_quick_fix_intake.jsonl へ追記して needs_review 候補として追跡対象にする。
    target_module を付けない（＝quick_fix にならない）ため自動実行はされず、人間の
    レビュー対象＝改善ログに載るだけ。同一 topic は id で重複排除して肥大化を防ぐ。
    起票した id を返す（対象外/失敗時は None）。
    """
    import hashlib

    topic = (topic or "").strip()
    if not topic:
        return None
    rec_id = "promise_" + hashlib.md5(topic.encode("utf-8")).hexdigest()[:12]
    try:
        path = Path(CHAT_INTAKE_PATH)
        # 既に同一 topic を起票済みなら追記しない（追記形式ファイルの肥大化防止）。
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("id") == rec_id:
                        return rec_id
                except Exception:
                    continue
        record = {
            "id": rec_id,
            "title": topic[:120],
            "description": topic,
            "target_module": "",
            "source": "shion_promise",
            "proposed_at": datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return rec_id
    except Exception:
        return None


def extract_and_save_promises(user_message: str, shion_reply: str) -> list[str]:
    """If Shion's reply contains an investigation promise, record the topic as pending.

    加えて、約束した topic を改善ログ（自律改善パイプラインの取り込み口）へも起票し、
    約束が立ち消えず「後で自分で調べる／パイプラインが拾う」導線に必ず載るようにする。
    """
    if not _PROMISE_RE.search(shion_reply):
        return []
    topic = user_message.strip()[:300]
    task_id = str(uuid.uuid4())[:8]
    tasks = _load()
    tasks.append({
        "id": task_id,
        "topic": topic,
        "promised_at": datetime.now().isoformat(),
        "status": "pending",
    })
    _save(tasks)
    # 改善ログへ起票（追跡対象化）。失敗しても pending 記録は維持する。
    dispatch_promise_to_improvement_log(topic)
    return [task_id]


def _parse_ts(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _is_stale_pending(task: dict, now: datetime | None = None) -> bool:
    """pending のまま STALE_PENDING_DAYS 以上放置された約束か判定する。

    promised_at が無い/壊れている場合は経過日数を判断できないため、安全側に倒して
    「陳腐化していない」とみなす（誤って消化済み扱いにしない）。
    """
    if not isinstance(task, dict) or task.get("status") != "pending":
        return False
    ts = _parse_ts(task.get("promised_at"))
    if ts is None:
        return False
    return (now or datetime.now()) - ts >= timedelta(days=STALE_PENDING_DAYS)


def is_pending_open(task: dict, now: datetime | None = None) -> bool:
    """本当に「未完了（要対応）」な約束か判定する純関数（ディスクへ書き込まない）。

    done/expired は対象外、陳腐化した pending も対象外。件数レポート側はこの定義で
    数えることで、放置された古い約束や done 以外の異常ステータスで件数が水増しされない。
    """
    if not isinstance(task, dict) or task.get("status") != "pending":
        return False
    return not _is_stale_pending(task, now)


def reconcile_pending(now: datetime | None = None) -> list[dict[str, Any]]:
    """放置された pending を expired に落とし、履歴を上限まで刈り込んで保存する。

    書き込みは変更が生じた場合のみ。返り値は整理後の全タスク一覧。
    """
    now = now or datetime.now()
    tasks = _load()
    changed = False
    for task in tasks:
        if _is_stale_pending(task, now):
            task["status"] = "expired"
            task["expired_at"] = now.isoformat()
            changed = True

    pending = [t for t in tasks if isinstance(t, dict) and t.get("status") == "pending"]
    history = [t for t in tasks if not (isinstance(t, dict) and t.get("status") == "pending")]
    if len(history) > MAX_HISTORY:
        def _recency(task: dict) -> str:
            if not isinstance(task, dict):
                return ""
            return str(task.get("done_at") or task.get("expired_at") or task.get("promised_at") or "")

        history = sorted(history, key=_recency, reverse=True)[:MAX_HISTORY]
        changed = True

    reconciled = pending + history
    if changed:
        _save(reconciled)
    return reconciled


def get_pending_tasks() -> list[dict[str, Any]]:
    return [t for t in reconcile_pending() if t.get("status") == "pending"]


def attach_finding(task_id: str, finding: str, now: datetime | None = None) -> bool:
    """pending タスクに下調べ結果(finding)を付与する。付与できたら True。

    紫苑が約束を read-only ツールで自動下調べした結果を保存し、次の対話や日次で
    報告できるようにする。status は変えない（pending のまま追跡を継続）。
    """
    task_id = (task_id or "").strip()
    finding = (finding or "").strip()
    if not task_id or not finding:
        return False
    tasks = _load()
    updated = False
    for t in tasks:
        if isinstance(t, dict) and t.get("id") == task_id and t.get("status") == "pending":
            t["finding"] = finding[:600]
            t["investigated_at"] = (now or datetime.now()).isoformat()
            updated = True
    if updated:
        _save(tasks)
    return updated


def mark_done(task_ids: list[str]) -> None:
    if not task_ids:
        return
    tasks = _load()
    id_set = set(task_ids)
    for t in tasks:
        if t.get("id") in id_set:
            t["status"] = "done"
            t["done_at"] = datetime.now().isoformat()
    _save(tasks)


def _extract_countermeasure_block(shion_reply: str) -> str:
    """Extract the ③対応策 section from Shion's reply."""
    m = _COUNTERMEASURE_RE.search(shion_reply)
    if not m:
        return ""
    return m.group(1).strip()


def _lines_to_candidates(block: str, user_message: str) -> list[dict]:
    """Convert countermeasure text into dispatch_queue candidate entries."""
    candidates = []
    # Each bullet or numbered line becomes one candidate
    lines = [l.strip().lstrip("-・•*0123456789.）) ").strip() for l in block.splitlines()]
    lines = [l for l in lines if len(l) > 5]
    if not lines:
        # Whole block as single candidate
        lines = [block[:120]]
    for line in lines[:5]:  # max 5 candidates per reply
        candidates.append({
            "id": f"SHION-{str(uuid.uuid4())[:6].upper()}",
            "title": line[:80],
            "category": "shion",
            "reason": f"紫苑が調査した結果の対応策。元の問い: {user_message[:60]}",
            "source": "shion_dialogue",
        })
    return candidates


def save_countermeasures_to_dispatch(user_message: str, shion_reply: str) -> int:
    """If Shion's reply has a ③対応策 section, append it to dispatch_queue.jsonl.
    Returns the number of candidates written (0 if none found).
    """
    block = _extract_countermeasure_block(shion_reply)
    if not block:
        return 0
    candidates = _lines_to_candidates(block, user_message)
    if not candidates:
        return 0
    entry = {
        "type": "improvement_candidates",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "shion",
        "candidates": candidates,
        "message": "紫苑の調査から生成された対応策です。着手・保留・破棄を決めてください。",
    }
    try:
        DISPATCH_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DISPATCH_QUEUE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return 0
    return len(candidates)
