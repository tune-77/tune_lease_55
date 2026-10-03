"""黙った失敗（握りつぶし・投げっぱなし保存・黙った切り詰め・既定値への差し替え）の記録。

挙動は変えずに「起きた」ことだけを data/silent_failures.jsonl に1行で残し、
AURION CORE 朝報の上部（`morning_report_lines`）で件数・新種・重要部品を警告する。

記録するのは部品名・種類・例外の型名・呼び出し箇所・呼び出し側が渡す短いラベルだけ。
例外メッセージや本文は個人情報を含み得るので残さない。

部品名は `<領域>.<モジュール>.<箇所>` とする。先頭の領域が CRITICAL_DOMAINS に入るもの
（利用者の答え・審査結果・判断資産・記憶・バックアップ・個人情報）は1件でも朝報に出す。

    from silent_failure_log import record_silent_failure
    try:
        save()
    except Exception as exc:  # noqa: BLE001 - 保存失敗で応答を止めない
        record_silent_failure("memory.chat_memory.save_summary", "save_failed", exc)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

_logger = logging.getLogger("silent_failure")

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "data" / "silent_failures.jsonl"
KINDS = ("swallowed", "save_failed", "fallback", "truncated", "timeout", "subprocess_failed")
CRITICAL_DOMAINS = ("answer", "scoring", "judgment", "memory", "backup", "privacy")
_MAX_BYTES = 5 * 1024 * 1024
_REPEAT_WINDOW_S = 60.0

_lock = threading.Lock()
_last_write: dict[tuple[str, str, str], float] = {}
_suppressed: Counter[tuple[str, str, str]] = Counter()


def log_path() -> Path | None:
    """書き込み先。`SILENT_FAILURE_LOG_PATH=off` なら記録しない。"""
    raw = str(os.environ.get("SILENT_FAILURE_LOG_PATH") or "").strip()
    if raw.lower() in {"off", "0", "false", "none"}:
        return None
    if raw:
        return Path(raw)
    data_dir = os.environ.get("DATA_DIR") or os.environ.get("LEASE_DATA_DIR")
    return Path(data_dir) / "silent_failures.jsonl" if data_dir else DEFAULT_LOG_PATH


def is_critical(component: str) -> bool:
    return str(component).split(".", 1)[0] in CRITICAL_DOMAINS


def _caller(depth: int = 2) -> str:
    try:
        frame = sys._getframe(depth)
        path = Path(frame.f_code.co_filename)
        root = Path(__file__).resolve().parent
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            rel = Path(path.name)
        return f"{rel}:{frame.f_lineno}"
    except Exception:  # noqa: BLE001 - 呼び出し箇所が取れなくても記録は続ける
        return ""


def record_silent_failure(component: str, kind: str, exc: BaseException | None = None, *, detail: str = "", stacklevel: int = 1) -> None:
    """黙った失敗を1行記録する。どんな場合も例外を投げない。

    detail は「どの経路で何を代わりにしたか」程度の固定ラベル（最大120字）。本文や利用者の入力は入れない。
    同じ箇所・同じ型が60秒以内に続いた分は数だけ数え、次の記録の repeat に載せる。
    """
    try:
        exc_type = type(exc).__name__ if exc is not None else ""
        where = _caller(stacklevel + 1)
        key = (component, kind, exc_type)
        now = time.monotonic()
        with _lock:
            last = _last_write.get(key)
            if last is not None and now - last < _REPEAT_WINDOW_S:
                _suppressed[key] += 1
                return
            _last_write[key] = now
            repeat = _suppressed.pop(key, 0)
        entry: dict[str, Any] = {
            "ts": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "component": component,
            "kind": kind,
            "exc_type": exc_type,
            "where": where,
            "critical": is_critical(component),
        }
        if detail:
            entry["detail"] = str(detail)[:120]
        if repeat:
            entry["repeat"] = repeat
        _logger.warning("silent failure: %s %s %s %s", component, kind, exc_type, where)
        path = log_path()
        if path is None:
            return
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > _MAX_BYTES:
                path.replace(path.with_suffix(path.suffix + ".1"))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - 記録の失敗で本処理を止めない（ここだけは記録先がない）
        pass


def clip(text: str, limit: int, component: str) -> str:
    """`text[:limit]` と同じ値を返す。切った時だけ truncated を記録する（長さだけで本文は残さない）。"""
    if len(text) > limit:
        record_silent_failure(component, "truncated", detail=f"{len(text)}→{limit}字", stacklevel=2)
    return text[:limit]


def _read_rows(path: Path, since: dt.datetime) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if dt.datetime.fromisoformat(row["ts"]) >= since:
                rows.append(row)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    return rows


def morning_report_lines(path: Path | None = None, *, now: dt.datetime | None = None, threshold: int = 20) -> list[str]:
    """直近24時間の黙った失敗を、件数が多い・新しい種類・重要部品のときだけ朝報の上部に出す。"""
    path = path or log_path()
    if path is None or not path.exists():
        return []
    now = now or dt.datetime.now().astimezone()
    try:
        rows = _read_rows(path, now - dt.timedelta(days=8))
    except OSError as exc:
        return [f"- ⚠️ 黙った失敗の記録を読めない `{type(exc).__name__}`"]
    since = now - dt.timedelta(hours=24)
    recent, older = [], set()
    for row in rows:
        key = (row.get("component", ""), row.get("kind", ""))
        if dt.datetime.fromisoformat(row["ts"]) >= since:
            recent.append(row)
        else:
            older.add(key)
    if not recent:
        return []
    counts: Counter[tuple[str, str]] = Counter()
    for row in recent:
        counts[(row.get("component", ""), row.get("kind", ""))] += 1 + int(row.get("repeat") or 0)
    total = sum(counts.values())
    critical = {k: v for k, v in counts.items() if is_critical(k[0])}
    new = {k: v for k, v in counts.items() if k not in older}
    if total < threshold and not critical and not new:
        return []

    def fmt(items: dict[tuple[str, str], int]) -> str:
        top = sorted(items.items(), key=lambda kv: -kv[1])[:4]
        rest = len(items) - len(top)
        return ", ".join(f"`{c}` {k}×{n}" for (c, k), n in top) + (f" ほか{rest}種" if rest > 0 else "")

    lines = [f"- ⚠️ 黙った失敗（直近24h）: {total}件・{len(counts)}種（記録: data/silent_failures.jsonl）"]
    if critical:
        lines.append(f"  - 重要部品: {fmt(critical)}")
    if new:
        lines.append(f"  - 新しい種類: {fmt(new)}")
    if not critical and not new:
        lines.append(f"  - 多い順: {fmt(dict(counts))}")
    return lines


LAUNCHD_PREFIXES = ("com.tunelease.", "com.lease.")


def launchd_failure_lines(listing: str | None = None) -> list[str]:
    """`launchctl list` で直近の終了コードが0以外の本プロジェクトのジョブを朝報の上部に出す。

    launchd は失敗しても終了コードを保持するだけで、どこにも通知しない（週次ジョブが何週も
    黙って止まっていた）。macOS 以外・launchctl が無い環境では何も出さない。
    """
    if listing is None:
        import shutil
        import subprocess

        if not shutil.which("launchctl"):
            return []
        try:
            listing = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=10, check=True).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"- ⚠️ launchd ジョブの状態を読めない `{type(exc).__name__}`"]
    failed = []
    for line in listing.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3 or not parts[2].startswith(LAUNCHD_PREFIXES):
            continue
        pid, status, label = parts
        if status.lstrip("-").isdigit() and int(status) != 0:
            failed.append(f"`{label.removeprefix('com.tunelease.')}`={status}" + ("（常駐・再起動を繰り返している）" if pid != "-" else ""))
    if not failed:
        return []
    return [f"- ⚠️ launchd ジョブの直近の終了コードが0以外: {', '.join(failed)}（ログは各 plist の StandardOutPath / StandardErrorPath）"]
