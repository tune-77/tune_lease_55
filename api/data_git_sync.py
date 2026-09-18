"""Shared data-git synchronization used by the app and batch routers."""

from __future__ import annotations

import asyncio
import datetime
import os
import shlex
import shutil

from api.db_connection import current_backend, get_connection, placeholder
from runtime_paths import get_db_path


LEASE_DB_PATH = get_db_path()
DATA_GIT_DIR = os.environ.get("DATA_GIT_DIR", "/app/data-git")
_git_lock = asyncio.Lock()


def _db_available() -> bool:
    return current_backend() == "postgresql" or os.path.exists(LEASE_DB_PATH)


def init_sync_log_table() -> None:
    if not _db_available():
        return
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            if current_backend() == "postgresql":
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sync_log (
                        id SERIAL PRIMARY KEY,
                        pushed_at TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        error TEXT
                    )
                    """
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sync_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pushed_at TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        error TEXT
                    )
                    """
                )
    except Exception as exc:
        print(f"[sync_log] テーブル作成失敗（非致命的）: {exc}")


def record_sync_log(success: bool, error: str = "") -> None:
    if not _db_available():
        return
    ph = placeholder()
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO sync_log (pushed_at, success, error) VALUES ({ph}, {ph}, {ph})",
                (datetime.datetime.now(datetime.timezone.utc).isoformat(), 1 if success else 0, error),
            )
    except Exception as exc:
        print(f"[sync_log] 記録失敗（非致命的）: {exc}")


async def git_push_db() -> None:
    """Copy the current DB and mind state to data-git, then push them."""
    if not os.path.isdir(os.path.join(DATA_GIT_DIR, ".git")):
        return
    db_name = os.path.basename(LEASE_DB_PATH)
    db_dst = os.path.join(DATA_GIT_DIR, "data", db_name)
    mind_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mind.json")
    mind_dst = os.path.join(DATA_GIT_DIR, "data", "mind.json")
    success = False
    error_msg = ""
    try:
        async with _git_lock:
            if os.path.exists(LEASE_DB_PATH):
                shutil.copy2(LEASE_DB_PATH, db_dst)
            if os.path.exists(mind_src):
                shutil.copy2(mind_src, mind_dst)
            db_name_q = shlex.quote(f"data/{db_name}")
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                f"git add {db_name_q} data/mind.json 2>/dev/null; "
                "git diff --cached --quiet || git commit -m 'auto: update from cloud-run'; git push",
                cwd=DATA_GIT_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            success = proc.returncode == 0
            error_msg = stderr.decode(errors="replace") if not success else ""
    except asyncio.TimeoutError:
        error_msg = "git push timeout"
    except Exception as exc:
        error_msg = str(exc)
    record_sync_log(success, error_msg)
    if not success:
        print(f"[git-push] 失敗: {error_msg}")
