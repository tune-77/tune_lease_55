"""GCS 書き込みロック機構（ロックファイル方式）

使い方:
    with GCSLock(target_file="shion/mind.json") as lock:
        bucket.blob("shion/mind.json").upload_from_string(data)
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage


GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
LOCK_PATH = "shion/.lock"
DEFAULT_TTL = 60
DEFAULT_TIMEOUT = 30
_RETRY_INTERVAL = 1.0


class GCSLockError(Exception):
    pass


class GCSLock:
    """GCS バケット上のロックファイルで書き込みを排他制御する。

    if_generation_match=0 で競合書き込みを防ぎ、TTL 超過のゾンビロックは
    自動的に強制解除する。
    """

    def __init__(
        self,
        bucket_name: str = GCS_BUCKET,
        lock_path: str = LOCK_PATH,
        ttl_seconds: int = DEFAULT_TTL,
        writer: str = "cloud-run",
        target_file: str = "shion/mind.json",
    ) -> None:
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._lock_path = lock_path
        self._ttl_seconds = ttl_seconds
        self._writer = writer
        self._target_file = target_file
        self._acquired = False

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _lock_blob(self) -> storage.Blob:
        return self._bucket.blob(self._lock_path)

    def _read_existing_lock(self) -> Optional[dict]:
        try:
            data = self._lock_blob().download_as_text()
        except Exception:
            return None  # ロックファイルが存在しない
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return {}  # 壊れたJSONはゾンビ扱い（_is_staleがTrueを返す）

    def _is_stale(self, lock_data: dict) -> bool:
        try:
            started_at = datetime.fromisoformat(lock_data["started_at"])
            ttl = int(lock_data.get("ttl_seconds", DEFAULT_TTL))
            return (self._now_utc() - started_at).total_seconds() > ttl
        except (KeyError, ValueError):
            return True  # 読めないロックはゾンビ扱い

    def _force_release(self) -> None:
        try:
            self._lock_blob().delete()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def acquire(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        """ロックを取得する。timeout 秒以内に取得できなければ GCSLockError。"""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            existing = self._read_existing_lock()
            if existing is not None:
                if self._is_stale(existing):
                    self._force_release()
                else:
                    time.sleep(_RETRY_INTERVAL)
                    continue

            lock_payload = json.dumps(
                {
                    "writer": self._writer,
                    "started_at": self._now_utc().isoformat(),
                    "ttl_seconds": self._ttl_seconds,
                    "target_file": self._target_file,
                },
                ensure_ascii=False,
            )
            try:
                self._lock_blob().upload_from_string(
                    lock_payload,
                    content_type="application/json",
                    if_generation_match=0,
                )
                self._acquired = True
                return
            except PreconditionFailed:
                # 別プロセスが同時に書き込んだ
                time.sleep(_RETRY_INTERVAL)

        raise GCSLockError(
            f"GCS ロック取得タイムアウト（{timeout}秒）: {self._lock_path}"
        )

    def release(self) -> None:
        """ロックを解除する。未取得の場合は何もしない（冪等）。"""
        if not self._acquired:
            return
        try:
            self._lock_blob().delete()
        finally:
            self._acquired = False

    def __enter__(self) -> "GCSLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False
