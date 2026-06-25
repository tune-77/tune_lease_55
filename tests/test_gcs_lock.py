"""GCS 書き込みロック機構の単体テスト（モックベース）

google-cloud-storage がインストールされていない環境でも実行できるよう、
sys.modules で google.cloud.storage を差し替えてからモジュールをインポートする。
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ------------------------------------------------------------------
# google.cloud.storage モック（未インストール環境対応）
# ------------------------------------------------------------------

def _setup_gcs_mock():
    """google.cloud.storage を sys.modules に差し込む。"""
    # google.api_core.exceptions.PreconditionFailed は installed なのでそのまま使う
    # google.cloud.storage だけ存在しないのでモック差し替え
    storage_mock = MagicMock()
    if "google.cloud.storage" not in sys.modules or not hasattr(
        sys.modules.get("google.cloud"), "storage"
    ):
        cloud_mod = sys.modules.get("google.cloud")
        if cloud_mod is None:
            cloud_mod = ModuleType("google.cloud")
            sys.modules["google.cloud"] = cloud_mod

        sys.modules["google.cloud.storage"] = storage_mock
        setattr(cloud_mod, "storage", storage_mock)

    return storage_mock


_setup_gcs_mock()


from scripts.gcs_lock import GCSLock, GCSLockError, DEFAULT_TTL  # noqa: E402
from google.api_core.exceptions import PreconditionFailed  # noqa: E402 (installed)


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _lock_json(
    started_at: Optional[datetime] = None,
    ttl: int = DEFAULT_TTL,
    writer: str = "test",
) -> str:
    return json.dumps({
        "writer": writer,
        "started_at": (started_at or _utc_now()).isoformat(),
        "ttl_seconds": ttl,
        "target_file": "shion/mind.json",
    })


@pytest.fixture
def mock_gcs():
    """storage.Client をモックし、(mock_bucket, mock_blob) を返す。"""
    with patch("scripts.gcs_lock.storage.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        yield mock_bucket, mock_blob


# ------------------------------------------------------------------
# 正常取得・解除
# ------------------------------------------------------------------

class TestAcquire:
    def test_success_when_no_lock_exists(self, mock_gcs):
        """ロックが存在しない場合、正常に取得できる。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")

        lock = GCSLock(bucket_name="test-bucket")
        lock.acquire(timeout=5)

        mock_blob.upload_from_string.assert_called_once()
        uploaded = json.loads(mock_blob.upload_from_string.call_args[0][0])
        assert uploaded["writer"] == "cloud-run"
        assert uploaded["ttl_seconds"] == DEFAULT_TTL
        assert lock._acquired is True

    def test_acquired_flag_is_false_before_acquire(self, mock_gcs):
        """acquire() 前は _acquired が False。"""
        lock = GCSLock(bucket_name="test-bucket")
        assert lock._acquired is False


class TestRelease:
    def test_release_deletes_lock_blob(self, mock_gcs):
        """release() でロック BLOB が削除される。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")

        lock = GCSLock(bucket_name="test-bucket")
        lock.acquire(timeout=5)
        lock.release()

        mock_blob.delete.assert_called_once()
        assert lock._acquired is False

    def test_release_is_idempotent(self, mock_gcs):
        """release() を複数回呼んでも delete は 1 回しか呼ばれない。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")

        lock = GCSLock(bucket_name="test-bucket")
        lock.acquire(timeout=5)
        lock.release()
        lock.release()

        assert mock_blob.delete.call_count == 1

    def test_release_without_acquire_is_safe(self, mock_gcs):
        """acquire() せずに release() を呼んでも例外にならない。"""
        _, mock_blob = mock_gcs
        lock = GCSLock(bucket_name="test-bucket")
        lock.release()
        mock_blob.delete.assert_not_called()


# ------------------------------------------------------------------
# 二重取得エラー
# ------------------------------------------------------------------

class TestLockConflict:
    def test_raises_gcs_lock_error_on_timeout(self, mock_gcs):
        """他プロセスがロックを保持中はタイムアウトで GCSLockError。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.return_value = _lock_json()
        mock_blob.upload_from_string.side_effect = PreconditionFailed("conflict")

        lock = GCSLock(bucket_name="test-bucket")
        with pytest.raises(GCSLockError, match="タイムアウト"):
            lock.acquire(timeout=1)

        assert lock._acquired is False

    def test_upload_precondition_failed_triggers_retry(self, mock_gcs):
        """upload で PreconditionFailed が来てもリトライし、次回成功する。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")
        mock_blob.upload_from_string.side_effect = [
            PreconditionFailed("conflict"),
            None,  # 2 回目は成功
        ]

        lock = GCSLock(bucket_name="test-bucket")
        lock.acquire(timeout=5)

        assert mock_blob.upload_from_string.call_count == 2
        assert lock._acquired is True


# ------------------------------------------------------------------
# TTL 超過ロックの強制解除
# ------------------------------------------------------------------

class TestStaleLock:
    def test_stale_lock_is_force_released_and_reacquired(self, mock_gcs):
        """TTL を超過したロックは強制解除されて再取得できる。"""
        _, mock_blob = mock_gcs
        old_time = _utc_now() - timedelta(seconds=120)
        stale_json = _lock_json(started_at=old_time, ttl=60)

        # 1回目の read: 古いロック / 2回目の read: 存在しない
        mock_blob.download_as_text.side_effect = [stale_json, Exception("Not Found")]

        lock = GCSLock(bucket_name="test-bucket")
        lock.acquire(timeout=5)

        mock_blob.delete.assert_called()
        mock_blob.upload_from_string.assert_called_once()
        assert lock._acquired is True

    def test_lock_with_bad_json_is_treated_as_stale(self, mock_gcs):
        """JSON が壊れているロックはゾンビ扱いで強制解除。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = [
            "this is not json {{{",
            Exception("Not Found"),
        ]

        lock = GCSLock(bucket_name="test-bucket")
        lock.acquire(timeout=5)

        mock_blob.delete.assert_called()
        assert lock._acquired is True

    def test_lock_with_missing_started_at_is_stale(self, mock_gcs):
        """started_at フィールドがないロックはゾンビ扱い。"""
        _, mock_blob = mock_gcs
        bad_lock = json.dumps({"writer": "ghost", "ttl_seconds": 60})
        mock_blob.download_as_text.side_effect = [bad_lock, Exception("Not Found")]

        lock = GCSLock(bucket_name="test-bucket")
        lock.acquire(timeout=5)

        mock_blob.delete.assert_called()
        assert lock._acquired is True


# ------------------------------------------------------------------
# コンテキストマネージャ
# ------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_acquires_and_releases(self, mock_gcs):
        """with 文でロック取得と解除が自動実行される。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")

        with GCSLock(bucket_name="test-bucket"):
            pass

        mock_blob.upload_from_string.assert_called_once()
        mock_blob.delete.assert_called_once()

    def test_context_manager_releases_even_on_exception(self, mock_gcs):
        """with ブロック内で例外が発生してもロックは解除される。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")

        with pytest.raises(RuntimeError, match="処理エラー"):
            with GCSLock(bucket_name="test-bucket"):
                raise RuntimeError("処理エラー")

        mock_blob.delete.assert_called_once()

    def test_context_manager_returns_lock_instance(self, mock_gcs):
        """as 節で GCSLock インスタンスを受け取れる。"""
        _, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")

        with GCSLock(bucket_name="test-bucket") as lock:
            assert isinstance(lock, GCSLock)
            assert lock._acquired is True

        assert lock._acquired is False


# ------------------------------------------------------------------
# 設定・環境変数
# ------------------------------------------------------------------

class TestConfig:
    def test_custom_bucket_and_lock_path(self, mock_gcs):
        """バケット名・ロックパスをカスタマイズできる。"""
        mock_bucket, mock_blob = mock_gcs
        mock_blob.download_as_text.side_effect = Exception("Not Found")

        lock = GCSLock(
            bucket_name="my-custom-bucket",
            lock_path="custom/.lock",
            writer="my-service",
            target_file="shion/world_view.json",
        )
        lock.acquire(timeout=5)

        mock_bucket.blob.assert_called_with("custom/.lock")
        uploaded = json.loads(mock_blob.upload_from_string.call_args[0][0])
        assert uploaded["writer"] == "my-service"
        assert uploaded["target_file"] == "shion/world_view.json"

    def test_default_env_bucket(self):
        """GCS_BUCKET 環境変数が gcs_lock モジュールのデフォルトに反映される。"""
        import os
        from importlib import reload
        import scripts.gcs_lock as gcs_lock_module

        with patch.dict(os.environ, {"GCS_BUCKET": "env-bucket-override"}):
            reload(gcs_lock_module)
            assert gcs_lock_module.GCS_BUCKET == "env-bucket-override"

        reload(gcs_lock_module)  # テスト後にリセット
