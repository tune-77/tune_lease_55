"""GCS → ローカル同期スクリプトの単体テスト（モックベース）"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from typing import Optional


# ------------------------------------------------------------------
# google.cloud.storage モック（未インストール環境対応）
# ------------------------------------------------------------------

def _setup_gcs_mock():
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

from scripts.gcs_to_local_sync import sync_file, main  # noqa: E402


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _make_blob(mtime: Optional[datetime] = None, reload_raises: bool = False) -> MagicMock:
    blob = MagicMock()
    if reload_raises:
        blob.reload.side_effect = Exception("Not Found")
    else:
        blob.reload.return_value = None
        blob.metadata = {"mtime": mtime.isoformat()} if mtime else {}
    return blob


# ------------------------------------------------------------------
# test_download_when_gcs_newer
# ------------------------------------------------------------------

class TestDownloadWhenGcsNewer:
    def test_downloads_file(self, tmp_path):
        """GCS の mtime がローカルより新しい場合にダウンロードされる。"""
        now = datetime.now(timezone.utc)
        gcs_mtime = now
        local_mtime = now - timedelta(minutes=10)

        # ローカルファイルを古い mtime で作成
        local_file = tmp_path / "mind.json"
        local_file.write_text("{}")
        import os
        ts = local_mtime.timestamp()
        os.utime(local_file, (ts, ts))

        blob = _make_blob(mtime=gcs_mtime)

        with patch("scripts.gcs_to_local_sync.storage.Client") as _:
            result = sync_file(
                bucket=MagicMock(**{"blob.return_value": blob}),
                gcs_path="shion/mind.json",
                local_dir=tmp_path,
                required=True,
            )

        blob.download_to_filename.assert_called_once_with(str(tmp_path / "mind.json"))
        assert "[DL]" in result

    def test_downloads_when_no_local_file(self, tmp_path):
        """ローカルファイルが存在しない場合もダウンロードされる。"""
        blob = _make_blob(mtime=datetime.now(timezone.utc))
        bucket = MagicMock(**{"blob.return_value": blob})

        result = sync_file(bucket, "shion/world_view.json", tmp_path, required=True)

        blob.download_to_filename.assert_called_once()
        assert "[DL]" in result


# ------------------------------------------------------------------
# test_skip_when_local_newer
# ------------------------------------------------------------------

class TestSkipWhenLocalNewer:
    def test_skips_when_local_is_newer(self, tmp_path):
        """ローカルファイルの mtime が GCS 以上の場合はスキップ。"""
        now = datetime.now(timezone.utc)
        gcs_mtime = now - timedelta(minutes=5)
        local_mtime = now

        local_file = tmp_path / "mind.json"
        local_file.write_text("{}")
        import os
        ts = local_mtime.timestamp()
        os.utime(local_file, (ts, ts))

        blob = _make_blob(mtime=gcs_mtime)
        bucket = MagicMock(**{"blob.return_value": blob})

        result = sync_file(bucket, "shion/mind.json", tmp_path, required=True)

        blob.download_to_filename.assert_not_called()
        assert "[SKIP]" in result

    def test_skips_when_same_mtime(self, tmp_path):
        """ローカルと GCS の mtime が同じ場合もスキップ。"""
        now = datetime.now(timezone.utc).replace(microsecond=0)

        local_file = tmp_path / "mind.json"
        local_file.write_text("{}")
        import os
        ts = now.timestamp()
        os.utime(local_file, (ts, ts))

        blob = _make_blob(mtime=now)
        bucket = MagicMock(**{"blob.return_value": blob})

        result = sync_file(bucket, "shion/mind.json", tmp_path, required=True)

        blob.download_to_filename.assert_not_called()
        assert "[SKIP]" in result


# ------------------------------------------------------------------
# test_skip_when_gcs_file_missing
# ------------------------------------------------------------------

class TestSkipWhenGcsFileMissing:
    def test_skips_required_file_when_not_in_gcs(self, tmp_path):
        """必須ファイルが GCS に存在しない場合もスキップ（エラーにしない）。"""
        blob = _make_blob(reload_raises=True)
        bucket = MagicMock(**{"blob.return_value": blob})

        result = sync_file(bucket, "shion/mind.json", tmp_path, required=True)

        blob.download_to_filename.assert_not_called()
        assert "[SKIP]" in result

    def test_skips_optional_file_when_not_in_gcs(self, tmp_path):
        """任意ファイルが GCS に存在しない場合もスキップ（エラーにしない）。"""
        blob = _make_blob(reload_raises=True)
        bucket = MagicMock(**{"blob.return_value": blob})

        result = sync_file(bucket, "shion/keypoints.json", tmp_path, required=False)

        blob.download_to_filename.assert_not_called()
        assert "[SKIP]" in result


# ------------------------------------------------------------------
# test_creates_local_dir
# ------------------------------------------------------------------

class TestCreatesLocalDir:
    def test_creates_dir_if_not_exists(self, tmp_path):
        """ローカルディレクトリが存在しない場合は自動作成してダウンロード。"""
        target_dir = tmp_path / "nonexistent" / "shion"
        assert not target_dir.exists()

        blob = _make_blob(mtime=datetime.now(timezone.utc))
        bucket = MagicMock(**{"blob.return_value": blob})

        result = sync_file(bucket, "shion/mind.json", target_dir, required=True)

        assert target_dir.exists()
        blob.download_to_filename.assert_called_once()
        assert "[DL]" in result
