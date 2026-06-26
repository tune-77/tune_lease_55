"""iCloud → GCS 差分アップロードスクリプトの単体テスト（モックベース）"""
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch


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

from scripts.icloud_to_gcs_sync import LOCAL_VAULT_DIR, collect_md_files, main, upload_file  # noqa: E402


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------

def _make_blob(local_mtime=None, reload_raises: bool = False) -> MagicMock:
    blob = MagicMock()
    if reload_raises:
        blob.reload.side_effect = Exception("Not Found")
    else:
        blob.reload.return_value = None
        blob.metadata = {"local_mtime": local_mtime} if local_mtime else {}
    return blob


# ------------------------------------------------------------------
# test_upload_new_file
# ------------------------------------------------------------------

class TestUploadNewFile:
    def test_upload_new_file(self, tmp_path):
        """GCS に存在しない新規ファイルがアップロードされること。"""
        md = tmp_path / "note.md"
        md.write_text("# hello")

        blob = _make_blob(reload_raises=True)
        bucket = MagicMock(**{"blob.return_value": blob, "name": "test-bucket"})

        result = upload_file(bucket, md, tmp_path, "vault/")

        blob.upload_from_filename.assert_called_once_with(str(md))
        assert "[UP]" in result
        assert "note.md" in result


# ------------------------------------------------------------------
# test_upload_when_local_newer
# ------------------------------------------------------------------

class TestUploadWhenLocalNewer:
    def test_upload_when_local_newer(self, tmp_path):
        """GCS の local_mtime と異なる（ローカルが新しい）場合にアップロードされること。"""
        md = tmp_path / "newer.md"
        md.write_text("# updated")

        local_mtime = str(md.stat().st_mtime)
        old_mtime = str(float(local_mtime) - 100.0)

        blob = _make_blob(local_mtime=old_mtime)
        bucket = MagicMock(**{"blob.return_value": blob, "name": "test-bucket"})

        result = upload_file(bucket, md, tmp_path, "vault/")

        blob.upload_from_filename.assert_called_once_with(str(md))
        assert "[UP]" in result


# ------------------------------------------------------------------
# test_skip_when_not_modified
# ------------------------------------------------------------------

class TestSkipWhenNotModified:
    def test_skip_when_not_modified(self, tmp_path):
        """GCS の local_mtime がローカルと同一の場合にスキップされること。"""
        md = tmp_path / "same.md"
        md.write_text("# no change")

        local_mtime = str(md.stat().st_mtime)
        blob = _make_blob(local_mtime=local_mtime)
        bucket = MagicMock(**{"blob.return_value": blob, "name": "test-bucket"})

        result = upload_file(bucket, md, tmp_path, "vault/")

        blob.upload_from_filename.assert_not_called()
        assert "[SKIP]" in result


# ------------------------------------------------------------------
# test_excludes_obsidian_dir
# ------------------------------------------------------------------

class TestExcludesObsidianDir:
    def test_excludes_obsidian_dir(self, tmp_path):
        """.obsidian/ 配下のファイルが除外されること。"""
        obsidian_dir = tmp_path / ".obsidian"
        obsidian_dir.mkdir()
        (obsidian_dir / "config").write_text("{}")
        (obsidian_dir / "workspace.md").write_text("# workspace")

        normal_md = tmp_path / "note.md"
        normal_md.write_text("# keep")

        files = collect_md_files(tmp_path)

        assert normal_md in files
        assert not any(".obsidian" in str(f) for f in files)

    def test_excludes_nested_obsidian(self, tmp_path):
        """.obsidian/ の深い階層のファイルも除外されること。"""
        nested = tmp_path / ".obsidian" / "plugins" / "some-plugin"
        nested.mkdir(parents=True)
        (nested / "README.md").write_text("# plugin")

        files = collect_md_files(tmp_path)
        assert len(files) == 0


# ------------------------------------------------------------------
# test_handles_empty_vault
# ------------------------------------------------------------------

class TestHandlesEmptyVault:
    def test_handles_empty_vault(self, tmp_path):
        """Vault が空でも collect_md_files がエラーにならないこと。"""
        files = collect_md_files(tmp_path)
        assert files == []

    def test_main_handles_empty_vault(self, tmp_path):
        """Vault が空でも main() がエラーにならないこと（正常終了）。"""
        with patch("scripts.icloud_to_gcs_sync.LOCAL_VAULT_DIR", str(tmp_path)):
            with patch("scripts.icloud_to_gcs_sync.storage.Client"):
                main()


def test_default_local_vault_dir_points_to_icloud_vault():
    assert "Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault" in LOCAL_VAULT_DIR
