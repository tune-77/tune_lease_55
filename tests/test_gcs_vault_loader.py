"""scripts/gcs_vault_loader.py の単体テスト（モックベース）"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------
# google.cloud.storage モック（未インストール環境対応）
# ------------------------------------------------------------------

def _setup_gcs_mock() -> MagicMock:
    storage_mock = MagicMock()
    cloud_mod = sys.modules.get("google.cloud")
    if cloud_mod is None:
        cloud_mod = ModuleType("google.cloud")
        sys.modules["google.cloud"] = cloud_mod
    sys.modules["google.cloud.storage"] = storage_mock
    setattr(cloud_mod, "storage", storage_mock)
    return storage_mock


_gcs_mock = _setup_gcs_mock()

from scripts.gcs_vault_loader import (  # noqa: E402
    GCS_BUCKET,
    GCS_VAULT_PREFIX,
    _download_worker_count,
    download_vault,
    load_vault_texts,
)


def _set_client_mock(client_mock: MagicMock) -> None:
    # download_vault は呼び出しごとに ``from google.cloud import storage``
    # するため、他テストが差し替えた import 状態も毎回復元する。
    cloud_mod = sys.modules.get("google.cloud")
    if cloud_mod is None:
        cloud_mod = ModuleType("google.cloud")
        sys.modules["google.cloud"] = cloud_mod
    sys.modules["google.cloud.storage"] = _gcs_mock
    setattr(cloud_mod, "storage", _gcs_mock)
    _gcs_mock.Client.return_value = client_mock


def _make_blob(name: str, content: bytes = b"# test") -> MagicMock:
    blob = MagicMock()
    blob.name = name
    def _download(filename: str, **_kwargs: object) -> None:
        Path(filename).write_bytes(content)
    blob.download_to_filename.side_effect = _download
    return blob


class TestDownloadVault:
    def test_returns_dest_dir(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = []
        _set_client_mock(client_mock)

        result = download_vault(dest_dir=tmp_path)
        assert result == tmp_path

    def test_creates_dest_dir(self, tmp_path: Path) -> None:
        dest = tmp_path / "vault_out"
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = []
        _set_client_mock(client_mock)

        download_vault(dest_dir=dest)
        assert dest.is_dir()

    def test_downloads_md_files(self, tmp_path: Path) -> None:
        blobs = [
            _make_blob("vault/notes/note1.md", b"# Note 1"),
            _make_blob("vault/notes/note2.md", b"# Note 2"),
        ]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, prefix="vault/")

        assert (tmp_path / "notes" / "note1.md").read_bytes() == b"# Note 1"
        assert (tmp_path / "notes" / "note2.md").read_bytes() == b"# Note 2"

    def test_downloads_multiple_files_in_parallel(self, tmp_path: Path) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def _slow_blob(name: str) -> MagicMock:
            blob = MagicMock()
            blob.name = name

            def _download(filename: str, **_kwargs: object) -> None:
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.03)
                Path(filename).write_text(name, encoding="utf-8")
                with lock:
                    active -= 1

            blob.download_to_filename.side_effect = _download
            return blob

        blobs = [_slow_blob(f"vault/note-{index}.md") for index in range(4)]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, prefix="vault/", max_workers=4)

        assert max_active > 1
        assert len(list(tmp_path.glob("*.md"))) == 4

    def test_worker_count_is_bounded(self, monkeypatch) -> None:
        monkeypatch.setenv("GCS_VAULT_DOWNLOAD_WORKERS", "999")
        assert _download_worker_count() == 32
        assert _download_worker_count(0) == 1

    def test_invalid_worker_env_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("GCS_VAULT_DOWNLOAD_WORKERS", "invalid")
        assert _download_worker_count() == 8

    def test_skips_non_md_files(self, tmp_path: Path) -> None:
        blobs = [
            _make_blob("vault/image.png"),
            _make_blob("vault/note.md", b"# keep"),
        ]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, prefix="vault/")

        assert not (tmp_path / "image.png").exists()
        assert (tmp_path / "note.md").exists()

    def test_skips_prefix_only_blob(self, tmp_path: Path) -> None:
        # blob.name == prefix（ディレクトリ自体 rel="" になる）はスキップする
        dir_blob = _make_blob("vault/")
        file_blob = _make_blob("vault/note.md", b"ok")
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = [dir_blob, file_blob]
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, prefix="vault/")

        assert (tmp_path / "note.md").exists()
        dir_blob.download_to_filename.assert_not_called()

    def test_uses_default_bucket_and_prefix(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = []
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path)

        client_mock.list_blobs.assert_called_once_with(GCS_BUCKET, prefix=GCS_VAULT_PREFIX, timeout=30)

    def test_uses_custom_bucket_and_prefix(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = []
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, bucket="my-bucket", prefix="custom/")

        client_mock.list_blobs.assert_called_once_with("my-bucket", prefix="custom/", timeout=30)

    def test_normalizes_gs_url_bucket(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = []
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, bucket="gs://my-bucket/some-prefix", prefix="custom/")

        client_mock.list_blobs.assert_called_once_with("my-bucket", prefix="custom/", timeout=30)

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        blobs = [
            _make_blob("vault/deep/nested/dir/note.md", b"deep"),
        ]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, prefix="vault/")

        assert (tmp_path / "deep" / "nested" / "dir" / "note.md").read_bytes() == b"deep"

    def test_prunes_local_md_missing_from_gcs(self, tmp_path: Path) -> None:
        stale = tmp_path / "old.md"
        stale.write_text("# stale")
        keep = tmp_path / "keep.md"
        keep.write_text("# old keep")

        blobs = [_make_blob("vault/keep.md", b"# fresh keep")]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, prefix="vault/")

        assert not stale.exists()
        assert keep.read_text() == "# fresh keep"

    def test_skips_unsafe_relative_paths(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "evil.md"
        if outside.exists():
            outside.unlink()

        blobs = [
            _make_blob("vault/../evil.md", b"bad"),
            _make_blob("vault/safe.md", b"safe"),
        ]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _set_client_mock(client_mock)

        download_vault(dest_dir=tmp_path, prefix="vault/")

        assert not outside.exists()
        assert (tmp_path / "safe.md").read_text() == "safe"


class TestLoadVaultTexts:
    def test_returns_list_of_texts(self, tmp_path: Path) -> None:
        blobs = [
            _make_blob("vault/a.md", b"# A"),
            _make_blob("vault/b.md", b"# B"),
        ]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _set_client_mock(client_mock)

        texts = load_vault_texts(dest_dir=tmp_path, prefix="vault/")

        assert len(texts) == 2
        assert "# A" in texts
        assert "# B" in texts

    def test_returns_empty_list_when_no_md(self, tmp_path: Path) -> None:
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = [_make_blob("vault/image.png")]
        _set_client_mock(client_mock)

        texts = load_vault_texts(dest_dir=tmp_path, prefix="vault/")

        assert texts == []

    def test_returns_texts_sorted_by_path(self, tmp_path: Path) -> None:
        blobs = [
            _make_blob("vault/z.md", b"Z"),
            _make_blob("vault/a.md", b"A"),
        ]
        client_mock = MagicMock()
        client_mock.list_blobs.return_value = blobs
        _gcs_mock.Client.return_value = client_mock

        texts = load_vault_texts(dest_dir=tmp_path, prefix="vault/")

        assert texts[0] == "A"
        assert texts[1] == "Z"
