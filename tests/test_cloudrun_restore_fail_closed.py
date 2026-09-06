"""A failed production restore must never fall through to bundle seeding."""
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import restore_lease_db_snapshot as restore


class NotFound(Exception):
    pass


@pytest.fixture
def gcs(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "production")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEASE_DB_FILENAME", "lease_data.db")
    bucket = Mock()
    storage = types.ModuleType("google.cloud.storage")
    storage.Client = Mock(return_value=Mock(bucket=Mock(return_value=bucket)))
    cloud = types.ModuleType("google.cloud")
    cloud.storage = storage
    google = types.ModuleType("google")
    google.cloud = cloud
    exceptions = types.ModuleType("google.api_core.exceptions")
    exceptions.NotFound = NotFound
    for name, module in {
        "google": google, "google.cloud": cloud, "google.cloud.storage": storage,
        "google.api_core": types.ModuleType("google.api_core"),
        "google.api_core.exceptions": exceptions,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return bucket, tmp_path / "lease_data.db"


@pytest.mark.parametrize("error", [TimeoutError("timeout"), PermissionError("denied")])
def test_restore_failure_aborts_and_cleans_partial_download(gcs, error):
    bucket, target = gcs
    def fail(filename):
        Path(filename).write_bytes(b"partial")
        raise error
    bucket.blob.return_value.download_to_filename.side_effect = fail
    with pytest.raises(RuntimeError, match="起動を中止"):
        restore.main()
    assert not target.exists()
    assert not target.with_suffix(".db.downloading").exists()


def test_missing_snapshot_allows_initial_seed_only_in_existing_bucket(gcs):
    bucket, target = gcs
    bucket.blob.return_value.download_to_filename.side_effect = NotFound()
    restore.main()
    bucket.reload.assert_called_once()
    assert not target.exists()


def test_missing_bucket_is_not_first_boot(gcs):
    bucket, _ = gcs
    bucket.blob.return_value.download_to_filename.side_effect = NotFound()
    bucket.reload.side_effect = NotFound("bucket missing")
    with pytest.raises(NotFound):
        restore.main()


def test_successful_restore_promotes_complete_download(gcs):
    bucket, target = gcs
    bucket.blob.return_value.download_to_filename.side_effect = lambda filename: Path(filename).write_bytes(b"snapshot")
    restore.main()
    assert target.read_bytes() == b"snapshot"


def test_demo_does_not_attempt_restore(gcs, monkeypatch):
    bucket, _ = gcs
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "demo")
    restore.main()
    bucket.blob.assert_not_called()


@pytest.mark.parametrize("name", ["start_api_cloud_run.sh", "start_cloud_run.sh"])
def test_startup_stops_before_seed_and_server_when_restore_fails(tmp_path, name):
    import os
    import subprocess
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / name
    script.write_text((root / "scripts" / name).read_text())
    (tmp_path / "restore_lease_db_snapshot.py").write_text("raise SystemExit(7)\n")
    bundle = tmp_path / "bundle" / "data"
    bundle.mkdir(parents=True)
    (bundle / "lease_data.db").write_bytes(b"stale bundle")
    server = tmp_path / "server.js"
    server.write_text("")
    data = tmp_path / "data"
    result = subprocess.run(["bash", str(script)], env={**os.environ,
        "DATA_DIR": str(data), "CLOUDRUN_BUNDLE_DIR": str(bundle.parent),
        "NEXT_SERVER": str(server), "CLOUDRUN_DATA_MODE": "production",
    }, capture_output=True, timeout=10)
    assert result.returncode == 7
    assert not (data / "lease_data.db").exists()
