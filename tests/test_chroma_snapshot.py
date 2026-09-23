import os

from api.knowledge import chroma_snapshot


def test_is_snapshot_enabled_matches_db_snapshot_semantics(monkeypatch):
    monkeypatch.delenv("CLOUDRUN_DATA_MODE", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    assert chroma_snapshot.is_snapshot_enabled() is False

    monkeypatch.setenv("K_SERVICE", "tune-lease-55-api")
    assert chroma_snapshot.is_snapshot_enabled() is True

    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "demo")
    assert chroma_snapshot.is_snapshot_enabled() is False


def test_restore_skips_when_dir_already_populated(tmp_path, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "tune-lease-55-api")
    populated = tmp_path / "chroma_db"
    populated.mkdir()
    (populated / "existing.bin").write_bytes(b"x")

    result = chroma_snapshot.restore(chroma_dir=str(populated))

    assert result["restored"] is False
    assert result["reason"] == "chroma_dir_already_populated"


def test_snapshot_and_upload_skips_when_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "tune-lease-55-api")
    empty = tmp_path / "chroma_db"
    empty.mkdir()

    result = chroma_snapshot.snapshot_and_upload(chroma_dir=str(empty))

    assert result["uploaded"] is False
    assert result["reason"] == "chroma_dir_missing_or_empty"


if __name__ == "__main__":
    os.environ["K_SERVICE"] = "tune-lease-55-api"
    r = chroma_snapshot.is_snapshot_enabled()
    assert r is True
    print("[chroma_snapshot self-check] ok")
