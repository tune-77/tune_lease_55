from __future__ import annotations

from api.cloudrun_db_snapshot import is_snapshot_enabled


def test_snapshot_disabled_for_plain_local_runtime(monkeypatch):
    monkeypatch.delenv("CLOUDRUN_DATA_MODE", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)

    assert is_snapshot_enabled() is False


def test_snapshot_disabled_for_demo_mode(monkeypatch):
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "demo")
    monkeypatch.setenv("K_SERVICE", "tune-lease-55-api")

    assert is_snapshot_enabled() is False


def test_snapshot_enabled_for_explicit_production_mode(monkeypatch):
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "production")
    monkeypatch.delenv("K_SERVICE", raising=False)

    assert is_snapshot_enabled() is True


def test_snapshot_enabled_for_cloud_run_when_mode_is_omitted(monkeypatch):
    monkeypatch.delenv("CLOUDRUN_DATA_MODE", raising=False)
    monkeypatch.setenv("K_SERVICE", "tune-lease-55-api")

    assert is_snapshot_enabled() is True
