from pathlib import Path

from api.routers import system_misc


def test_cloud_db_status_uses_runtime_db_path(monkeypatch, tmp_path):
    db_path = tmp_path / "lease_data.db"
    db_path.write_text("", encoding="utf-8")

    class DummyCursor:
        def execute(self, _sql):
            return None

        def fetchone(self):
            return (1,)

    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return DummyCursor()

    monkeypatch.setattr(system_misc, "_LEASE_DB_PATH", Path(db_path))
    monkeypatch.setattr(system_misc, "current_backend", lambda: "sqlite")
    monkeypatch.setattr(system_misc, "_db_available", lambda: True)
    monkeypatch.setattr(system_misc, "get_connection", lambda: DummyConnection())

    status = system_misc._cloud_db_status()

    assert status["backend"] == "sqlite"
    assert status["local_db_exists"] is True
    assert status["available"] is True
