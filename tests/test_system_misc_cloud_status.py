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


def test_knowledge_sync_ready_when_chroma_covers_vault():
    status = system_misc._knowledge_sync_status(
        {"markdown_count": 98},
        {"indexing_enabled": True, "document_count": 581},
    )

    assert status == {
        "ready": True,
        "state": "ready",
        "reason": "ok",
        "vault_markdown_count": 98,
        "chroma_document_count": 581,
        "coverage_ratio": 1.0,
    }


def test_knowledge_sync_reports_empty_chroma():
    status = system_misc._knowledge_sync_status(
        {"markdown_count": 98},
        {"indexing_enabled": True, "document_count": 0},
    )

    assert status["ready"] is False
    assert status["state"] == "empty"
    assert status["coverage_ratio"] == 0.0
