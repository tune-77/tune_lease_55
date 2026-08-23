from __future__ import annotations

import importlib.util
import sys
import types
import inspect
import sqlite3

from api.cloudrun_data_safety_audit import audit_cloudrun_data_safety
from api.shion_agent_tools import READ_ONLY_DB_TOOLS
from scripts.promote_cloudrun_return_data import promote_approved_return_data
import data_cases


def test_cloudrun_data_safety_audit_is_green():
    result = audit_cloudrun_data_safety()

    assert result["mode"] == "cloudrun_data_safety_audit"
    assert result["status"] == "ok"
    assert result["issue_count"] == 0
    assert "no_db_write" in result["guardrail"]


def test_cloudrun_data_safety_adk_tool_is_registered():
    names = {tool.__name__ for tool in READ_ONLY_DB_TOOLS}

    assert "audit_cloudrun_data_safety" in names


def test_restore_lease_snapshot_skips_existing_db(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "lease_data.db"
    target.write_bytes(b"local-db")

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("LEASE_DB_FILENAME", "lease_data.db")
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "production")

    spec = importlib.util.spec_from_file_location(
        "restore_lease_db_snapshot",
        "scripts/restore_lease_db_snapshot.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()

    assert target.read_bytes() == b"local-db"


def test_score_input_skips_when_return_registered_case_id_exists(tmp_path, monkeypatch):
    return_db = tmp_path / "return.db"
    target_db = tmp_path / "lease_data.db"
    sqlite3.connect(target_db).close()
    monkeypatch.setattr(data_cases, "DB_PATH", str(target_db))

    with sqlite3.connect(return_db) as conn:
        conn.execute(
            """
            CREATE TABLE cloudrun_score_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                event_type TEXT,
                case_id TEXT,
                surface TEXT,
                score REAL,
                hantei TEXT,
                industry_major TEXT,
                industry_sub TEXT,
                inputs_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                return_review_status TEXT,
                return_review_note TEXT,
                return_registered_case_id TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            INSERT INTO cloudrun_score_inputs (
                event_id, event_type, case_id, surface, score, hantei, industry_major,
                industry_sub, inputs_json, result_json, created_at, return_review_status,
                return_review_note, return_registered_case_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "evt-score-registered",
                "score_full_calculated",
                "case-registered",
                "screening",
                72.5,
                "承認",
                "D 建設業",
                "06 総合工事業",
                '{"company_name":"[REDACTED]","nenshu":200}',
                '{"score":72.5}',
                "2026-07-01T00:01:00Z",
                "approved",
                "already registered",
                "case-existing-1",
            ),
        )

    result = promote_approved_return_data(
        return_db=return_db,
        target_db=target_db,
        backup_dir=tmp_path / "backups",
        apply=True,
        backup=False,
    )

    assert result["summary"] == {"score_input:skipped_already_registered": 1}
    with sqlite3.connect(target_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM cloudrun_return_promotions").fetchone()[0] == 1
        assert conn.execute(
            "SELECT target_id FROM cloudrun_return_promotions"
        ).fetchone()[0] == "case-existing-1"


class _FakeBlob:
    def __init__(self, store: dict[str, bytes], name: str) -> None:
        self._store = store
        self.name = name

    def upload_from_filename(self, filename: str) -> None:
        self._store[self.name] = open(filename, "rb").read()

    def upload_from_string(self, data: str, **_kwargs) -> None:
        self._store[self.name] = data.encode("utf-8")

    def download_to_filename(self, filename: str) -> None:
        with open(filename, "wb") as fh:
            fh.write(self._store[self.name])

    def download_as_text(self) -> str:
        if self.name not in self._store:
            raise FileNotFoundError(self.name)
        return self._store[self.name].decode("utf-8")

    def delete(self) -> None:
        self._store.pop(self.name, None)


class _FakeBucket:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._store, name)

    def list_blobs(self, prefix: str):
        return [_FakeBlob(self._store, name) for name in self._store if name.startswith(prefix)]


class _FakeStorageClient:
    store: dict[str, bytes] = {}

    def bucket(self, _bucket_name: str) -> _FakeBucket:
        return _FakeBucket(self.store)


def _install_fake_google_storage(monkeypatch):
    google_mod = types.ModuleType("google")
    cloud_mod = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    api_core_mod = types.ModuleType("google.api_core")
    exceptions_mod = types.ModuleType("google.api_core.exceptions")

    class PreconditionFailed(Exception):
        pass

    storage_mod.Client = _FakeStorageClient
    storage_mod.Blob = _FakeBlob
    exceptions_mod.PreconditionFailed = PreconditionFailed

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_mod)
    monkeypatch.setitem(sys.modules, "google.api_core", api_core_mod)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", exceptions_mod)
    monkeypatch.delitem(sys.modules, "scripts.gcs_lock", raising=False)
    _FakeStorageClient.store = {}


def test_snapshot_upload_restore_roundtrip_with_fake_gcs(tmp_path, monkeypatch):
    _install_fake_google_storage(monkeypatch)
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "production")
    monkeypatch.setenv("GCS_BUCKET", "fake-bucket")
    monkeypatch.setenv("GCS_SNAPSHOT_PREFIX", "cloudrun-snapshots")

    db_path = tmp_path / "lease_data.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE cases (id TEXT PRIMARY KEY, company_name TEXT NOT NULL)")
        conn.execute("INSERT INTO cases VALUES (?, ?)", ("case-1", "復元テスト会社"))

    from api.cloudrun_db_snapshot import snapshot_and_upload

    uploaded = snapshot_and_upload(str(db_path))
    assert uploaded["uploaded"] is True
    assert "cloudrun-snapshots/lease_data.db" in _FakeStorageClient.store
    assert any(name.startswith("cloudrun-snapshots/history/lease_data.db.") for name in _FakeStorageClient.store)

    db_path.unlink()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEASE_DB_FILENAME", "lease_data.db")
    spec = importlib.util.spec_from_file_location(
        "restore_lease_db_snapshot_roundtrip",
        "scripts/restore_lease_db_snapshot.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT company_name FROM cases WHERE id = ?", ("case-1",)).fetchone()
    assert row == ("復元テスト会社",)


def test_read_only_adk_tools_do_not_contain_dangerous_side_effect_calls():
    banned_snippets = [
        ".write_text(",
        ".write_bytes(",
        ".execute(\"INSERT",
        ".execute(\"UPDATE",
        ".execute(\"DELETE",
        ".execute(\"DROP",
        ".execute(\"CREATE",
        "subprocess.",
        "os.system(",
        "upload_from_filename(",
        "upload_from_string(",
        "download_to_filename(",
        "git push",
        "gcloud ",
        "requests.post(",
        "requests.put(",
        "requests.delete(",
    ]
    allowlisted_tools = {"score_full_case"}

    offenders: list[str] = []
    for tool in READ_ONLY_DB_TOOLS:
        if tool.__name__ in allowlisted_tools:
            continue
        source = inspect.getsource(tool)
        for snippet in banned_snippets:
            if snippet in source:
                offenders.append(f"{tool.__name__}: {snippet}")

    assert offenders == []
