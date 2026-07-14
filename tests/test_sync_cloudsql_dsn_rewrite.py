"""sync_cloudsql_to_obsidian.py のソケットDSN書き換え（REV-026a）のテスト。"""

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_cloudsql_to_obsidian.py"
_spec = importlib.util.spec_from_file_location("sync_cloudsql_to_obsidian", _SCRIPT)
sync_mod = importlib.util.module_from_spec(_spec)
sys.modules["sync_cloudsql_to_obsidian"] = sync_mod
_spec.loader.exec_module(sync_mod)

SOCKET = "/cloudsql/gen-lang-client-0420497423:asia-northeast1:tune-lease-db"


def test_url_form_socket_dsn_is_rewritten_to_tcp():
    dsn = f"postgresql://postgres:secret@/lease-db-demo?host={SOCKET}"
    result = sync_mod.rewrite_cloudsql_socket_dsn(dsn, "127.0.0.1", 15432)
    assert result == "postgresql://postgres:secret@127.0.0.1:15432/lease-db-demo"


def test_url_form_keeps_other_query_params():
    dsn = f"postgresql://postgres:secret@/lease-db-demo?host={SOCKET}&sslmode=disable"
    result = sync_mod.rewrite_cloudsql_socket_dsn(dsn, "127.0.0.1", 15432)
    assert "sslmode=disable" in result
    assert "/cloudsql/" not in result
    assert "127.0.0.1:15432" in result


def test_keyword_form_socket_dsn_is_rewritten():
    dsn = f"host={SOCKET} dbname=lease-db-demo user=postgres password=secret"
    result = sync_mod.rewrite_cloudsql_socket_dsn(dsn, "127.0.0.1", 15432)
    assert "host=127.0.0.1" in result
    assert "port=15432" in result
    assert "dbname=lease-db-demo" in result
    assert "/cloudsql/" not in result


def test_keyword_form_existing_port_is_replaced():
    dsn = f"host={SOCKET} port=5432 dbname=lease-db-demo"
    result = sync_mod.rewrite_cloudsql_socket_dsn(dsn, "127.0.0.1", 15432)
    assert "port=15432" in result
    assert "port=5432" not in result
    assert result.count("port=") == 1


def test_tcp_dsn_is_unchanged():
    dsn = "postgresql://postgres:secret@35.194.127.102:5432/lease-db-demo"
    assert sync_mod.rewrite_cloudsql_socket_dsn(dsn, "127.0.0.1", 15432) == dsn


def test_maybe_rewrite_respects_proxy_env(monkeypatch):
    monkeypatch.setenv("CLOUD_SQL_PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("CLOUD_SQL_PROXY_PORT", "25432")
    dsn = f"postgresql://postgres:secret@/lease-db-demo?host={SOCKET}"
    result = sync_mod._maybe_rewrite_socket_dsn(dsn)
    assert "127.0.0.1:25432" in result


def test_unreachable_socket_dsn_when_proxy_down(monkeypatch):
    """ソケットDSN + プロキシ未起動 → 到達不能（スキップ対象）と判定される。"""
    monkeypatch.setattr(sync_mod, "_local_proxy_reachable", lambda host, port, timeout=1.0: False)
    dsn = f"postgresql://postgres:secret@/lease-db-demo?host={SOCKET}"
    assert sync_mod._is_unreachable_socket_dsn(dsn) is True


def test_socket_dsn_reachable_when_proxy_up(monkeypatch):
    monkeypatch.setattr(sync_mod, "_local_proxy_reachable", lambda host, port, timeout=1.0: True)
    dsn = f"postgresql://postgres:secret@/lease-db-demo?host={SOCKET}"
    assert sync_mod._is_unreachable_socket_dsn(dsn) is False


def test_tcp_dsn_is_never_unreachable_socket():
    dsn = "postgresql://postgres:secret@35.194.127.102:5432/lease-db-demo"
    assert sync_mod._is_unreachable_socket_dsn(dsn) is False


def test_secret_unavailable_without_password_skips_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sync_mod, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(sync_mod, "_database_url", lambda: "")
    monkeypatch.setenv("DATABASE_URL_SECRET_NAME", "DATABASE_URL")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CLOUD_SQL_PASSWORD", raising=False)

    sync_mod.sync(dry_run=True)

    captured = capsys.readouterr()
    assert "Cloud SQL会話ログ同期をスキップ" in captured.out
