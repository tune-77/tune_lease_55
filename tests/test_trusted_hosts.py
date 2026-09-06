from api.security_headers import get_trusted_hosts


def test_trusted_hosts_default_to_local_only(monkeypatch):
    for name in ("K_SERVICE", "PUBLIC_TUNNEL", "TRUSTED_HOSTS", "CORS_ALLOWED_ORIGINS"):
        monkeypatch.delenv(name, raising=False)

    hosts = get_trusted_hosts()

    assert "localhost" in hosts
    assert "127.0.0.1" in hosts
    assert "*" not in hosts


def test_trusted_hosts_include_managed_and_configured_hosts(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "lease-api")
    monkeypatch.setenv("PUBLIC_TUNNEL", "1")
    monkeypatch.setenv("TRUSTED_HOSTS", "api.example.com")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://app.example.com")

    hosts = get_trusted_hosts()

    assert "*.run.app" in hosts
    assert "*.trycloudflare.com" in hosts
    assert "api.example.com" in hosts
    assert "app.example.com" in hosts
