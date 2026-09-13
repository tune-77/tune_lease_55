from __future__ import annotations

import socket

import pytest
from fastapi import HTTPException

from api import main
from api.routers import lease_news


def test_validate_public_http_url_allows_public_http(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )

    main._validate_public_http_url("https://example.com/news")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "https://metadata.google.internal/computeMetadata/v1/",
        "https://user:pass@example.com/news",
    ],
)
def test_validate_public_http_url_rejects_unsafe_forms(url):
    with pytest.raises(HTTPException):
        main._validate_public_http_url(url)


def test_validate_public_http_url_rejects_private_resolved_ip(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))],
    )

    with pytest.raises(HTTPException):
        main._validate_public_http_url("https://example.com/news")


def test_fetch_url_validates_redirect_before_following(monkeypatch):
    class RedirectResponse:
        is_redirect = True
        is_permanent_redirect = False
        headers = {"location": "http://internal.example/secret"}

    requested_urls = []

    def fake_get(url, **_kwargs):
        requested_urls.append(url)
        return RedirectResponse()

    def fake_getaddrinfo(host, *_args, **_kwargs):
        ip = "10.0.0.5" if host == "internal.example" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(lease_news.requests, "get", fake_get)
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(HTTPException):
        lease_news._fetch_url_text("https://example.com/news")

    assert requested_urls == ["https://example.com/news"]


def test_read_obsidian_files_stays_inside_vault(tmp_path):
    vault = tmp_path / "vault"
    sibling = tmp_path / "vault_evil"
    vault.mkdir()
    sibling.mkdir()
    (vault / "allowed.md").write_text("inside", encoding="utf-8")
    (sibling / "secret.md").write_text("outside", encoding="utf-8")

    content, files_read = main._read_obsidian_files(
        str(vault),
        ["allowed.md", "../vault_evil/secret.md"],
    )

    assert files_read == ["allowed.md"]
    assert "inside" in content
    assert "outside" not in content
