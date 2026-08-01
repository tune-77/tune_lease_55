"""Obsidian Vault パス解決の契約テスト。

Vault パスは「書き込み先」と「RAG 索引先」の両方を決めるため、
解決順序が崩れると審査ナレッジが静かに古い Vault を向く。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import runtime_paths  # noqa: E402


def test_env_precedence_is_vault_path_first():
    resolution = runtime_paths.describe_obsidian_vault_resolution(
        {"OBSIDIAN_VAULT_PATH": "/a/vault", "OBSIDIAN_VAULT": "/a/vault"}
    )
    assert resolution.path == Path("/a/vault")
    assert resolution.source == "env:OBSIDIAN_VAULT_PATH"


def test_falls_back_to_obsidian_vault_when_path_var_absent():
    resolution = runtime_paths.describe_obsidian_vault_resolution(
        {"OBSIDIAN_VAULT": "/b/vault"}
    )
    assert resolution.path == Path("/b/vault")
    assert resolution.source == "env:OBSIDIAN_VAULT"


def test_conflicting_env_vars_warn_and_prefer_canonical():
    resolution = runtime_paths.describe_obsidian_vault_resolution(
        {"OBSIDIAN_VAULT_PATH": "/a/vault", "OBSIDIAN_VAULT": "/b/vault"}
    )
    assert resolution.path == Path("/a/vault")
    assert any("食い違" in w for w in resolution.warnings)


def test_blank_env_value_is_ignored():
    resolution = runtime_paths.describe_obsidian_vault_resolution(
        {"OBSIDIAN_VAULT_PATH": "   ", "OBSIDIAN_VAULT": "/b/vault"}
    )
    assert resolution.path == Path("/b/vault")


def test_tilde_in_env_is_expanded():
    resolution = runtime_paths.describe_obsidian_vault_resolution(
        {"OBSIDIAN_VAULT_PATH": "~/vault"}
    )
    assert resolution.path == Path.home() / "vault"


def test_missing_vault_is_reported(tmp_path):
    resolution = runtime_paths.describe_obsidian_vault_resolution(
        {"OBSIDIAN_VAULT_PATH": str(tmp_path / "nope")}
    )
    assert resolution.exists is False
    assert any("見つかりません" in w for w in resolution.warnings)


def test_both_vault_dirs_present_warns_about_split(monkeypatch, tmp_path):
    new_vault = tmp_path / "obsidian-vault"
    legacy_vault = tmp_path / "Obsidian Vault"
    new_vault.mkdir()
    legacy_vault.mkdir()
    monkeypatch.setattr(runtime_paths, "DEFAULT_OBSIDIAN_VAULT", new_vault)
    monkeypatch.setattr(runtime_paths, "LEGACY_OBSIDIAN_VAULT", legacy_vault)

    resolution = runtime_paths.describe_obsidian_vault_resolution({})

    assert resolution.path == new_vault
    assert resolution.source == "default"
    assert any("両方存在" in w for w in resolution.warnings)


def test_legacy_vault_used_when_only_legacy_exists(monkeypatch, tmp_path):
    new_vault = tmp_path / "obsidian-vault"
    legacy_vault = tmp_path / "Obsidian Vault"
    legacy_vault.mkdir()
    monkeypatch.setattr(runtime_paths, "DEFAULT_OBSIDIAN_VAULT", new_vault)
    monkeypatch.setattr(runtime_paths, "LEGACY_OBSIDIAN_VAULT", legacy_vault)

    resolution = runtime_paths.describe_obsidian_vault_resolution({})

    assert resolution.path == legacy_vault
    assert resolution.source == "legacy"
    assert resolution.warnings == []


def test_get_obsidian_vault_path_returns_str(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "/c/vault")
    value = runtime_paths.get_obsidian_vault_path()
    assert isinstance(value, str)
    assert value == "/c/vault"


@pytest.mark.parametrize("name", runtime_paths.OBSIDIAN_VAULT_ENV_VARS)
def test_every_canonical_env_var_is_honoured(monkeypatch, name):
    for candidate in runtime_paths.OBSIDIAN_VAULT_ENV_VARS:
        monkeypatch.delenv(candidate, raising=False)
    monkeypatch.setenv(name, "/d/vault")
    assert runtime_paths.resolve_obsidian_vault() == Path("/d/vault")


def test_lease_wiki_vault_is_nested_in_the_obsidian_vault(monkeypatch):
    """lease-wiki-vault は Vault の入れ子。Documents 直下（兄弟）ではない。

    以前は参照側ごとに入れ子・兄弟が割れており、書き出しと読み出しが
    別ツリーを向いていた。ここを固定して再発を防ぐ。
    """
    for candidate in runtime_paths.OBSIDIAN_VAULT_ENV_VARS:
        monkeypatch.delenv(candidate, raising=False)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "/e/Obsidian Vault")

    assert runtime_paths.resolve_lease_wiki_vault() == Path("/e/Obsidian Vault/lease-wiki-vault")


def test_lease_wiki_vault_follows_the_resolved_vault(monkeypatch):
    """Vault の解決先が変われば lease-wiki-vault も追随する。"""
    for candidate in runtime_paths.OBSIDIAN_VAULT_ENV_VARS:
        monkeypatch.delenv(candidate, raising=False)
    monkeypatch.setenv("OBSIDIAN_VAULT", "/f/other-vault")

    resolved = runtime_paths.resolve_lease_wiki_vault()

    assert resolved.parent == runtime_paths.resolve_obsidian_vault()
    assert resolved.name == runtime_paths.LEASE_WIKI_VAULT_DIRNAME
