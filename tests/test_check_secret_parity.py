"""scripts/check_secret_parity.py の突き合わせロジックを、実gcloudなしで検証する。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import scripts.check_secret_parity as checker


@pytest.fixture
def fake_gcloud(tmp_path, monkeypatch):
    """SECRET_MANAGER_STATE (ok/missing/denied) に応じて偽のgcloud応答を返す。"""

    gcloud = tmp_path / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "state = os.environ.get('SECRET_MANAGER_STATE', 'ok')\n"
        "value = os.environ.get('SECRET_MANAGER_VALUE', '')\n"
        "if state == 'ok':\n"
        "    sys.stdout.write(value)\n"
        "    sys.exit(0)\n"
        "if state == 'missing':\n"
        "    sys.stderr.write('ERROR: (gcloud.secrets.versions.access) NOT_FOUND: Secret not found.\\n')\n"
        "    sys.exit(1)\n"
        "sys.stderr.write(\"ERROR: (gcloud.secrets.versions.access) PERMISSION_DENIED: Permission 'secretmanager.versions.access' denied.\\n\")\n"
        "sys.exit(1)\n"
    )
    gcloud.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return gcloud


def test_values_match(fake_gcloud, monkeypatch):
    monkeypatch.setenv("SECRET_MANAGER_STATE", "ok")
    monkeypatch.setenv("SECRET_MANAGER_VALUE", "shared-secret")
    monkeypatch.setenv("MY_TOKEN", "shared-secret")

    result = checker.check_secret_parity("MY_TOKEN", "test-project")

    assert result.ok is True
    assert "match" in result.message


def test_values_differ_fails(fake_gcloud, monkeypatch):
    monkeypatch.setenv("SECRET_MANAGER_STATE", "ok")
    monkeypatch.setenv("SECRET_MANAGER_VALUE", "new-secret")
    monkeypatch.setenv("MY_TOKEN", "old-secret")

    result = checker.check_secret_parity("MY_TOKEN", "test-project")

    assert result.ok is False
    assert "differs" in result.message


def test_github_secret_missing_fails(fake_gcloud, monkeypatch):
    monkeypatch.setenv("SECRET_MANAGER_STATE", "ok")
    monkeypatch.setenv("SECRET_MANAGER_VALUE", "shared-secret")
    monkeypatch.delenv("MY_TOKEN", raising=False)

    result = checker.check_secret_parity("MY_TOKEN", "test-project")

    assert result.ok is False
    assert "GitHub Actions secret" in result.message


def test_secret_manager_secret_missing_but_github_set_fails(fake_gcloud, monkeypatch):
    monkeypatch.setenv("SECRET_MANAGER_STATE", "missing")
    monkeypatch.setenv("MY_TOKEN", "shared-secret")

    result = checker.check_secret_parity("MY_TOKEN", "test-project")

    assert result.ok is False
    assert "no matching Secret Manager secret" in result.message


def test_neither_side_configured_skips(fake_gcloud, monkeypatch):
    monkeypatch.setenv("SECRET_MANAGER_STATE", "missing")
    monkeypatch.delenv("MY_TOKEN", raising=False)

    result = checker.check_secret_parity("MY_TOKEN", "test-project")

    assert result.ok is True
    assert "skip" in result.message


def test_permission_denied_warns_but_does_not_fail(fake_gcloud, monkeypatch):
    monkeypatch.setenv("SECRET_MANAGER_STATE", "denied")
    monkeypatch.setenv("MY_TOKEN", "shared-secret")

    result = checker.check_secret_parity("MY_TOKEN", "test-project")

    assert result.ok is True
    assert "PERMISSION_DENIED" in result.message


def test_workflow_checks_both_probe_tokens():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/secret-parity-check.yml").read_text(encoding="utf-8")

    assert "--secret DASHBOARD_HEALTH_PROBE_TOKEN" in workflow
    assert "--secret KNOWLEDGE_SYNC_PROBE_TOKEN" in workflow
    assert "DASHBOARD_HEALTH_PROBE_TOKEN: ${{ secrets.DASHBOARD_HEALTH_PROBE_TOKEN }}" in workflow
    assert "KNOWLEDGE_SYNC_PROBE_TOKEN: ${{ secrets.KNOWLEDGE_SYNC_PROBE_TOKEN }}" in workflow
