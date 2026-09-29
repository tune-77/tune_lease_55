"""REV-425: 本番Cloud RunのJev設定が両デプロイ経路に載り、キー欠落でデプロイを止めないこと。"""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "lib" / "jev_cloud_run_config.sh"
DEPLOY_YML = ROOT / ".github" / "workflows" / "deploy.yml"
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_cloud_run_api.sh"


def _jev_env_vars() -> str:
    match = re.search(r'^JEV_CLOUD_RUN_ENV_VARS="([^"]+)"', LIB.read_text(encoding="utf-8"), re.M)
    assert match
    return match.group(1)


def _api_deploy_step() -> str:
    text = DEPLOY_YML.read_text(encoding="utf-8")
    start = text.index("- name: Deploy to Cloud Run")
    end = text.index("- name: Show deployed URL", start)
    return text[start:end]


def test_production_modes_are_shadow_only():
    env_vars = _jev_env_vars()
    assert "TYPESAFE_ROUTING_MODE=shadow" in env_vars
    assert "TYPESAFE_RAG_MODE=shadow" in env_vars
    assert "TYPESAFE_RECIPE_CLASSIFY_ENABLED=0" in env_vars
    assert "enforce" not in env_vars
    for forbidden in ("TYPESAFE_NEWS_MODE", "TYPESAFE_RESEARCH_VERIFY_MODE", "TYPESAFE_ALLOW_"):
        assert forbidden not in env_vars


@pytest.mark.parametrize("text", [_api_deploy_step(), DEPLOY_SCRIPT.read_text(encoding="utf-8")])
def test_both_deploy_paths_carry_jev_config(text):
    assert "jev_cloud_run_config.sh" in text
    set_env = re.search(r'--set-env-vars "([^"]+)"', text)
    assert set_env and "${JEV_CLOUD_RUN_ENV_VARS}" in set_env.group(1)
    assert 'jev_typesafe_secret_ref "' in text
    assert "TYPESAFE_API_KEY=${jev_ref}" in text


def test_ci_wires_the_key_only_by_explicit_opt_in():
    assert "TYPESAFE_API_KEY_SECRET: ${{ vars.TYPESAFE_API_KEY_SECRET }}" in _api_deploy_step()


def _run_secret_ref(tmp_path, *, gcloud_exit: int, secret_var: str | None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "gcloud"
    stub.write_text(f"#!/bin/sh\nexit {gcloud_exit}\n", encoding="utf-8")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("TYPESAFE_API_KEY_SECRET", None)
    if secret_var is not None:
        env["TYPESAFE_API_KEY_SECRET"] = secret_var
    return subprocess.run(
        ["bash", "-c", f'set -Eeuo pipefail; source "{LIB}"; jev_typesafe_secret_ref p'],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_secret_ref_uses_explicit_variable(tmp_path):
    result = _run_secret_ref(tmp_path, gcloud_exit=1, secret_var="FOO")
    assert result.returncode == 0
    assert result.stdout.strip() == "FOO:latest"


def test_secret_ref_detects_existing_secret(tmp_path):
    result = _run_secret_ref(tmp_path, gcloud_exit=0, secret_var=None)
    assert result.returncode == 0
    assert result.stdout.strip() == "TYPESAFE_API_KEY:latest"


def test_missing_secret_never_blocks_deploy(tmp_path):
    result = _run_secret_ref(tmp_path, gcloud_exit=1, secret_var=None)
    assert result.returncode == 0
    assert result.stdout == ""
    assert "Jev guards stay off" in result.stderr
