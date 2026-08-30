from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import check_cloudrun_demo_readiness as readiness


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "cloud_run_database_deploy_args.sh"


def _database_args(*, mode: str, instance: str, secret_exists: bool = True) -> subprocess.CompletedProcess[str]:
    secret_result = "return 0" if secret_exists else "return 1"
    script = f"""
set -u
source {HELPER!s}
gcloud() {{ {secret_result}; }}
declare -a deploy_args=()
PROJECT_ID=test-project
SERVICE_NAME=test-service
REGION=asia-northeast1
DATABASE_SECRET_NAME=DATABASE_URL
CLOUDRUN_DATA_MODE={mode}
CLOUDSQL_INSTANCE={instance}
configure_cloud_run_database_deploy_args || exit $?
printf '%s\n' "${{deploy_args[@]}}"
"""
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=False)


def test_production_without_instance_removes_stale_database_url() -> None:
    result = _database_args(mode="production", instance="")

    assert result.returncode == 0
    assert "--clear-cloudsql-instances" in result.stdout
    assert "--set-secrets" not in result.stdout


def test_demo_removes_stale_database_url() -> None:
    result = _database_args(mode="demo", instance="")

    assert result.returncode == 0
    assert "--clear-cloudsql-instances" in result.stdout


def test_stale_database_url_is_removed_in_separate_update() -> None:
    script = f"""
set -u
source {HELPER!s}
gcloud() {{
  if [[ "$1 $2 $3" == "run services describe" ]]; then
    printf 'API_ACCESS_KEY DATABASE_URL GEMINI_API_KEY\n'
    return 0
  fi
  printf 'GCLOUD_CALL %s\n' "$*"
}}
declare -a deploy_args=()
PROJECT_ID=test-project
SERVICE_NAME=test-service
REGION=asia-northeast1
DATABASE_SECRET_NAME=DATABASE_URL
CLOUDRUN_DATA_MODE=production
CLOUDSQL_INSTANCE=
configure_cloud_run_database_deploy_args || exit $?
printf 'DEPLOY_ARG %s\n' "${{deploy_args[@]}}"
"""

    result = subprocess.run(
        ["bash", "-c", script], text=True, capture_output=True, check=False
    )

    assert result.returncode == 0
    assert "GCLOUD_CALL run services update test-service" in result.stdout
    assert "--remove-secrets=DATABASE_URL" in result.stdout
    assert "DEPLOY_ARG --clear-cloudsql-instances" in result.stdout


def test_cloud_sql_requires_instance_and_secret_together() -> None:
    result = _database_args(
        mode="production",
        instance="test-project:asia-northeast1:tune-lease-db",
    )

    assert result.returncode == 0
    assert "--set-secrets" in result.stdout
    assert "DATABASE_URL=DATABASE_URL:latest" in result.stdout
    assert "--add-cloudsql-instances" in result.stdout
    assert "test-project:asia-northeast1:tune-lease-db" in result.stdout


def test_cloud_sql_instance_without_secret_fails_closed() -> None:
    result = _database_args(
        mode="production",
        instance="test-project:asia-northeast1:tune-lease-db",
        secret_exists=False,
    )

    assert result.returncode != 0
    assert "Secret Manager secret DATABASE_URL was not found" in result.stderr


def test_readiness_accepts_shared_database_deploy_guard() -> None:
    checks = readiness.CheckRun()

    readiness.check_deploy_scripts(checks)

    assert checks.failures == []
