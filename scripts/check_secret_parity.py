#!/usr/bin/env python3
"""Detect drift between a GCP Secret Manager secret and its GitHub Actions copy.

DASHBOARD_HEALTH_PROBE_TOKEN / KNOWLEDGE_SYNC_PROBE_TOKEN は Secret Manager と
GitHub Actions Secrets の両方に同じ値を人手で登録する運用になっている。
どちらか片方だけ作成・ローテーションされると、無言でプローブ用ワークフローが
401落ちし続ける（2026-09、DASHBOARD_HEALTH_PROBE_TOKENで実際に発生：GitHub
Actions側の登録を忘れたままCloud Run側だけ配線されていた）。このスクリプトは
両者の値を突き合わせてズレを検知する。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass


@dataclass
class ParityResult:
    secret_name: str
    ok: bool
    message: str


def _fetch_secret_manager_value(secret_name: str, project_id: str) -> tuple[str | None, str | None]:
    """Return (value, error). Exactly one of the two is None."""
    proc = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         "--secret", secret_name, "--project", project_id],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return proc.stdout, None
    return None, proc.stderr


def check_secret_parity(secret_name: str, project_id: str) -> ParityResult:
    github_value = os.environ.get(secret_name, "").strip()
    gcp_value, gcp_error = _fetch_secret_manager_value(secret_name, project_id)

    if gcp_error is not None and "NOT_FOUND" in gcp_error:
        if github_value:
            return ParityResult(
                secret_name, False,
                f"{secret_name}: GitHub Actions secret is set but no matching Secret Manager "
                f"secret exists in {project_id}",
            )
        return ParityResult(secret_name, True, f"{secret_name}: neither side configured (skip)")

    if gcp_error is not None:
        # このジョブのWIFサービスアカウントに secretmanager.versions.access が
        # 無いと値を読めない。deploy.yml の describe 確認と同じ理由で、権限不足を
        # 実在しないと誤認して止めない（require_api_access_key_secret.sh 参照）。
        return ParityResult(
            secret_name, True,
            f"{secret_name}: could not read Secret Manager value to compare (needs "
            f"secretmanager.versions.access on the WIF service account); skipping this secret.\n"
            f"--- gcloud stderr ---\n{gcp_error}",
        )

    if not github_value:
        return ParityResult(
            secret_name, False,
            f"{secret_name}: Secret Manager has a value but the GitHub Actions secret is not "
            f"set; the probe workflow using it will keep failing with 401",
        )

    if gcp_value.strip() != github_value:
        return ParityResult(
            secret_name, False,
            f"{secret_name}: value differs between Secret Manager and the GitHub Actions secret",
        )

    return ParityResult(secret_name, True, f"{secret_name}: values match")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--secret", action="append", required=True, dest="secrets")
    args = parser.parse_args()

    failed = False
    for secret_name in args.secrets:
        result = check_secret_parity(secret_name, args.project)
        print(f"{'[OK]' if result.ok else '[FAIL]'} {result.message}")
        failed = failed or not result.ok

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
