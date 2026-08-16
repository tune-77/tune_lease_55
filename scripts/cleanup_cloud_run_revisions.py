#!/usr/bin/env python3
"""Clean up old, 0%-traffic Cloud Run revisions for tune-lease-55-api / tune-lease-55-web.

Read-only by default (prints the deletion plan). Pass --apply to actually
delete revisions.

Safety rules:
  - Any revision referenced by the service's current traffic spec (serving
    traffic or tagged at 0%) is never deleted.
  - Among the remaining revisions, the --keep most recently created ones per
    service are kept for rollback headroom; only older revisions beyond that
    are deleted.

Requires the gcloud CLI to be installed and authenticated, e.g.:
  gcloud auth activate-service-account --key-file=/path/to/key.json
  # or Application Default Credentials:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

Usage:
  python3 scripts/cleanup_cloud_run_revisions.py                # dry run
  python3 scripts/cleanup_cloud_run_revisions.py --apply         # delete
  python3 scripts/cleanup_cloud_run_revisions.py --keep 3 --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

DEFAULT_SERVICES = ["tune-lease-55-api", "tune-lease-55-web"]
DEFAULT_REGION = "asia-northeast1"


def run_gcloud_json(args: list[str]) -> object:
    result = subprocess.run(
        ["gcloud", *args, "--format=json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout or "null")


def get_project(explicit: str | None) -> str:
    if explicit:
        return explicit
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=True,
    )
    project = result.stdout.strip()
    if not project or project == "(unset)":
        raise SystemExit("PROJECT_ID is required (gcloud config set project ... or --project).")
    return project


def traffic_revision_names(project: str, region: str, service: str) -> set[str]:
    """Revisions referenced by the service's current traffic spec (any percent, tagged or not)."""
    data = run_gcloud_json(
        ["run", "services", "describe", service, "--project", project, "--region", region]
    )
    traffic = (data or {}).get("status", {}).get("traffic") or []
    return {entry["revisionName"] for entry in traffic if entry.get("revisionName")}


def list_revisions(project: str, region: str, service: str) -> list[dict]:
    data = run_gcloud_json(
        [
            "run", "revisions", "list",
            "--service", service,
            "--project", project,
            "--region", region,
            "--sort-by", "~metadata.creationTimestamp",
        ]
    )
    return list(data or [])


def delete_revision(project: str, region: str, name: str) -> None:
    subprocess.run(
        ["gcloud", "run", "revisions", "delete", name, "--project", project, "--region", region, "--quiet"],
        capture_output=True,
        text=True,
        check=True,
    )


def plan_for_service(project: str, region: str, service: str, keep: int) -> tuple[list[str], list[str]]:
    """Returns (kept_names, delete_names). Revisions are newest-first."""
    protected = traffic_revision_names(project, region, service)
    revisions = list_revisions(project, region, service)

    kept: list[str] = []
    delete: list[str] = []
    headroom = keep
    for rev in revisions:
        name = rev["metadata"]["name"]
        if name in protected:
            kept.append(name)
            continue
        if headroom > 0:
            kept.append(name)
            headroom -= 1
            continue
        delete.append(name)
    return kept, delete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", help="GCP project ID (default: gcloud config get-value project)")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--service",
        action="append",
        dest="services",
        help="Service name; repeatable. Default: tune-lease-55-api, tune-lease-55-web",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Number of most-recent non-serving revisions to keep per service (default: 5)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this flag, only prints the plan (dry run).",
    )
    args = parser.parse_args()

    if shutil.which("gcloud") is None:
        print("gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install", file=sys.stderr)
        return 1

    project = get_project(args.project)
    services = args.services or DEFAULT_SERVICES

    print(f"Project: {project} / Region: {args.region} / Keep: {args.keep} most recent 0%-traffic revisions per service")
    print("Mode: APPLY (will delete)" if args.apply else "Mode: DRY RUN (no changes; pass --apply to delete)")

    total_delete = 0
    for service in services:
        print(f"\n=== {service} ===")
        try:
            kept, delete = plan_for_service(project, args.region, service, args.keep)
        except subprocess.CalledProcessError as exc:
            print(f"  failed to inspect service (skipping): {exc.stderr.strip()}", file=sys.stderr)
            continue

        print(f"  keeping {len(kept)} revision(s) (serving traffic or within keep window)")
        if not delete:
            print("  no 0%-traffic revisions beyond the keep window; nothing to delete")
            continue

        for name in delete:
            print(f"  {'deleting' if args.apply else 'would delete'}: {name}")
        total_delete += len(delete)

        if args.apply:
            for name in delete:
                try:
                    delete_revision(project, args.region, name)
                except subprocess.CalledProcessError as exc:
                    print(f"    failed to delete {name}: {exc.stderr.strip()}", file=sys.stderr)

    print(f"\n{'Deleted' if args.apply else 'Would delete'} {total_delete} revision(s) total.")
    if not args.apply and total_delete:
        print("Re-run with --apply to actually delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
