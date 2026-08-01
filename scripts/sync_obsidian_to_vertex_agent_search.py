#!/usr/bin/env python3
"""Sync selected Obsidian lease notes to the Vertex AI Search pilot corpus.

Default mode is local-only. Add --upload to copy the exported corpus to GCS,
and --import-documents to trigger Discovery Engine import from documents.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_obsidian_for_agent_search import (  # noqa: E402
    DEFAULT_GCS_PREFIX,
    DEFAULT_OUTPUT,
    DEFAULT_VAULT,
    export_notes,
)

DEFAULT_PROJECT_ID = "gen-lang-client-0420497423"
DEFAULT_LOCATION = "global"
DEFAULT_COLLECTION = "default_collection"
DEFAULT_DATA_STORE_ID = "tune-lease-55-obsidian-lease"
DEFAULT_ERROR_PREFIX = "gs://tune-lease-55-data/agent-search/lease-knowledge-errors/"
DEFAULT_BRANCH = "default_branch"
DEFAULT_REPORT = REPO_ROOT / "reports" / "vertex_obsidian_sync_latest.json"
DEFAULT_STATE = REPO_ROOT / "data" / "agent_search" / "vertex_obsidian_sync_state.json"


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def access_token() -> str:
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True, timeout=15).strip()


def post_json(url: str, body: dict[str, Any], *, project_id: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token()}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": project_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def get_json(url: str, *, project_id: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {access_token()}",
            "X-Goog-User-Project": project_id,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def import_documents(
    *,
    project_id: str,
    location: str,
    collection: str,
    data_store_id: str,
    branch: str,
    manifest_uri: str,
    error_prefix: str,
    reconciliation_mode: str,
    force_refresh_content: bool,
    timeout: float,
) -> dict[str, Any]:
    parent = (
        f"projects/{project_id}/locations/{location}/collections/{collection}/"
        f"dataStores/{data_store_id}/branches/{branch}"
    )
    encoded_parent = urllib.parse.quote(parent, safe="/")
    url = f"https://discoveryengine.googleapis.com/v1/{encoded_parent}/documents:import"
    body = {
        "gcsSource": {
            "inputUris": [manifest_uri],
            "dataSchema": "document",
        },
        "errorConfig": {"gcsPrefix": error_prefix},
        "reconciliationMode": reconciliation_mode,
        "forceRefreshContent": force_refresh_content,
    }
    return post_json(url, body, project_id=project_id, timeout=timeout)


def wait_operation(name: str, *, project_id: str, timeout: float, poll_seconds: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    encoded_name = urllib.parse.quote(name, safe="/")
    url = f"https://discoveryengine.googleapis.com/v1/{encoded_name}"
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = get_json(url, project_id=project_id, timeout=30)
        if last.get("done"):
            return last
        print(f"operation pending: {name}")
        time.sleep(poll_seconds)
    return {"done": False, "name": name, "timeout": True, "last": last}


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_signature(exported: list[Any]) -> str:
    payload = [
        {
            "source_path": note.source_path,
            "digest": note.digest,
            "size_chars": note.size_chars,
        }
        for note in exported
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-docs", type=int, default=180)
    parser.add_argument("--gcs-prefix", default=DEFAULT_GCS_PREFIX)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--data-store-id", default=DEFAULT_DATA_STORE_ID)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--error-prefix", default=DEFAULT_ERROR_PREFIX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--upload", action="store_true", help="Upload exported corpus to GCS.")
    parser.add_argument("--import-documents", action="store_true", help="Trigger Discovery Engine import from documents.jsonl.")
    parser.add_argument("--wait", action="store_true", help="Wait for the import operation to finish.")
    parser.add_argument("--delete-stale-gcs", action="store_true", help="Delete GCS objects that no longer exist in the export output.")
    parser.add_argument("--force", action="store_true", help="Upload/import even when the exported corpus signature did not change.")
    parser.add_argument("--reconciliation-mode", choices=["INCREMENTAL", "FULL"], default="INCREMENTAL")
    parser.add_argument("--no-force-refresh-content", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    exported = export_notes(args.vault, args.output, args.max_docs, args.gcs_prefix)
    signature = export_signature(exported)
    previous_state = read_state(args.state)
    previous_signature = str(previous_state.get("export_signature") or "")
    changed = bool(args.force or signature != previous_signature)
    manifest_uri = f"{args.gcs_prefix.rstrip('/')}/documents.jsonl"
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "vault": str(args.vault),
        "output": str(args.output),
        "gcs_prefix": args.gcs_prefix,
        "manifest_uri": manifest_uri,
        "exported_count": len(exported),
        "export_signature": signature,
        "previous_export_signature": previous_signature,
        "changed": changed,
        "uploaded": False,
        "import_requested": False,
        "delete_stale_gcs": bool(args.delete_stale_gcs),
        "dry_run": bool(args.dry_run),
    }

    if not changed and (args.upload or args.import_documents):
        report["skipped_reason"] = "export_signature_unchanged"
        write_report(args.report, report)
        print(f"exported={len(exported)}")
        print(f"changed=false")
        print(f"skipped=export_signature_unchanged")
        print(f"report={args.report}")
        return

    if args.upload:
        if args.delete_stale_gcs:
            run_command(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    "--delete-unmatched-destination-objects",
                    str(args.output),
                    args.gcs_prefix,
                ],
                dry_run=args.dry_run,
            )
        else:
            run_command(
                ["gcloud", "storage", "rsync", "--recursive", str(args.output), args.gcs_prefix],
                dry_run=args.dry_run,
            )
        report["uploaded"] = not args.dry_run

    if args.import_documents:
        if not args.upload and not args.dry_run:
            raise SystemExit("--import-documents requires --upload unless --dry-run is set")
        report["import_requested"] = True
        if args.dry_run:
            report["operation"] = {"dry_run": True}
        else:
            try:
                operation = import_documents(
                    project_id=args.project_id,
                    location=args.location,
                    collection=args.collection,
                    data_store_id=args.data_store_id,
                    branch=args.branch,
                    manifest_uri=manifest_uri,
                    error_prefix=args.error_prefix,
                    reconciliation_mode=args.reconciliation_mode,
                    force_refresh_content=not args.no_force_refresh_content,
                    timeout=args.timeout_seconds,
                )
                report["operation"] = operation
                name = str(operation.get("name") or "")
                if args.wait and name:
                    report["operation_result"] = wait_operation(
                        name,
                        project_id=args.project_id,
                        timeout=max(args.timeout_seconds, 300),
                    )
            except (OSError, subprocess.SubprocessError, urllib.error.URLError, TimeoutError) as exc:
                report["import_error"] = str(exc)
                write_report(args.report, report)
                raise

    write_report(args.report, report)
    if changed and report.get("uploaded"):
        write_state(
            args.state,
            {
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "export_signature": signature,
                "exported_count": len(exported),
                "manifest_uri": manifest_uri,
                "gcs_prefix": args.gcs_prefix,
                "operation_name": (report.get("operation") or {}).get("name"),
                "dry_run": bool(args.dry_run),
            },
        )
    print(f"exported={len(exported)}")
    print(f"changed={str(changed).lower()}")
    print(f"manifest_uri={manifest_uri}")
    print(f"report={args.report}")
    if report.get("operation"):
        print(f"operation={report['operation'].get('name')}")


if __name__ == "__main__":
    main()
