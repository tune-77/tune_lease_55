#!/usr/bin/env python3
"""Check that Cloud Run demo data can drive the dashboard and graph APIs.

This is intentionally read-only. It validates the local demo DB, the packaged
Cloud Run bundle, and optionally live HTTP endpoints when --base-url is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEMO_DB = ROOT / "data" / "demo.db"
BUNDLE_DATA_DIR = ROOT / ".cloudrun_bundle" / "data"
BUNDLE_DEMO_DB = BUNDLE_DATA_DIR / "demo.db"
BUNDLE_LEASE_DB = BUNDLE_DATA_DIR / "lease_data.db"
LOCAL_PERSONAL_MEMORY = ROOT / "data" / "user_personal_memory.md"
BUNDLE_PERSONAL_MEMORY = BUNDLE_DATA_DIR / "user_personal_memory.md"


class CheckRun:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def print_report(self) -> None:
        for message in self.infos:
            print(f"[OK] {message}")
        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.failures:
            print(f"[FAIL] {message}")
        if not self.failures:
            print("Cloud Run demo readiness: PASS")
        else:
            print("Cloud Run demo readiness: FAIL")


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def check_db(path: Path, label: str, checks: CheckRun) -> dict[str, Any] | None:
    if not path.exists():
        checks.fail(f"{label} missing: {path.relative_to(ROOT)}")
        return None

    try:
        conn = connect(path)
    except sqlite3.Error as exc:
        checks.fail(f"{label} cannot open: {exc}")
        return None

    try:
        tables = table_names(conn)
        if "past_cases" not in tables:
            checks.fail(f"{label} has no past_cases table")
            return None

        cols = column_names(conn, "past_cases")
        required_cols = {"id", "timestamp", "industry_sub", "score", "final_status", "data"}
        missing_cols = sorted(required_cols - cols)
        if missing_cols:
            checks.fail(f"{label} missing columns in past_cases: {', '.join(missing_cols)}")
            return None

        total = int(scalar(conn, "SELECT COUNT(*) FROM past_cases") or 0)
        scored = int(scalar(conn, "SELECT COUNT(*) FROM past_cases WHERE score IS NOT NULL AND score > 0") or 0)
        won = int(scalar(conn, "SELECT COUNT(*) FROM past_cases WHERE final_status IN ('成約','検収完了','検収')") or 0)
        lost = int(scalar(conn, "SELECT COUNT(*) FROM past_cases WHERE final_status = '失注'") or 0)
        industries = int(
            scalar(
                conn,
                "SELECT COUNT(DISTINCT industry_sub) FROM past_cases "
                "WHERE industry_sub IS NOT NULL AND industry_sub != '' AND industry_sub != '0'",
            )
            or 0
        )
        dated = int(
            scalar(
                conn,
                "SELECT COUNT(*) FROM past_cases WHERE timestamp IS NOT NULL AND timestamp != ''",
            )
            or 0
        )
        visual_ready = int(
            scalar(
                conn,
                """
                SELECT COUNT(*) FROM past_cases
                WHERE score IS NOT NULL AND score > 0
                  AND final_status IN ('成約','失注','検収完了','検収')
                  AND json_extract(data, '$.inputs.acquisition_cost') IS NOT NULL
                """,
            )
            or 0
        )
        dept_count = int(
            scalar(
                conn,
                "SELECT COUNT(DISTINCT sales_dept) FROM past_cases "
                "WHERE sales_dept IS NOT NULL AND sales_dept NOT IN ('', '未設定', '0')",
            )
            or 0
        ) if "sales_dept" in cols else 0

        if total == 0:
            checks.fail(f"{label} past_cases is empty")
        if scored == 0:
            checks.fail(f"{label} has no positive scores")
        if won == 0 or lost == 0:
            checks.fail(f"{label} needs both won and lost cases for graphs: won={won}, lost={lost}")
        if industries == 0:
            checks.fail(f"{label} has no industry_sub values")
        if visual_ready < 3:
            checks.fail(f"{label} has too few visual-ready cases: {visual_ready}")
        if dated == 0:
            checks.warn(f"{label} has no timestamp values")
        if dept_count == 0:
            checks.warn(f"{label} has no usable sales_dept values; department charts may be sparse")

        checks.info(
            f"{label}: cases={total}, scored={scored}, won={won}, lost={lost}, "
            f"industries={industries}, visual_ready={visual_ready}, departments={dept_count}"
        )
        return {
            "total": total,
            "scored": scored,
            "won": won,
            "lost": lost,
            "industries": industries,
            "visual_ready": visual_ready,
            "departments": dept_count,
        }
    except sqlite3.Error as exc:
        checks.fail(f"{label} query failed: {exc}")
        return None
    finally:
        conn.close()


def check_bundle_alias(checks: CheckRun) -> None:
    if not BUNDLE_DEMO_DB.exists() or not BUNDLE_LEASE_DB.exists():
        return
    if BUNDLE_DEMO_DB.stat().st_size != BUNDLE_LEASE_DB.stat().st_size:
        checks.warn(
            "bundle data/demo.db and data/lease_data.db differ. "
            "For demo Cloud Run, legacy graph modules read lease_data.db."
        )
        return
    checks.info("bundle demo.db is mirrored to lease_data.db for legacy graph/stat modules")


def check_personal_memory_pack(checks: CheckRun) -> None:
    if not LOCAL_PERSONAL_MEMORY.exists():
        checks.fail("data/user_personal_memory.md missing; Shion personal memory priority cannot be enforced")
        return
    text = LOCAL_PERSONAL_MEMORY.read_text(encoding="utf-8", errors="replace")
    if "Dog name:" not in text or "Priority Rule" not in text:
        checks.fail("data/user_personal_memory.md is missing required personal memory sections")
    else:
        checks.info("local personal memory file is present")
    if BUNDLE_PERSONAL_MEMORY.exists():
        checks.info("bundle data/user_personal_memory.md is present")
    else:
        checks.warn("bundle data/user_personal_memory.md is missing; rerun package_cloud_run_bundle.sh before deploy")


def check_ignore_files(checks: CheckRun) -> None:
    for rel in (".dockerignore", ".gcloudignore"):
        path = ROOT / rel
        if not path.exists():
            checks.warn(f"{rel} missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "data/*.db" in text:
            checks.info(f"{rel} excludes data/*.db; .cloudrun_bundle seeding is required")
        else:
            checks.warn(f"{rel} does not exclude data/*.db; confirm private DBs are not deployed")


def check_packaging_script(checks: CheckRun) -> None:
    path = ROOT / "scripts" / "package_cloud_run_bundle.sh"
    if not path.exists():
        checks.fail("scripts/package_cloud_run_bundle.sh missing")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for needle in ("CLOUDRUN_DATA_MODE", "data/demo.db", "DATA_OUT/lease_data.db"):
        if needle not in text:
            checks.fail(f"package_cloud_run_bundle.sh does not mention {needle}")
    checks.info("package_cloud_run_bundle.sh contains demo-mode bundle safeguards")


def check_deploy_scripts(checks: CheckRun) -> None:
    for rel in ("scripts/deploy_cloud_run.sh", "scripts/deploy_cloud_run_api.sh"):
        path = ROOT / rel
        if not path.exists():
            checks.warn(f"{rel} missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        required = (
            "check_cloudrun_demo_readiness.py",
            "CLOUDRUN_DATA_MODE",
            "DATABASE_URL/Cloud SQL is intentionally not attached",
            "DB_PATH=data/demo.db",
        )
        missing = [needle for needle in required if needle not in text]
        if missing:
            checks.fail(f"{rel} is missing predeploy safeguards: {', '.join(missing)}")
        else:
            checks.info(f"{rel} runs predeploy checks and protects demo mode from Cloud SQL")


def http_json(base_url: str, path: str, timeout: float) -> Any:
    url = base_url.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def check_http(base_url: str, timeout: float, checks: CheckRun) -> None:
    endpoints = {
        "/api/dashboard/stats": lambda payload: isinstance(payload, dict) and bool(payload),
        "/api/cases/industry-winrate": lambda payload: bool((payload or {}).get("items")),
        "/api/visual/data": lambda payload: bool((payload or {}).get("cases")),
        "/api/shion/inner-state": lambda payload: isinstance(payload, dict) and bool(payload),
    }
    for path, predicate in endpoints.items():
        try:
            payload = http_json(base_url, path, timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            checks.fail(f"{base_url.rstrip('/')}{path} failed: {exc}")
            continue
        if predicate(payload):
            checks.info(f"HTTP {path} returned non-empty payload")
        else:
            checks.fail(f"HTTP {path} returned empty/unexpected payload")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", help="Optional running app/API base URL to probe")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()

    checks = CheckRun()
    check_ignore_files(checks)
    check_packaging_script(checks)
    check_deploy_scripts(checks)
    check_personal_memory_pack(checks)
    check_db(LOCAL_DEMO_DB, "local data/demo.db", checks)
    check_db(BUNDLE_DEMO_DB, "bundle data/demo.db", checks)
    check_db(BUNDLE_LEASE_DB, "bundle data/lease_data.db", checks)
    check_bundle_alias(checks)

    if os.environ.get("DATA_DIR") and os.environ.get("DATA_DIR") != "/app/data":
        checks.warn(f"DATA_DIR is set locally to {os.environ['DATA_DIR']}")
    if os.environ.get("DB_PATH"):
        checks.info(f"DB_PATH is set: {os.environ['DB_PATH']}")
    else:
        checks.info(
            "DB_PATH is not set in this checker process; "
            "bundle lease_data.db mirroring covers the default Cloud Run demo path"
        )

    if args.base_url:
        check_http(args.base_url, args.timeout, checks)

    checks.print_report()
    if checks.failures:
        return 1
    if args.strict_warnings and checks.warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
