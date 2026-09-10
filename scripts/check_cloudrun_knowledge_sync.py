#!/usr/bin/env python3
"""Fail when the live Cloud Run Vault -> ChromaDB sync is incomplete."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PATH = "/api/system/cloud-status"

# frontend/src/proxy.ts が /api/system/knowledge-sync-health を Basic 認証の
# 代わりに要求する専用シークレット。ブラウザ用のトンネルパスワードとは別物。
PROBE_TOKEN_ENV_VAR = "KNOWLEDGE_SYNC_PROBE_TOKEN"


def validate_sync_response(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "response is not a JSON object"

    sync = payload.get("knowledge_sync")
    if not isinstance(sync, dict):
        return False, "knowledge_sync status is missing"

    state = sync.get("state")
    if sync.get("ready") is not True or state != "ready":
        reason = sync.get("reason") or "unknown"
        vault_count = sync.get("vault_markdown_count", 0)
        chroma_count = sync.get("chroma_document_count", 0)
        return (
            False,
            f"knowledge sync {state}: {reason} "
            f"(vault={vault_count}, chroma={chroma_count})",
        )
    return True, "ok"


def fetch_sync_response(base_url: str, path: str, timeout: float, probe_token: str) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Accept": "application/json", "User-Agent": "tune-lease-knowledge-sync/1.0"}
    if probe_token:
        headers["X-Sync-Probe-Key"] = probe_token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=15.0)
    args = parser.parse_args()

    probe_token = os.environ.get(PROBE_TOKEN_ENV_VAR, "").strip()
    if not probe_token:
        print(f"[FAIL] {PROBE_TOKEN_ENV_VAR} is not set; the probe endpoint requires it")
        return 1

    attempts = max(1, args.attempts)
    last_reason = "unknown failure"
    for attempt in range(1, attempts + 1):
        try:
            payload = fetch_sync_response(args.base_url, args.path, args.timeout, probe_token)
            healthy, reason = validate_sync_response(payload)
            if healthy:
                sync = payload["knowledge_sync"]
                print(
                    "[OK] knowledge sync ready: "
                    f"vault={sync['vault_markdown_count']} "
                    f"chroma={sync['chroma_document_count']}"
                )
                return 0
            last_reason = reason
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_reason = f"{type(exc).__name__}: {exc}"

        print(f"[WARN] knowledge sync attempt {attempt}/{attempts} failed: {last_reason}")
        if attempt < attempts:
            time.sleep(max(0.0, args.retry_delay))

    print(f"[FAIL] knowledge sync unhealthy after {attempts} attempts: {last_reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
