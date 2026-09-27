#!/usr/bin/env python3
"""Prepare a privacy-minimized payload for a bounded Jev judgment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jev_safe_gateway import (
    DEFAULT_AUDIT_PATH,
    GatewayInputError,
    prepare_gateway_request,
    verify_public_source,
)


def _read_payload(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSON file, or - for stdin")
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="Local content-free JSONL audit path",
    )
    parser.add_argument(
        "--outbound-only",
        action="store_true",
        help="Print only the safe outbound object when allowed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_gateway_request(
            _read_payload(args.input),
            audit_path=args.audit,
            public_source_verifier=verify_public_source,
        )
    except (GatewayInputError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 1
    if args.outbound_only:
        print(json.dumps(result["outbound"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "allowed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
