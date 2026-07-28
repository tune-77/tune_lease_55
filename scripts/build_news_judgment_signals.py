#!/usr/bin/env python3
"""Build screening-time news judgment signals from Obsidian news actions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lease_news_digest import find_vault, write_news_judgment_signals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert collected Obsidian news into data/news_judgment_signals.jsonl."
    )
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian Vault path")
    parser.add_argument("--limit", type=int, default=5, help="Maximum news actions to convert")
    parser.add_argument("--use-llm", action="store_true", help="Ask Gemini to refine candidates before guard checks")
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    args = parser.parse_args()

    vault = args.vault.expanduser() if args.vault else find_vault()
    result = write_news_judgment_signals(
        date_str=args.date,
        vault=vault,
        limit=max(1, args.limit),
        use_llm=args.use_llm,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"date={result['date']}")
        print(f"count={result['count']}")
        print(f"path={result['path']}")
        print(f"latest_path={result['latest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
