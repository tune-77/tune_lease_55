#!/usr/bin/env python3
"""既存の判断資産・判断資産候補に knowledge_kind（policy=方針 / insight=知見）を一括で付ける。

対象: data/canonical_judgment_rules.json の rules、data/autoresearch_judgment_asset_candidates.jsonl の各行。
既に付いているものは変えない。分類は api.judgment_policy の決定的な規則（迷うものは知見）。
    python scripts/backfill_knowledge_kind.py            # 書き込む
    python scripts/backfill_knowledge_kind.py --dry-run  # 件数だけ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.judgment_policy import POLICY, classify_knowledge_kind  # noqa: E402


def backfill_rules(path: Path, *, dry_run: bool) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = {"added": 0, "policy": 0}
    for rule in data.get("rules") or []:
        if isinstance(rule, dict) and not rule.get("knowledge_kind"):
            rule["knowledge_kind"] = classify_knowledge_kind(str(rule.get("canonical_statement") or ""))
            counts["added"] += 1
            counts["policy"] += rule["knowledge_kind"] == POLICY
    if counts["added"] and not dry_run:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    return counts


def backfill_candidates(path: Path, *, dry_run: bool) -> dict[str, int]:
    counts = {"added": 0, "policy": 0}
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if isinstance(row, dict) and not row.get("knowledge_kind"):
            row["knowledge_kind"] = classify_knowledge_kind(str(row.get("edited_claim") or row.get("claim") or ""))
            counts["added"] += 1
            counts["policy"] += row["knowledge_kind"] == POLICY
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        out.append(line)
    if counts["added"] and not dry_run:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        tmp.replace(path)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=ROOT / "data" / "canonical_judgment_rules.json")
    parser.add_argument("--candidates", type=Path, default=ROOT / "data" / "autoresearch_judgment_asset_candidates.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for label, fn, path in (("rules", backfill_rules, args.rules), ("candidates", backfill_candidates, args.candidates)):
        if path.exists():
            print(label, fn(path, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
