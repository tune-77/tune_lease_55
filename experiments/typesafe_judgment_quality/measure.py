"""Offline measurement for the TypeSafe/Jev judgment-asset quality pilot.

Why this exists
---------------
``_judgment_asset_quality()`` decides, from keyword markers alone, whether an
extracted claim is a reusable judgment asset or a textbook generality. That one
verdict decides whether a claim ever reaches a human via ``_promotion_status()``.
On the corpus as of this writing it suppresses roughly three quarters of all
candidates, and a suppressed claim leaves no review trace -- so the error rate
of that filter is currently unobservable.

This script makes it observable before anything is switched on in production.

The script is read-only with respect to the pipeline: it reads the candidate
JSONL the daily run already wrote, imports the guard without enabling it, and
writes only under ``results/`` (which is gitignored).

Usage
-----
    python3 experiments/typesafe_judgment_quality/measure.py              # rule distribution, offline
    python3 experiments/typesafe_judgment_quality/measure.py --inspect 20 # sample suppressed claims
    python3 experiments/typesafe_judgment_quality/measure.py --send       # calls TypeSafe
    python3 experiments/typesafe_judgment_quality/measure.py --compare results/judged_*.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_JSONL = ROOT / "data" / "autoresearch_judgment_asset_candidates.jsonl"

sys.path.insert(0, str(ROOT))

import typesafe_judgment_quality_guard as guard  # noqa: E402


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_autoresearch_judgment_asset_candidates_measure",
        ROOT / "scripts" / "build_autoresearch_judgment_asset_candidates.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"candidate JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rule_verdicts(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, list[str]]]:
    """Recompute the rule verdict rather than trusting the stored field."""
    builder = _load_builder()
    out = []
    for row in rows:
        quality, reasons = builder._judgment_asset_quality(
            str(row.get("claim") or ""), str(row.get("candidate_type") or "")
        )
        out.append((row, quality, reasons))
    return out


def report_rule_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = rule_verdicts(rows)
    by_quality = collections.Counter(quality for _row, quality, _r in verdicts)
    by_type = collections.Counter(
        (str(row.get("candidate_type")), quality) for row, quality, _r in verdicts
    )
    reasons = collections.Counter(
        reason for _row, _q, rs in verdicts for reason in rs
    )
    safe, skipped = guard.filter_safe_claims(rows)

    summary = {
        "rows": len(rows),
        "rule_verdicts": dict(by_quality),
        "suppressed_share": round(
            by_quality.get("textbook_general", 0) / max(1, len(rows)), 3
        ),
        "privacy_safe_to_send": len(safe),
        "privacy_skipped": skipped,
        "top_reasons": dict(reasons.most_common(8)),
        "by_candidate_type": {
            f"{kind}/{quality}": count for (kind, quality), count in sorted(by_type.items())
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def inspect(rows: list[dict[str, Any]], limit: int) -> None:
    """Print claims the rule suppressed, so a human can judge the filter itself."""
    shown = 0
    for row, quality, reasons in rule_verdicts(rows):
        if quality == "textbook_general" and shown < limit:
            shown += 1
            print(f"--- [{shown}] {row.get('candidate_type')} / {row.get('research_date')}")
            print(f"    claim  : {row.get('claim')}")
            print(f"    reasons: {', '.join(reasons) or 'none'}")
    print(f"\n{shown} suppressed claims shown (limit {limit}).")


def send(rows: list[dict[str, Any]], *, batch_size: int) -> Path:
    if not guard.typesafe_judgment_quality_enabled():
        raise SystemExit(
            "TYPESAFE_JUDGMENT_QUALITY_ENABLED is not set (or no API key); refusing to send."
        )
    missing = [n for n, t in guard.JUDGMENT_QUALITY_CLASSES.items() if not str(t).strip()]
    if missing:
        raise SystemExit("class catalog is incomplete: " + ", ".join(missing))

    judged_all: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    safe, _skipped = guard.filter_safe_claims(rows)
    for start in range(0, len(safe), batch_size):
        window = safe[start : start + batch_size]
        subset = [rows[index] for index in window]
        judged, meta = guard.judge_claims(subset, max_claims=batch_size)
        for item in judged:
            item = dict(item)
            item["index"] = window[int(item["index"])]
            judged_all.append(item)
        metas.append(meta)
        print(f"batch {start // batch_size}: {meta.get('status')} n={len(window)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"judged_{stamp}.json"
    out.write_text(
        json.dumps({"judged": judged_all, "meta": metas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return out


def compare(rows: list[dict[str, Any]], paths: list[str]) -> dict[str, Any]:
    judged: list[dict[str, Any]] = []
    for pattern in paths:
        for name in sorted(glob.glob(pattern)):
            judged.extend(json.loads(Path(name).read_text(encoding="utf-8"))["judged"])
    if not judged:
        raise SystemExit("no judged records found")

    verdicts = {index: q for index, (_r, q, _rs) in enumerate(rule_verdicts(rows))}
    confusion: collections.Counter = collections.Counter()
    decisions: collections.Counter = collections.Counter()
    rescued = 0
    for item in judged:
        index = int(item["index"])
        rule = verdicts.get(index, "unknown")
        typesafe = str(item["asset_quality"])
        decision = str(item["decision"])
        confusion[f"{rule} -> {typesafe}"] += 1
        decisions[decision] += 1
        # Claims the rule dropped that a human would now see, either because Jev
        # confidently disagreed or because it could not resolve them.
        if rule == "textbook_general" and (typesafe != "textbook_general" or decision == "review"):
            rescued += 1

    summary = {
        "judged": len(judged),
        "decisions": dict(decisions),
        "confusion": dict(confusion.most_common()),
        "rescued_from_silent_suppression": rescued,
        "rescued_share_of_suppressed": round(
            rescued / max(1, sum(1 for v in verdicts.values() if v == "textbook_general")), 3
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--inspect", type=int, nargs="?", const=20, default=0)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--batch-size", type=int, default=guard.MAX_CLAIMS_PER_REQUEST)
    parser.add_argument("--compare", nargs="+", default=[])
    args = parser.parse_args()

    rows = load_rows(Path(args.jsonl))
    if args.compare:
        compare(rows, args.compare)
        return 0
    if args.send:
        send(rows, batch_size=max(1, min(guard.MAX_CLAIMS_PER_REQUEST, args.batch_size)))
        return 0
    if args.inspect:
        inspect(rows, args.inspect)
        return 0
    report_rule_distribution(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
