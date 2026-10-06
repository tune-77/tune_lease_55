#!/usr/bin/env python3
"""Summarize privacy-safe local AI usage metrics."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_runtime_client import usage_log_lock, usage_log_path  # noqa: E402


def summarize(path: Path, *, by_source: bool = False) -> list[dict]:
    groups: dict[tuple[str, ...], dict] = defaultdict(
        lambda: {
            "calls": 0,
            "errors": 0,
            "duration_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
    )
    log_paths = (path.with_suffix(f"{path.suffix}.1"), path)
    # Rotationと同じプロセス間ロック内で両世代を読む。途中でactiveが.1へ
    # 入れ替わり、一世代分が集計から抜ける競合を防ぐ。
    with usage_log_lock(path):
        if not any(log_path.exists() for log_path in log_paths):
            return []
        for log_path in log_paths:
            if not log_path.exists():
                continue
            with log_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    key = (str(item.get("provider") or "unknown"), str(item.get("feature") or "unknown"), str(item.get("model") or "unknown"))
                    if by_source:
                        # 呼び出し元（入口スクリプト名、worktree なら名前付き）。REV-484 以前の行は unknown
                        source = str(item.get("source") or "unknown")
                        key += (f"{source}@{item['worktree']}" if item.get("worktree") else source,)
                    row = groups[key]
                    row["calls"] += 1
                    row["errors"] += 0 if item.get("ok") else 1
                    row["duration_ms"] += float(item.get("duration_ms") or 0)
                    input_tokens = int(item.get("input_tokens") or 0)
                    output_tokens = int(item.get("output_tokens") or 0)
                    total_tokens = item.get("total_tokens")
                    row["input_tokens"] += input_tokens
                    row["output_tokens"] += output_tokens
                    row["cached_tokens"] += int(item.get("cached_tokens") or 0)
                    row["total_tokens"] += (
                        int(total_tokens) if total_tokens is not None else input_tokens + output_tokens
                    )
    result = []
    for key, row in sorted(groups.items()):
        provider, feature, model = key[:3]
        calls = row["calls"]
        result.append(
            {
                **({"source": key[3]} if by_source else {}),
                "provider": provider,
                "feature": feature,
                "model": model,
                "calls": calls,
                "errors": row["errors"],
                "avg_duration_ms": round(row["duration_ms"] / calls, 2) if calls else 0,
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "total_tokens": row["total_tokens"],
                "cached_tokens": row["cached_tokens"],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=usage_log_path())
    parser.add_argument("--by-source", action="store_true", help="呼び出し元（入口スクリプト・worktree）別にも分ける")
    args = parser.parse_args()
    rows = summarize(args.path, by_source=args.by_source)
    if not rows:
        print(f"AI usage log is empty: {args.path}")
        return 0
    prefix = "source\t" if args.by_source else ""
    print(prefix + "provider\tfeature\tmodel\tcalls\terrors\tavg_ms\tinput_tokens\toutput_tokens\ttotal_tokens\tcached_tokens")
    for row in rows:
        print(
            (f"{row['source']}\t" if args.by_source else "")
            + f"{row['provider']}\t{row['feature']}\t{row['model']}\t{row['calls']}\t{row['errors']}\t"
            f"{row['avg_duration_ms']}\t{row['input_tokens']}\t{row['output_tokens']}\t{row['total_tokens']}\t{row['cached_tokens']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
