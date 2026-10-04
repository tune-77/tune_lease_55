#!/usr/bin/env python3
"""Cloudflare RAGシャドー比較ログを集計する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "data" / "cloudflare_rag_shadow_log.jsonl"


def summarize_rows(rows: list[dict]) -> dict:
    ok = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") == "error"]
    overlaps = [int(row.get("local_vectorize_overlap_at_5") or 0) for row in ok]
    changed = sum(bool(row.get("vectorize_reranker_changed_top1")) for row in ok)
    return {
        "total": len(rows),
        "ok": len(ok),
        "errors": len(errors),
        "reranker_changed_top1": changed,
        "reranker_changed_top1_rate": changed / len(ok) if ok else 0.0,
        "mean_local_vectorize_overlap_at_5": sum(overlaps) / len(overlaps) if overlaps else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    args = parser.parse_args()
    path = Path(args.log)
    if not path.is_file():
        print("比較ログはまだありません。一般的なリース知識質問を使うと自動記録されます。")
        return 0
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    summary = summarize_rows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
