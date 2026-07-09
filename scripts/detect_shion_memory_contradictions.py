#!/usr/bin/env python3
"""紫苑記憶の矛盾候補検出（レポートのみ・自動修正しない）。

高類似の記憶ペアのうち「同じ話題なのに数値が食い違う」ものを矛盾候補として
レポートに出す（例: 建機の耐用年数 6年 vs 8年）。判断と改訂は人間が行う:

    python3 scripts/revise_shion_memory.py --old-id <負けた記憶のID> ...

検出ロジック（意図的に保守的）:
- active な記憶のみ対象（revised / stale / deprecated は除外）
- 漢字連・カタカナ連・英数語をトークン化し Jaccard 類似度 >= しきい値のペア
- かつ「単位付き数値」（N年 / N日 / N% / N万円 / N点 など）が食い違うもの
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = REPO_ROOT / "data" / "shion_memory_index.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "shion_memory_contradictions_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "shion_memory_contradictions_latest.md"

_TOKEN_RE = re.compile(r"[一-龥]{2,}|[ァ-ヴー]{2,}|[a-zA-Z_]{3,}")
# 単位付き数値: 「6年」「35%」「1400万円」等。単位ごとに数値集合を比較する
_UNIT_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(年|日|ヶ月|か月|%|％|点|万円|千円|円|件|回)")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text or ""))


def _unit_numbers(text: str) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = {}
    for num, unit in _UNIT_NUM_RE.findall(text or ""):
        unit = unit.replace("％", "%").replace("か月", "ヶ月")
        result.setdefault(unit, set()).add(num)
    return {k: frozenset(v) for k, v in result.items()}


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_contradictions(
    records: list[dict],
    *,
    min_similarity: float = 0.45,
    max_pairs: int = 30,
) -> list[dict]:
    """矛盾候補ペアのリストを返す（スコア降順）。"""
    active = [
        r
        for r in records
        if str(r.get("status") or "active") == "active"
        and len(str(r.get("content") or "")) >= 12
    ]
    prepared = []
    for r in active:
        content = str(r.get("content") or "")
        units = _unit_numbers(content)
        if not units:
            continue  # 数値を持たない記憶は本方式では判定できない
        prepared.append((r, _tokens(content), units))

    candidates: list[dict] = []
    for (r1, t1, u1), (r2, t2, u2) in combinations(prepared, 2):
        shared_units = {
            unit: (u1[unit], u2[unit])
            for unit in u1.keys() & u2.keys()
            if u1[unit] != u2[unit] and not (u1[unit] & u2[unit])
        }
        if not shared_units:
            continue
        similarity = _jaccard(t1, t2)
        if similarity < min_similarity:
            continue
        candidates.append(
            {
                "similarity": round(similarity, 3),
                "conflicting_values": {
                    unit: {"a": sorted(v1), "b": sorted(v2)}
                    for unit, (v1, v2) in sorted(shared_units.items())
                },
                "a": {
                    "id": r1.get("id"),
                    "content": str(r1.get("content") or "")[:200],
                    "source_path": r1.get("source_path"),
                    "memory_type": r1.get("memory_type"),
                },
                "b": {
                    "id": r2.get("id"),
                    "content": str(r2.get("content") or "")[:200],
                    "source_path": r2.get("source_path"),
                    "memory_type": r2.get("memory_type"),
                },
            }
        )
    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates[:max_pairs]


def _render_markdown(candidates: list[dict], total_records: int) -> str:
    lines = [
        "# 紫苑記憶 矛盾候補レポート",
        "",
        f"- 生成: {datetime.now().isoformat(timespec='seconds')}",
        f"- 対象レコード: {total_records} 件 / 矛盾候補: {len(candidates)} 組",
        "- 対応方法: 正しい方を残し `scripts/revise_shion_memory.py --old-id <ID>` で改訂する",
        "",
    ]
    if not candidates:
        lines.append("矛盾候補は検出されませんでした。")
        return "\n".join(lines) + "\n"
    for i, c in enumerate(candidates, 1):
        conflicts = " / ".join(
            f"{unit}: {'・'.join(v['a'])} vs {'・'.join(v['b'])}"
            for unit, v in c["conflicting_values"].items()
        )
        lines += [
            f"## 候補 {i}（類似度 {c['similarity']}）",
            f"- 食い違い: {conflicts}",
            f"- A `{c['a']['id']}` ({c['a']['source_path']}): {c['a']['content']}",
            f"- B `{c['b']['id']}` ({c['b']['source_path']}): {c['b']['content']}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶の矛盾候補検出（レポートのみ）")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--max-pairs", type=int, default=30)
    args = parser.parse_args()

    try:
        data = json.loads(args.index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"警告: 記憶インデックスを読めません: {exc}", file=sys.stderr)
        return 0  # レポート専用のため異常終了でパイプラインを汚さない
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        print("警告: 記憶インデックスに records がありません", file=sys.stderr)
        return 0

    candidates = find_contradictions(
        records, min_similarity=args.min_similarity, max_pairs=args.max_pairs
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "total_records": len(records),
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_md.write_text(_render_markdown(candidates, len(records)), encoding="utf-8")
    print(f"矛盾候補: {len(candidates)} 組 → {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
