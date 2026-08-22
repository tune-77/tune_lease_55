#!/usr/bin/env python3
"""Summarize `/api/chat` system-prompt composition from the measurement log.

data/chat_prompt_composition.jsonl（api/chat_prompt_cost.py が追記）を読み、
どのプロンプトブロックがトークンを食っているかをブロック別に集計する。

計測結果を読むだけの read-only レポート。プロンプト・ルーティング・
スコアリングには一切触れない。削減対象は、この出力の上位を見て人が決める。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "chat_prompt_composition.jsonl"


def read_entries(path: Path, *, surface: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if surface and str(entry.get("surface") or "") != surface:
            continue
        rows.append(entry)
    return rows


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """ブロック別の平均トークンを、出現回数つきで集計する。

    合計ではなく平均で並べる。たまにしか出ないが出ると巨大なブロックを、
    毎回出る小さなブロックに埋もれさせないため。
    """
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for entry in entries:
        for name, stat in (entry.get("blocks") or {}).items():
            if not isinstance(stat, dict):
                continue
            totals[name] = totals.get(name, 0) + int(stat.get("tokens") or 0)
            counts[name] = counts.get(name, 0) + 1

    blocks = [
        {
            "block": name,
            "appearances": counts[name],
            "avg_tokens": round(totals[name] / counts[name], 1),
            "total_tokens": totals[name],
        }
        for name in totals
    ]
    blocks.sort(key=lambda item: -item["avg_tokens"])

    sample_count = len(entries)
    final_tokens = [int(e.get("final_tokens") or 0) for e in entries]
    unaccounted = [int(e.get("unaccounted_chars") or 0) for e in entries]
    return {
        "samples": sample_count,
        "avg_final_tokens": round(sum(final_tokens) / sample_count, 1) if sample_count else 0,
        "max_final_tokens": max(final_tokens) if final_tokens else 0,
        "capped_samples": sum(1 for e in entries if e.get("capped")),
        "avg_unaccounted_chars": round(sum(unaccounted) / sample_count, 1) if sample_count else 0,
        "blocks": blocks,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Chat Prompt Composition",
        "",
        f"- Samples: {summary['samples']}",
        f"- Avg final tokens: {summary['avg_final_tokens']}",
        f"- Max final tokens: {summary['max_final_tokens']}",
        f"- Capped samples (末尾ブロックが落とされた回数): {summary['capped_samples']}",
        f"- Avg unaccounted chars (名前を付けていないブロックの合計): {summary['avg_unaccounted_chars']}",
        "",
        "## Blocks by average tokens",
        "",
        "| block | appearances | avg tokens | total tokens |",
        "|---|---:|---:|---:|",
    ]
    for item in summary["blocks"]:
        lines.append(
            f"| {item['block']} | {item['appearances']} | {item['avg_tokens']} | {item['total_tokens']} |"
        )
    if not summary["blocks"]:
        lines.append("| (計測データなし) | 0 | 0 | 0 |")
    lines += [
        "",
        "avg unaccounted chars が大きい場合、まだ名前を付けていないブロックが太っている。",
        "api/chat_prompt_cost.py の呼び出し側 blocks に名前を足して再計測する。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--surface", default="", help="next_chat_rag / next_chat_general で絞る")
    parser.add_argument("--json", action="store_true", help="Markdown ではなく JSON で出力")
    args = parser.parse_args()

    entries = read_entries(Path(args.input), surface=args.surface)
    summary = summarize(entries)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render(summary))


if __name__ == "__main__":
    main()
