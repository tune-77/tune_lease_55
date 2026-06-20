#!/usr/bin/env python3
"""
RAG フィードバックログを集計し、ブースト/ペナルティ候補を
api/rule_engine/ledger_rules.json に pending_review: true で追記する。

基準:
  ペナルティ候補: bad 率 >= 60% かつ 5 件以上
  ブースト候補:   good 率 >= 70% かつ 5 件以上
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_FEEDBACK_LOG = _REPO_ROOT / "data" / "rag_feedback_log.jsonl"
_LEDGER_PATH = _REPO_ROOT / "api" / "rule_engine" / "ledger_rules.json"

_MIN_COUNT = 5
_BAD_RATE_THRESHOLD = 0.60
_GOOD_RATE_THRESHOLD = 0.70


def _normalize_path(obsidian_ref: str) -> str:
    """[[path#section]] → path（ファイルレベルで集計）"""
    ref = re.sub(r"^\[\[", "", obsidian_ref)
    ref = re.sub(r"\]\]$", "", ref)
    ref = ref.split("#")[0].strip()
    return ref or obsidian_ref


def _aggregate(log_path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"good": 0, "bad": 0})
    if not log_path.exists():
        return counts
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ref = _normalize_path(str(entry.get("obsidian_ref") or entry.get("doc_id") or ""))
            if not ref:
                continue
            rating = entry.get("rating", "")
            if rating == "good":
                counts[ref]["good"] += 1
            elif rating == "bad":
                counts[ref]["bad"] += 1
    return counts


def _load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_ledger(path: Path, rules: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _existing_rev_ids(rules: list[dict]) -> set[str]:
    return {str(r.get("rev_id", "")) for r in rules}


def main() -> None:
    counts = _aggregate(_FEEDBACK_LOG)

    penalty_candidates: list[tuple[str, dict[str, int]]] = []
    boost_candidates: list[tuple[str, dict[str, int]]] = []

    for ref, c in counts.items():
        total = c["good"] + c["bad"]
        if total < _MIN_COUNT:
            continue
        bad_rate = c["bad"] / total
        good_rate = c["good"] / total
        if bad_rate >= _BAD_RATE_THRESHOLD:
            penalty_candidates.append((ref, c))
        elif good_rate >= _GOOD_RATE_THRESHOLD:
            boost_candidates.append((ref, c))

    if not penalty_candidates and not boost_candidates:
        print("[analyze_rag_feedback] 閾値を超えた候補なし。変更なし。")
        return

    rules = _load_ledger(_LEDGER_PATH)
    existing_ids = _existing_rev_ids(rules)
    now_iso = datetime.now(timezone.utc).isoformat()
    added = 0

    for ref, c in penalty_candidates:
        total = c["good"] + c["bad"]
        rev_id = f"RAG-PENALTY-{re.sub(r'[^A-Za-z0-9]', '_', ref)[:40]}"
        if rev_id in existing_ids:
            print(f"[analyze_rag_feedback] スキップ（既存）: {rev_id}")
            continue
        rule = {
            "rev_id": rev_id,
            "type": "rag_boost_adjust",
            "pending_review": True,
            "description": f"RAGフィードバック: {ref} のペナルティ値を引き上げ候補（bad率 {c['bad']}/{total}={c['bad']/total:.0%}）",
            "source": "analyze_rag_feedback.py",
            "target": "config/rag_ranking.json",
            "path": ref,
            "action": "penalty",
            "good_count": c["good"],
            "bad_count": c["bad"],
            "bad_rate": round(c["bad"] / total, 4),
            "generated_at": now_iso,
        }
        rules.append(rule)
        existing_ids.add(rev_id)
        added += 1
        print(f"[analyze_rag_feedback] ペナルティ候補追加: {rev_id}")

    for ref, c in boost_candidates:
        total = c["good"] + c["bad"]
        rev_id = f"RAG-BOOST-{re.sub(r'[^A-Za-z0-9]', '_', ref)[:40]}"
        if rev_id in existing_ids:
            print(f"[analyze_rag_feedback] スキップ（既存）: {rev_id}")
            continue
        rule = {
            "rev_id": rev_id,
            "type": "rag_boost_adjust",
            "pending_review": True,
            "description": f"RAGフィードバック: {ref} のブースト値を引き上げ候補（good率 {c['good']}/{total}={c['good']/total:.0%}）",
            "source": "analyze_rag_feedback.py",
            "target": "config/rag_ranking.json",
            "path": ref,
            "action": "boost",
            "good_count": c["good"],
            "bad_count": c["bad"],
            "good_rate": round(c["good"] / total, 4),
            "generated_at": now_iso,
        }
        rules.append(rule)
        existing_ids.add(rev_id)
        added += 1
        print(f"[analyze_rag_feedback] ブースト候補追加: {rev_id}")

    if added > 0:
        _save_ledger(_LEDGER_PATH, rules)
        print(f"[analyze_rag_feedback] {added} 件を ledger_rules.json に追記しました。")
    else:
        print("[analyze_rag_feedback] 新規候補なし。変更なし。")


if __name__ == "__main__":
    main()
    sys.exit(0)
