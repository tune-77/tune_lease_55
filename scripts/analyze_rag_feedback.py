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
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_FEEDBACK_LOG = _REPO_ROOT / "data" / "rag_feedback_log.jsonl"
_SEARCH_LOG = _REPO_ROOT / "data" / "rag_search_log.jsonl"
_LEDGER_PATH = _REPO_ROOT / "api" / "rule_engine" / "ledger_rules.json"

_MIN_COUNT = 5
_BAD_RATE_THRESHOLD = 0.60
_GOOD_RATE_THRESHOLD = 0.70
_UNRATED_HIT_DAYS = 30


def _normalize_path(obsidian_ref: str) -> str:
    """[[path#section]] → path（ファイルレベルで集計）"""
    ref = re.sub(r"^\[\[", "", obsidian_ref)
    ref = re.sub(r"\]\]$", "", ref)
    ref = ref.split("#")[0].strip()
    return ref or obsidian_ref


def _load_recent_search_refs(lookback_days: int = _UNRATED_HIT_DAYS) -> set[str]:
    """rag_search_log.jsonl から直近 lookback_days 日以内にヒットした obsidian_ref の集合を返す。"""
    refs: set[str] = set()
    if not _SEARCH_LOG.exists():
        return refs
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    try:
        lines = _SEARCH_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return refs
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts_str = str(entry.get("ts") or "")
        try:
            ts = datetime.fromisoformat(ts_str).astimezone(timezone.utc)
        except Exception:
            continue
        if ts < cutoff:
            continue
        for r in entry.get("results") or []:
            ref = _normalize_path(str(r.get("obsidian_ref") or r.get("doc_id") or ""))
            if ref:
                refs.add(ref)
    return refs


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
    recent_search_refs = _load_recent_search_refs()
    rated_refs = set(counts.keys())

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

    # ヒット済みだが直近評価なし（rag_search_log にあるが feedback log にない）
    unrated_hit_refs = recent_search_refs - rated_refs

    if not penalty_candidates and not boost_candidates and not unrated_hit_refs:
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
            "penalty_strength": "strong",
            "good_count": c["good"],
            "bad_count": c["bad"],
            "bad_rate": round(c["bad"] / total, 4),
            "generated_at": now_iso,
        }
        rules.append(rule)
        existing_ids.add(rev_id)
        added += 1
        print(f"[analyze_rag_feedback] ペナルティ候補追加(strong): {rev_id}")

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
            "penalty_strength": None,
            "good_count": c["good"],
            "bad_count": c["bad"],
            "good_rate": round(c["good"] / total, 4),
            "generated_at": now_iso,
        }
        rules.append(rule)
        existing_ids.add(rev_id)
        added += 1
        print(f"[analyze_rag_feedback] ブースト候補追加: {rev_id}")

    for ref in sorted(unrated_hit_refs)[:10]:
        rev_id = f"RAG-UNRATED-{re.sub(r'[^A-Za-z0-9]', '_', ref)[:40]}"
        if rev_id in existing_ids:
            continue
        rule = {
            "rev_id": rev_id,
            "type": "rag_boost_adjust",
            "pending_review": True,
            "description": (
                f"RAGヒット済み未評価: {ref} が直近{_UNRATED_HIT_DAYS}日で検索ヒットしたが評価なし（要内容確認）"
            ),
            "source": "analyze_rag_feedback.py",
            "target": "config/rag_ranking.json",
            "path": ref,
            "action": "review_content",
            "penalty_strength": "weak",
            "generated_at": now_iso,
        }
        rules.append(rule)
        existing_ids.add(rev_id)
        added += 1
        print(f"[analyze_rag_feedback] 未評価ヒット候補追加(weak): {rev_id}")

    if added > 0:
        _save_ledger(_LEDGER_PATH, rules)
        print(f"[analyze_rag_feedback] {added} 件を ledger_rules.json に追記しました。")
    else:
        print("[analyze_rag_feedback] 新規候補なし。変更なし。")


if __name__ == "__main__":
    main()
    sys.exit(0)
