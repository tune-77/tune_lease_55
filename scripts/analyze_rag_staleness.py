#!/usr/bin/env python3
"""RAG ノードの鮮度を分析して、長期アクセスなし文書を改善台帳に追記するスクリプト。

ソース:
  data/rag_feedback_log.jsonl  — ユーザー評価ログ（フィードバックがあった文書）
  data/rag_hit_log.jsonl       — 検索ヒットログ（REV-114 で生成、任意）
  api/chroma_db/               — ChromaDB（全既知 obsidian_ref を取得）

30日以上アクセスがない obsidian_ref を「陳腐化候補」とみなし、
reports/stale_rag_nodes_YYYYMMDD.json に出力して台帳に追記する。
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_LOG = PROJECT_ROOT / "data" / "rag_feedback_log.jsonl"
HIT_LOG = PROJECT_ROOT / "data" / "rag_hit_log.jsonl"
LEDGER_FILE = PROJECT_ROOT / "api" / "rule_engine" / "ledger_rules.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

STALE_DAYS = 30
MIN_HIT_COUNT_THRESHOLD = 1


def _parse_ts(ts_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts_str[:25], fmt[:len(ts_str[:25])])
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
    except Exception:
        return None


def collect_last_access_from_jsonl(log_path: Path) -> dict[str, datetime]:
    """jsonl ログから obsidian_ref ごとの最終アクセス日時を返す。"""
    last_access: dict[str, datetime] = {}
    if not log_path.exists():
        return last_access
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return last_access
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        ref = str(entry.get("obsidian_ref") or entry.get("ref") or "").strip()
        ts_str = str(entry.get("ts") or "").strip()
        if not ref or not ts_str:
            continue
        ts = _parse_ts(ts_str)
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        existing = last_access.get(ref)
        if existing is None or ts > existing:
            last_access[ref] = ts
    return last_access


def collect_all_refs_from_chroma() -> set[str]:
    """ChromaDB から全 obsidian_ref を取得（任意）。"""
    refs: set[str] = set()
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from api.knowledge.vector_store import KnowledgeVectorStore
        store = KnowledgeVectorStore()
        result = store._col.get(include=["metadatas"])
        for meta in result.get("metadatas") or []:
            ref = str(meta.get("obsidian_ref") or "").strip()
            if ref:
                refs.add(ref)
    except Exception:
        pass
    return refs


def load_ledger() -> list[dict]:
    if not LEDGER_FILE.exists():
        return []
    try:
        return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_ledger(rules: list[dict]) -> None:
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_FILE.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def already_exists(ledger: list[dict], ref: str) -> bool:
    snippet = ref[:40]
    for entry in ledger:
        if snippet in str(entry.get("description") or "") and entry.get("category") == "rag_staleness":
            return True
    return False


def max_rev_number(ledger: list[dict]) -> int:
    max_num = 0
    for entry in ledger:
        m = re.match(r"REV-(\d+)", str(entry.get("rev_id") or ""))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def main() -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=STALE_DAYS)
    today_str = date.today().strftime("%Y%m%d")

    # ログからアクセス記録を収集
    last_access: dict[str, datetime] = {}
    for src in (FEEDBACK_LOG, HIT_LOG):
        for ref, ts in collect_last_access_from_jsonl(src).items():
            existing = last_access.get(ref)
            if existing is None or ts > existing:
                last_access[ref] = ts

    # ChromaDB から全 ref を取得して「一度も評価なし」の文書も候補に加える
    all_chroma_refs = collect_all_refs_from_chroma()
    never_accessed = {ref for ref in all_chroma_refs if ref not in last_access}

    # 陳腐化判定
    stale_refs: list[dict] = []
    for ref, ts in last_access.items():
        if ts < cutoff:
            stale_refs.append(
                {
                    "obsidian_ref": ref,
                    "last_accessed_at": ts.isoformat(),
                    "days_since_access": (now - ts).days,
                    "reason": "long_inactive",
                }
            )
    for ref in sorted(never_accessed):
        stale_refs.append(
            {
                "obsidian_ref": ref,
                "last_accessed_at": None,
                "days_since_access": None,
                "reason": "never_accessed",
            }
        )

    stale_refs.sort(
        key=lambda x: x.get("days_since_access") or 9999,
        reverse=True,
    )

    report = {
        "generated_at": now.isoformat(),
        "stale_threshold_days": STALE_DAYS,
        "total_known_refs": len(last_access) + len(never_accessed),
        "stale_count": len(stale_refs),
        "stale_nodes": stale_refs,
    }

    output_path = REPORTS_DIR / f"stale_rag_nodes_{today_str}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[analyze_rag_staleness] 陳腐化候補: {len(stale_refs)} 件 → {output_path}",
        flush=True,
    )

    if not stale_refs:
        return

    # 上位5件のみ台帳に追記（多すぎる場合の台帳肥大化を防ぐ）
    ledger = load_ledger()
    base_rev = max_rev_number(ledger)
    now_iso = now.isoformat()
    added = 0

    for node in stale_refs[:5]:
        ref = node["obsidian_ref"]
        if already_exists(ledger, ref):
            continue
        base_rev += 1
        rev_id = f"REV-{base_rev:03d}r"
        days = node.get("days_since_access")
        days_str = f"{days}日間" if days is not None else "未アクセス"
        desc = f"[RAG陳腐化] {days_str}アクセスなし: {ref[:60]}"
        new_entry = {
            "rev_id": rev_id,
            "type": "manual",
            "pending_review": True,
            "category": "rag_staleness",
            "description": desc,
            "status": "pending_review",
            "source": "analyze_rag_staleness",
            "detected_at": now_iso,
            "obsidian_ref": ref,
            "days_since_access": days,
            "affected_files": [],
            "risk": "low",
            "auto_fix_allowed": False,
        }
        ledger.append(new_entry)
        added += 1
        print(f"[analyze_rag_staleness] 追記: {rev_id} — {desc}", flush=True)

    if added > 0:
        save_ledger(ledger)
        print(f"[analyze_rag_staleness] {added} 件を台帳に追記しました。", flush=True)
    else:
        print("[analyze_rag_staleness] 新規候補なし（既存エントリと重複）。", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
