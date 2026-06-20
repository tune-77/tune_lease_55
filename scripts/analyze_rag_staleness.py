#!/usr/bin/env python3
"""RAG ノードの鮮度を分析して、長期アクセスなし文書を改善台帳に追記するスクリプト。

ソース:
  data/rag_feedback_log.jsonl  — ユーザー評価ログ（フィードバックがあった文書）
  data/rag_hit_log.jsonl       — 検索ヒットログ（REV-114 で生成、任意）
  api/chroma_db/               — ChromaDB（全既知 obsidian_ref とメタデータを取得）

30日以上アクセスがない obsidian_ref を分類する:
  - stale              : アクセスなし かつ 重要度低い → 陳腐化候補
  - important_but_unused: アクセスなし だが 重要度高い → 削除候補ではなく要確認
  - active             : 直近30日内にアクセスあり

出力: reports/stale_rag_nodes_YYYYMMDD.json
台帳: api/rule_engine/ledger_rules.json（stale/important_but_unused それぞれ追記）
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_LOG = PROJECT_ROOT / "data" / "rag_feedback_log.jsonl"
HIT_LOG = PROJECT_ROOT / "data" / "rag_hit_log.jsonl"
LEDGER_FILE = PROJECT_ROOT / "api" / "rule_engine" / "ledger_rules.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

STALE_DAYS = 30

# 重要度が高いと判断するキーワード（タイトル・tags に含まれる場合）
_IMPORTANCE_KEYWORDS = frozenset([
    "法定", "規定", "必須", "基本", "重要", "コア", "必要", "標準",
    "ガイドライン", "規則", "規約", "要件", "定義", "マスタ", "基幹",
    "important", "core", "required", "mandatory", "fundamental",
])


def _parse_ts(ts_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _is_important_node(meta: dict) -> bool:
    """ChromaDB メタデータから重要度を判定する。"""
    # 明示的な importance フィールド
    importance = str(meta.get("importance") or "").lower()
    if importance in ("high", "critical", "必須", "高"):
        return True

    # tags フィールド（カンマ区切りや JSON 文字列のことも）
    tags_raw = str(meta.get("tags") or "")
    tags_lower = tags_raw.lower()
    if any(kw in tags_lower for kw in _IMPORTANCE_KEYWORDS):
        return True

    # title フィールド
    title = str(meta.get("title") or meta.get("file_name") or "").lower()
    if any(kw in title for kw in _IMPORTANCE_KEYWORDS):
        return True

    # source_path / obsidian_ref のパスに手がかりがある場合
    path = str(meta.get("source_path") or meta.get("obsidian_ref") or "").lower()
    for kw in ("core", "master", "required", "law", "legal", "regulation", "基本", "法定"):
        if kw in path:
            return True

    return False


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


def collect_refs_and_meta_from_chroma() -> dict[str, dict]:
    """ChromaDB から全 obsidian_ref とメタデータを取得する。ref → meta のマップを返す。"""
    ref_meta: dict[str, dict] = {}
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from api.knowledge.vector_store import KnowledgeVectorStore
        store = KnowledgeVectorStore()
        result = store._col.get(include=["metadatas"])
        for meta in result.get("metadatas") or []:
            if not isinstance(meta, dict):
                continue
            ref = str(meta.get("obsidian_ref") or "").strip()
            if ref and ref not in ref_meta:
                ref_meta[ref] = meta
    except Exception:
        pass
    return ref_meta


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


def already_exists(ledger: list[dict], ref: str, category: str) -> bool:
    snippet = ref[:40]
    for entry in ledger:
        if snippet in str(entry.get("description") or "") and entry.get("category") == category:
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

    # ChromaDB から全 ref とメタデータを取得
    chroma_ref_meta = collect_refs_and_meta_from_chroma()
    all_chroma_refs = set(chroma_ref_meta.keys())

    # 各 ref を分類
    nodes: list[dict] = []

    # アクセス記録がある ref → active か stale
    for ref, ts in last_access.items():
        meta = chroma_ref_meta.get(ref, {})
        important = _is_important_node(meta)
        if ts >= cutoff:
            node_category = "active"
        elif important:
            node_category = "important_but_unused"
        else:
            node_category = "stale"
        nodes.append({
            "obsidian_ref": ref,
            "last_accessed_at": ts.isoformat(),
            "days_since_access": (now - ts).days,
            "reason": "long_inactive" if node_category != "active" else "recent_access",
            "category": node_category,
            "is_important": important,
        })

    # ChromaDB にあるが一度もアクセスされていない ref
    never_accessed = all_chroma_refs - set(last_access.keys())
    for ref in sorted(never_accessed):
        meta = chroma_ref_meta.get(ref, {})
        important = _is_important_node(meta)
        node_category = "important_but_unused" if important else "stale"
        nodes.append({
            "obsidian_ref": ref,
            "last_accessed_at": None,
            "days_since_access": None,
            "reason": "never_accessed",
            "category": node_category,
            "is_important": important,
        })

    # サマリー集計
    stale_nodes = [n for n in nodes if n["category"] == "stale"]
    important_nodes = [n for n in nodes if n["category"] == "important_but_unused"]
    active_nodes = [n for n in nodes if n["category"] == "active"]

    stale_nodes.sort(key=lambda x: x.get("days_since_access") or 9999, reverse=True)
    important_nodes.sort(key=lambda x: x.get("days_since_access") or 9999, reverse=True)

    report = {
        "generated_at": now.isoformat(),
        "stale_threshold_days": STALE_DAYS,
        "total_known_refs": len(nodes),
        "stale_count": len(stale_nodes),
        "important_but_unused_count": len(important_nodes),
        "active_count": len(active_nodes),
        "stale_nodes": stale_nodes,
        "important_but_unused_nodes": important_nodes,
    }

    output_path = REPORTS_DIR / f"stale_rag_nodes_{today_str}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[analyze_rag_staleness] stale={len(stale_nodes)} important_but_unused={len(important_nodes)} active={len(active_nodes)} → {output_path}",
        flush=True,
    )

    if not stale_nodes and not important_nodes:
        return

    ledger = load_ledger()
    base_rev = max_rev_number(ledger)
    now_iso = now.isoformat()
    added = 0

    # stale: 削除・更新候補（上位5件）
    for node in stale_nodes[:5]:
        ref = node["obsidian_ref"]
        if already_exists(ledger, ref, "rag_staleness"):
            continue
        base_rev += 1
        rev_id = f"REV-{base_rev:03d}r"
        days = node.get("days_since_access")
        days_str = f"{days}日間" if days is not None else "未アクセス"
        desc = f"[RAG陳腐化] {days_str}アクセスなし: {ref[:60]}"
        ledger.append({
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
        })
        added += 1
        print(f"[analyze_rag_staleness] stale追記: {rev_id} — {desc}", flush=True)

    # important_but_unused: 要確認（削除禁止、上位5件）
    for node in important_nodes[:5]:
        ref = node["obsidian_ref"]
        if already_exists(ledger, ref, "rag_important_unused"):
            continue
        base_rev += 1
        rev_id = f"REV-{base_rev:03d}r"
        days = node.get("days_since_access")
        days_str = f"{days}日間" if days is not None else "未アクセス"
        desc = (
            f"[RAG重要ノード未使用] {days_str}アクセスなし（重要度高・削除不可）: {ref[:55]}"
        )
        ledger.append({
            "rev_id": rev_id,
            "type": "manual",
            "pending_review": True,
            "category": "rag_important_unused",
            "description": desc,
            "status": "pending_review",
            "source": "analyze_rag_staleness",
            "detected_at": now_iso,
            "obsidian_ref": ref,
            "days_since_access": days,
            "affected_files": [],
            "risk": "low",
            "auto_fix_allowed": False,
            "note": "重要度高のため削除候補ではなく内容確認・再インデックスを推奨",
        })
        added += 1
        print(f"[analyze_rag_staleness] important_but_unused追記: {rev_id} — {desc}", flush=True)

    if added > 0:
        save_ledger(ledger)
        print(f"[analyze_rag_staleness] {added} 件を台帳に追記しました。", flush=True)
    else:
        print("[analyze_rag_staleness] 新規候補なし（既存エントリと重複）。", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
