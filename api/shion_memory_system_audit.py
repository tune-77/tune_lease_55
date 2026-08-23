"""紫苑の記憶システム整合性監査（読み取り専用・google.adk非依存）。

docs/shion_memory_architecture.md が定義する記憶索引（data/shion_memory_index.json）
と、それを生成するソース（MEMORY.md/memory/*.md/mind.json）、鮮度バッチ
（api/shion_memory_decay.py）、改訂履歴（scripts/revise_shion_memory.py）の間で
断線がないかを検出する。

api/shion_system_self_inspection.py と同じ方針: google.adk へ依存せず import
できるようにし、書き込みは一切行わない。api/shion_agent_tools.py 経由で
紫苑 ADK エージェントのツールとしても登録し、紫苑自身が「自分の記憶は健全か」を
調査できるようにする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = _REPO_ROOT / "data" / "shion_memory_index.json"
_REVISIONS_PATH = _REPO_ROOT / "data" / "shion_memory_revisions.jsonl"


def _load_index() -> dict[str, Any] | None:
    try:
        data = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _records(index: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in index.get("records") or [] if isinstance(r, dict)]


def _clamp_limit(limit: int | None, *, default: int, maximum: int) -> int:
    try:
        raw = int(limit if limit is not None else default)
    except (TypeError, ValueError):
        raw = default
    return max(1, min(maximum, raw))


def audit_memory_index_orphans(limit: int = 10) -> dict[str, Any]:
    """記憶索引の各レコードが指すsource_pathが今も実ファイルとして存在するか確認する。

    Args:
        limit: 孤立レコード（source_pathが消えたレコード）のサンプル表示件数上限（1..50）。

    Returns:
        total_records, orphan_count, orphan_samples、status/memory_type別件数を含む辞書。
        索引ファイルが無い場合は status="index_not_found" のみを返す。
    """
    max_items = _clamp_limit(limit, default=10, maximum=50)
    index = _load_index()
    if index is None:
        return {
            "mode": "memory_index_orphans",
            "status": "index_not_found",
            "index_path": str(_INDEX_PATH.relative_to(_REPO_ROOT)),
        }

    records = _records(index)
    orphans: list[dict[str, str]] = []
    status_breakdown: dict[str, int] = {}
    type_breakdown: dict[str, int] = {}
    for rec in records:
        status = str(rec.get("status") or "active")
        status_breakdown[status] = status_breakdown.get(status, 0) + 1
        mtype = str(rec.get("memory_type") or "unknown")
        type_breakdown[mtype] = type_breakdown.get(mtype, 0) + 1

        source_path = str(rec.get("source_path") or "")
        if not source_path:
            continue
        if not (_REPO_ROOT / source_path).exists():
            orphans.append(
                {
                    "id": str(rec.get("id") or ""),
                    "source_path": source_path,
                    "content_preview": str(rec.get("content") or "")[:80],
                }
            )

    return {
        "mode": "memory_index_orphans",
        "status": "ok",
        "total_records": len(records),
        "orphan_count": len(orphans),
        "orphan_samples": orphans[:max_items],
        "status_breakdown": status_breakdown,
        "memory_type_breakdown": type_breakdown,
        "guardrail": "read_only_no_index_rewrite",
    }


def audit_memory_freshness_pipeline() -> dict[str, Any]:
    """鮮度バッチ（api/shion_memory_decay.py）が索引と同期して動いているか確認する。

    最新の data/shion_memory_freshness.jsonl スナップショットの件数と、
    索引の現在の非deprecatedレコード数を突き合わせる（減衰バッチは
    deprecatedレコードをスキップするため、その差分は除いて比較する）。

    Returns:
        index_total, snapshot_total, record_count_drift, at_risk_count を含む辞書。
        索引もスナップショットも無ければ status="no_data" を返す。
    """
    from api.shion_memory_decay import get_latest_freshness_snapshot

    index = _load_index()
    snapshot = get_latest_freshness_snapshot()

    result: dict[str, Any] = {"mode": "memory_freshness_pipeline"}

    if index is None:
        result["index_status"] = "index_not_found"
        index_total = 0
        deprecated_count = 0
    else:
        records = _records(index)
        index_total = len(records)
        deprecated_count = sum(1 for r in records if str(r.get("status") or "") == "deprecated")
        result["index_status"] = "ok"
    result["index_total"] = index_total
    result["index_deprecated_count"] = deprecated_count

    if snapshot is None:
        result["snapshot_status"] = "no_snapshot_yet"
        result["status"] = "batch_never_ran" if index_total else "no_data"
        return result

    result["snapshot_status"] = "ok"
    result["snapshot_at"] = snapshot.get("snapshot_at")
    result["snapshot_total"] = snapshot.get("total")
    result["at_risk_count"] = snapshot.get("at_risk_count")

    expected_snapshot_total = index_total - deprecated_count
    snapshot_total = snapshot.get("total")
    drift = (
        expected_snapshot_total - int(snapshot_total)
        if index is not None and isinstance(snapshot_total, (int, float))
        else None
    )
    result["record_count_drift"] = drift
    result["status"] = "in_sync" if drift in (None, 0) else "drifted"
    return result


def audit_memory_revision_integrity(limit: int = 10) -> dict[str, Any]:
    """改訂宣言（scripts/revise_shion_memory.py）が索引へ適用済みか確認する。

    data/shion_memory_revisions.jsonl の各宣言を、コピーした索引へ再適用
    （apply_revisions は冪等）し、実際の索引と差分が出るレコードを
    「未適用の改訂」として検出する。索引を書き換えることはしない。

    Args:
        limit: 未適用改訂のサンプル表示件数上限（1..50）。

    Returns:
        total_revisions, pending_count, pending_samples を含む辞書。
    """
    import copy

    from scripts.revise_shion_memory import apply_revisions, load_revisions

    max_items = _clamp_limit(limit, default=10, maximum=50)
    index = _load_index()
    revisions = load_revisions(_REVISIONS_PATH)

    if index is None:
        return {
            "mode": "memory_revision_integrity",
            "status": "index_not_found",
            "total_revisions": len(revisions),
        }
    if not revisions:
        return {
            "mode": "memory_revision_integrity",
            "status": "ok",
            "total_revisions": 0,
            "pending_count": 0,
            "pending_samples": [],
        }

    before = copy.deepcopy(index)
    working = copy.deepcopy(index)
    apply_summary = apply_revisions(working, revisions)

    before_by_id = {str(r.get("id") or ""): r for r in _records(before)}
    pending: list[dict[str, str]] = []
    for rec in _records(working):
        rid = str(rec.get("id") or "")
        prior = before_by_id.get(rid)
        if prior is None:
            # 改訂適用で新規に生まれた後継レコード（索引再生成前は存在しない）
            pending.append({"id": rid, "reason": "successor_not_yet_in_index"})
            continue
        if str(prior.get("status") or "") != str(rec.get("status") or ""):
            pending.append(
                {
                    "id": rid,
                    "reason": f"status {prior.get('status')} -> {rec.get('status')} not applied",
                }
            )
        elif list(prior.get("supersedes") or []) != list(rec.get("supersedes") or []):
            pending.append({"id": rid, "reason": "supersedes not applied"})

    return {
        "mode": "memory_revision_integrity",
        "status": "ok" if not pending else "pending_index_rebuild",
        "total_revisions": len(revisions),
        "reapply_summary": apply_summary,
        "pending_count": len(pending),
        "pending_samples": pending[:max_items],
        "guardrail": "read_only_dry_run_reapply_no_index_rewrite",
    }


def audit_memory_recall_eval_health() -> dict[str, Any]:
    """想起精度の評価セット（回帰ゲート）が使える状態か確認する。

    api/knowledge/shion_recall_eval_set.json の件数と、
    tests/test_shion_recall_eval.py が参照する評価ハーネスの存在だけを見る。
    LLM呼び出しや実際の評価実行は行わない。

    Returns:
        eval_set_path, eval_case_count, harness_script_found を含む辞書。
    """
    eval_set_path = _REPO_ROOT / "api" / "knowledge" / "shion_recall_eval_set.json"
    harness_path = _REPO_ROOT / "scripts" / "eval_shion_memory_recall.py"

    result: dict[str, Any] = {
        "mode": "memory_recall_eval_health",
        "eval_set_path": str(eval_set_path.relative_to(_REPO_ROOT)),
        "harness_script_found": harness_path.exists(),
    }
    try:
        data = json.loads(eval_set_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["status"] = "eval_set_not_found"
        result["eval_case_count"] = 0
        return result

    cases = data.get("cases") if isinstance(data, dict) else data
    case_count = len(cases) if isinstance(cases, list) else 0
    result["eval_case_count"] = case_count
    result["status"] = "ok" if case_count > 0 and result["harness_script_found"] else "incomplete"
    return result


def run_shion_memory_system_audit() -> dict[str, Any]:
    """記憶システムの整合性チェックを一括で実行する（索引孤立/鮮度パイプライン/改訂適用/評価健全性）。

    紫苑が「自分の記憶は健全か」という質問に自律的に答えるための入口。
    個別の詳細を掘り下げたい場合は audit_memory_index_orphans /
    audit_memory_freshness_pipeline / audit_memory_revision_integrity /
    audit_memory_recall_eval_health を個別に呼ぶ。

    Returns:
        4観点それぞれの結果と、issue_count（要対応と見なせる件数の合計）を含む辞書。
    """
    orphans = audit_memory_index_orphans()
    freshness = audit_memory_freshness_pipeline()
    revisions = audit_memory_revision_integrity()
    eval_health = audit_memory_recall_eval_health()

    issue_count = (
        orphans.get("orphan_count", 0) or 0
    ) + (
        1 if freshness.get("status") == "drifted" else 0
    ) + (
        revisions.get("pending_count", 0) or 0
    ) + (
        1 if eval_health.get("status") not in {"ok"} else 0
    )

    return {
        "mode": "shion_memory_system_audit",
        "index_orphans": orphans,
        "freshness_pipeline": freshness,
        "revision_integrity": revisions,
        "recall_eval_health": eval_health,
        "issue_count": issue_count,
        "guardrail": "read_only_no_index_rewrite_no_llm_calls",
    }


SHION_MEMORY_SYSTEM_AUDIT_TOOLS = [
    run_shion_memory_system_audit,
    audit_memory_index_orphans,
    audit_memory_freshness_pipeline,
    audit_memory_revision_integrity,
    audit_memory_recall_eval_health,
]
