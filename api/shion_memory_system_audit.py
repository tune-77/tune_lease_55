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
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INDEX_PATH = _REPO_ROOT / "data" / "shion_memory_index.json"
_REVISIONS_PATH = _REPO_ROOT / "data" / "shion_memory_revisions.jsonl"
_MEMORY_EFFECT_PATH = _REPO_ROOT / "reports" / "shion_memory_effect_latest.json"
_MEMORY_ENGINEERING_PATH = _REPO_ROOT / "reports" / "memory_engineering_latest.json"
_MEMORY_CONTRADICTIONS_PATH = _REPO_ROOT / "reports" / "shion_memory_contradictions_latest.json"
_PERSISTENT_MEMORY_AUDIT_PATH = _REPO_ROOT / "reports" / "persistent_memory_audit_latest.json"
_OBSIDIAN_MEMORY_EFFECTIVENESS_PATH = _REPO_ROOT / "reports" / "obsidian_memory_effectiveness_latest.json"


def _load_index() -> dict[str, Any] | None:
    try:
        data = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _records(index: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in index.get("records") or [] if isinstance(r, dict)]


def _clamp_limit(limit: int | None, *, default: int, maximum: int) -> int:
    try:
        raw = int(limit if limit is not None else default)
    except (TypeError, ValueError):
        raw = default
    return max(1, min(maximum, raw))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _report_source(path: Path, payload: dict[str, Any]) -> dict[str, str]:
    try:
        rel = str(path.relative_to(_REPO_ROOT))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "status": "loaded" if payload else "missing_or_invalid",
        "generated_at": str(payload.get("generated_at") or ""),
    }


def _memory_metadata_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    long_term = [r for r in records if str(r.get("memory_layer") or "") == "long_term"]
    active = [r for r in long_term if str(r.get("status") or "active") != "deprecated"]
    domain_count = sum(1 for r in active if str(r.get("domain") or "").strip())
    use_when_count = sum(1 for r in active if str(r.get("use_when") or "").strip())
    return {
        "long_term_active_records": len(active),
        "domain_count": domain_count,
        "domain_coverage": _ratio(domain_count, len(active)),
        "use_when_count": use_when_count,
        "use_when_coverage": _ratio(use_when_count, len(active)),
    }


def _add_signal(
    signals: list[dict[str, Any]],
    *,
    area: str,
    level: str,
    metric: str,
    reason: str,
    next_action: str,
) -> None:
    signals.append(
        {
            "area": area,
            "level": level,
            "metric": metric,
            "reason": reason,
            "next_action": next_action,
        }
    )


def _memory_effect_feedback_batches(effect: dict[str, Any], *, limit: int) -> dict[str, Any]:
    existing = effect.get("needs_feedback_triage")
    if isinstance(existing, dict) and isinstance(existing.get("top_batches"), list):
        batches = [batch for batch in existing.get("top_batches") or [] if isinstance(batch, dict)]
        return {
            **existing,
            "top_batches": batches[:max(1, min(limit, 20))],
            "source": "reports/shion_memory_effect_latest.json",
        }
    rows = effect.get("needs_feedback") if isinstance(effect.get("needs_feedback"), list) else []
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        domain = str(row.get("domain") or "").strip()
        if not domain:
            source = str(row.get("source_path") or "").strip()
            domain = Path(source).stem if source else "unknown"
        layer = str(row.get("memory_layer") or "unknown")
        mtype = str(row.get("memory_type") or "unknown")
        key = (domain, layer, mtype)
        group = groups.setdefault(
            key,
            {
                "domain": domain,
                "memory_layer": layer,
                "memory_type": mtype,
                "count": 0,
                "used_count": 0,
                "impact_hint_count": 0,
                "sample_ids": [],
                "samples": [],
            },
        )
        group["count"] += 1
        group["used_count"] += int(row.get("used_count") or 0)
        group["impact_hint_count"] += int(row.get("impact_hint_count") or 0)
        if len(group["samples"]) < 3:
            rid = str(row.get("id") or "")
            group["sample_ids"].append(rid)
            group["samples"].append(
                {
                    "id": rid,
                    "used_count": int(row.get("used_count") or 0),
                    "utility_state": str(row.get("utility_state") or ""),
                    "reason": str(row.get("reason") or ""),
                    "content": str(row.get("content") or "")[:140],
                }
            )
    batches = sorted(groups.values(), key=lambda item: (-int(item["used_count"]), -int(item["count"]), str(item["domain"])))
    return {
        "source": "reports/shion_memory_effect_latest.json",
        "total_sampled_needs_feedback": len(rows),
        "batch_count": len(batches),
        "top_batches": batches[:max(1, min(limit, 20))],
        "policy": "group_needs_feedback_by_domain_layer_type_no_auto_feedback",
    }


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


def run_shion_memory_sentinel(limit: int = 5) -> dict[str, Any]:
    """記憶監視系レポートを統合し、紫苑の記憶システム運用リスクを一括で見る。

    統合対象は、記憶索引メタデータ、実回答での利用実績、Memory Engineering、
    矛盾候補、永続記憶監査、Obsidian Memory Effectiveness。読み取り専用で、
    記憶昇格・削除・RAG順位変更・プロンプト変更は行わない。
    """
    max_items = _clamp_limit(limit, default=5, maximum=20)
    index = _load_index() or {}
    records = _records(index)
    effect = _load_json(_MEMORY_EFFECT_PATH)
    engineering = _load_json(_MEMORY_ENGINEERING_PATH)
    contradictions = _load_json(_MEMORY_CONTRADICTIONS_PATH)
    persistent = _load_json(_PERSISTENT_MEMORY_AUDIT_PATH)
    obsidian_effect = _load_json(_OBSIDIAN_MEMORY_EFFECTIVENESS_PATH)

    metadata = _memory_metadata_coverage(records)
    effect_summary = effect.get("summary") if isinstance(effect.get("summary"), dict) else {}
    engineering_summary = engineering.get("summary") if isinstance(engineering.get("summary"), dict) else {}
    persistent_summary = persistent.get("summary") if isinstance(persistent.get("summary"), dict) else {}
    obsidian_summary = obsidian_effect.get("summary") if isinstance(obsidian_effect.get("summary"), dict) else {}
    contradiction_candidates = contradictions.get("candidates") if isinstance(contradictions.get("candidates"), list) else []
    feedback_batches = _memory_effect_feedback_batches(effect, limit=max_items)

    summary = {
        "memory_records": len(records),
        "long_term_domain_coverage": metadata["domain_coverage"],
        "long_term_use_when_coverage": metadata["use_when_coverage"],
        "usage_events": int(effect_summary.get("usage_events") or 0),
        "used_memory_ids": int(effect_summary.get("used_memory_ids") or 0),
        "likely_helpful_memory_ids": int(effect_summary.get("likely_helpful_memory_ids") or 0),
        "needs_feedback_memory_ids": int(effect_summary.get("needs_feedback_memory_ids") or 0),
        "possible_noise_memory_ids": int(effect_summary.get("possible_noise_memory_ids") or 0),
        "open_human_review_records": int(engineering_summary.get("open_human_review_records") or 0),
        "open_human_review_batches": int(engineering_summary.get("open_human_review_batches") or 0),
        "candidate_to_active_pressure": float(engineering_summary.get("candidate_to_active_pressure") or 0.0),
        "write_policy_metadata_completion_rate": float(
            engineering_summary.get("write_policy_metadata_completion_rate") or 0.0
        ),
        "contradiction_candidates": len(contradiction_candidates),
        "persistent_findings": int(persistent_summary.get("findings") or 0),
        "persistent_high_findings": int(persistent_summary.get("high") or 0),
        "obsidian_used": int(obsidian_summary.get("used") or 0),
        "obsidian_validated": int(obsidian_summary.get("validated") or 0),
    }

    signals: list[dict[str, Any]] = []
    if not index:
        _add_signal(
            signals,
            area="memory_index",
            level="action_required",
            metric="missing_index",
            reason="data/shion_memory_index.json を読めない",
            next_action="build_shion_memory_index.py を先に復旧する",
        )
    if not effect:
        _add_signal(
            signals,
            area="usage_effect",
            level="watch",
            metric="missing_effect_report",
            reason="実回答で想起記憶が役に立ったかの最新レポートがない",
            next_action="build_shion_memory_effect_report.py を実行する",
        )
    if metadata["domain_coverage"] < 1.0 or metadata["use_when_coverage"] < 1.0:
        _add_signal(
            signals,
            area="memory_metadata",
            level="watch",
            metric=(
                f"domain={metadata['domain_coverage']}, "
                f"use_when={metadata['use_when_coverage']}"
            ),
            reason="長期記憶の domain/use_when 付与が未完了",
            next_action="build_shion_memory_index.py の長期記憶メタデータ推定を確認する",
        )
    if summary["possible_noise_memory_ids"] > 0:
        _add_signal(
            signals,
            area="usage_effect",
            level="action_required",
            metric=f"possible_noise={summary['possible_noise_memory_ids']}",
            reason="実回答で邪魔になった可能性がある記憶がある",
            next_action="shion_memory_effect_latest.md の Possible Noise を人間レビューする",
        )
    if summary["needs_feedback_memory_ids"] > 0:
        _add_signal(
            signals,
            area="usage_effect",
            level="watch",
            metric=f"needs_feedback={summary['needs_feedback_memory_ids']}",
            reason="想起はされたが、回答で本当に効いたか未確認の記憶が残っている",
            next_action="チャットUIの 効いた/微妙/違う フィードバックを優先的に集める",
        )
    if summary["contradiction_candidates"] > 0:
        level = "action_required" if summary["contradiction_candidates"] >= 3 else "watch"
        _add_signal(
            signals,
            area="contradictions",
            level=level,
            metric=f"candidates={summary['contradiction_candidates']}",
            reason="記憶内に矛盾候補がある",
            next_action="shion_memory_contradictions_latest.md を確認し、必要なら改訂宣言を作る",
        )
    if summary["open_human_review_records"] >= 100:
        _add_signal(
            signals,
            area="memory_engineering",
            level="watch",
            metric=(
                f"open_reviews={summary['open_human_review_records']}, "
                f"batches={summary['open_human_review_batches']}"
            ),
            reason="候補記憶・判断資産の人間レビュー待ちが多い",
            next_action="review inbox を同種テーマで束ね、承認/保留/却下を分ける",
        )
    if summary["write_policy_metadata_completion_rate"] < 0.8:
        _add_signal(
            signals,
            area="memory_engineering",
            level="watch",
            metric=f"write_policy_metadata={summary['write_policy_metadata_completion_rate']}",
            reason="候補記憶の importance/confidence/trust/provenance が薄い",
            next_action="新規候補生成時に write policy metadata を必須化する",
        )
    if summary["persistent_high_findings"] > 0:
        _add_signal(
            signals,
            area="persistent_memory",
            level="action_required",
            metric=f"high={summary['persistent_high_findings']}",
            reason="永続記憶に高優先度の監査所見がある",
            next_action="persistent_memory_audit_latest.md の high findings を先に直す",
        )
    elif summary["persistent_findings"] > 0:
        _add_signal(
            signals,
            area="persistent_memory",
            level="watch",
            metric=f"findings={summary['persistent_findings']}",
            reason="永続記憶に軽微な監査所見がある",
            next_action="永続記憶の人格・運用原則・安全境界以外を整理する",
        )

    status = "ok"
    if any(s["level"] == "action_required" for s in signals):
        status = "action_required"
    elif signals:
        status = "watch"

    return {
        "mode": "shion_memory_sentinel",
        "agent": "Shion Memory Sentinel",
        "status": status,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "guardrail": "read_only_no_memory_write_no_prompt_no_rag_rank_no_scoring_no_auto_promotion",
        "source_reports": [
            _report_source(_INDEX_PATH, index),
            _report_source(_MEMORY_EFFECT_PATH, effect),
            _report_source(_MEMORY_ENGINEERING_PATH, engineering),
            _report_source(_MEMORY_CONTRADICTIONS_PATH, contradictions),
            _report_source(_PERSISTENT_MEMORY_AUDIT_PATH, persistent),
            _report_source(_OBSIDIAN_MEMORY_EFFECTIVENESS_PATH, obsidian_effect),
        ],
        "summary": summary,
        "metadata_coverage": metadata,
        "feedback_triage": feedback_batches,
        "signals": signals[:max_items],
        "signal_count": len(signals),
        "next_actions": [str(s["next_action"]) for s in signals[:max_items]],
    }


SHION_MEMORY_SYSTEM_AUDIT_TOOLS = [
    run_shion_memory_system_audit,
    run_shion_memory_sentinel,
    audit_memory_index_orphans,
    audit_memory_freshness_pipeline,
    audit_memory_revision_integrity,
    audit_memory_recall_eval_health,
]
