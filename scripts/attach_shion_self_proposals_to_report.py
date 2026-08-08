#!/usr/bin/env python3
"""Attach Shion self proposals to daily improvement reports as a separate section."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_LATEST = REPORTS_DIR / "latest.json"
LOCAL_IMPROVEMENT_LOG = DATA_DIR / "cloudrun_improvement_log.jsonl"
REPO_LEDGER = PROJECT_ROOT / "scripts" / "improvement_ledger.jsonl"

LAYER_LABELS = {
    "usage_based": "利用ログ由来",
    "feedback_based": "人間反応・判断ログ由来",
    "system_audit_based": "システム監査由来",
    "domestic_mode_based": "内政モード由来",
    "reflection_based": "内省アクション由来",
}

DEFAULT_REVIEW_LIMIT = 3
DEFAULT_LEAP_LIMIT = 1
DEFAULT_REFLECTION_LIMIT = 1
FITNESS_WEIGHT = 0.35
REQUIRED_HYPOTHESIS_FIELDS = (
    "hypothesis",
    "evidence",
    "proposed_change",
    "success_metric",
    "verification_plan",
    "risk",
)
DOMESTIC_MODE_INPUT_FIELDS = (
    "importance",
    "usage_context",
    "intent",
    "decision_rule",
    "do_not_touch",
    "success_metric",
    "review_timing",
)
LAYER_QUALITY_WEIGHTS = {
    "feedback_based": 30,
    "system_audit_based": 24,
    "domestic_mode_based": 34,
    "usage_based": 18,
}
PRIORITY_QUALITY_WEIGHTS = {
    "high": 8,
    "medium": 5,
    "low": 2,
}
LEAP_MARKERS = (
    "跳躍",
    "実験",
    "探索",
    "違和感",
    "逆算",
    "弁護",
    "仮称",
    "思い切って",
    "あえて",
    "意外",
)
APPROVED_STATUSES = {"approved", "applied", "adopt", "adopted", "promoted"}
REJECTED_STATUSES = {"rejected", "reject", "deleted", "suppressed"}
DEFERRED_STATUSES = {"deferred", "parked", "hold", "held", "observe", "needs_human_review"}

SOURCES = [
    {
        "path": DATA_DIR / "usage_loop_proposals.jsonl",
        "source": "usage_loop",
        "kind": "画面利用",
        "evidence_layer": "usage_based",
        "summary_keys": ("hypothesis", "evidence", "reason"),
    },
    {
        "path": DATA_DIR / "feedback_pattern_proposals.jsonl",
        "source": "feedback_pattern_loop",
        "kind": "人間反応",
        "evidence_layer": "feedback_based",
        "summary_keys": ("hypothesis", "evidence", "pattern", "suggestion"),
    },
    {
        "path": DATA_DIR / "judgment_divergence_proposals.jsonl",
        "source": "judgment_divergence_loop",
        "kind": "審査判断乖離",
        "evidence_layer": "feedback_based",
        "summary_keys": ("observation", "review_point", "suggestion"),
    },
    {
        "path": DATA_DIR / "outcome_drift_proposals.jsonl",
        "source": "outcome_drift_loop",
        "kind": "実績ドリフト",
        "evidence_layer": "feedback_based",
        "summary_keys": ("observation", "review_point", "suggestion"),
    },
    {
        "path": DATA_DIR / "knowledge_gap_proposals.jsonl",
        "source": "knowledge_gap_loop",
        "kind": "ナレッジ穴探し",
        "evidence_layer": "feedback_based",
        "summary_keys": ("reason", "search_hint"),
    },
    {
        "path": DATA_DIR / "domestic_mode_proposals.jsonl",
        "source": "domestic_mode",
        "kind": "内政モード",
        "evidence_layer": "domestic_mode_based",
        "summary_keys": ("hypothesis", "decision_layer", "proposed_change", "success_metric"),
    },
    {
        "path": DATA_DIR / "reflection_action_candidates.jsonl",
        "source": "reflection_action_candidates",
        "kind": "内省アクション",
        "evidence_layer": "reflection_based",
        "summary_keys": ("hypothesis", "proposed_change", "success_metric", "risk"),
    },
    {
        "path": REPORTS_DIR / "lease_system_gap_analysis.json",
        "source": "lease_system_gap_analysis",
        "kind": "システム監査",
        "evidence_layer": "system_audit_based",
        "format": "system_gap_report",
        "summary_keys": ("impact", "recommended_action", "guardrail"),
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _load_system_gap_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    rows = payload.get("gaps") if isinstance(payload.get("gaps"), list) else []
    normalized: list[dict[str, Any]] = []
    generated_at = str(payload.get("generated_at") or "")
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "title": row.get("title") or row.get("id") or "",
                "topic": row.get("id") or row.get("category") or "",
                "status": "system_audit_candidate",
                "priority": row.get("priority") or "",
                "generated_at": generated_at,
                "hypothesis": row.get("impact") or "",
                "evidence": " / ".join(str(x) for x in (row.get("evidence") or []) if str(x).strip()),
                "proposed_change": row.get("recommended_action") or "",
                "success_metric": row.get("suggested_program") or "",
                "verification_plan": "システム監査レポートを再生成し、同一GAPが改善・解消したか確認する。",
                "risk": row.get("guardrail") or "",
                "proposal_schema": "shion_self_hypothesis_v1",
                "human_decision_status": "needs_human_review",
                "impact": row.get("impact") or "",
                "recommended_action": row.get("recommended_action") or "",
                "guardrail": row.get("guardrail") or "",
            }
        )
    return normalized


def _load_source_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = source["path"]
    if source.get("format") == "system_gap_report":
        return _load_system_gap_rows(path)
    return _load_jsonl(path)


def _summary(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        value = str(item.get(key) or "").strip()
        if value:
            parts.append(value)
    text = " / ".join(parts).strip()
    return text[:240].rstrip()


def _proposal_sort_key(item: dict[str, Any]) -> str:
    return str(item.get("generated_at") or item.get("ts") or "")


def _proposal_quality_score(item: dict[str, Any]) -> int:
    """Score proposals for review ordering without auto-approving them."""
    score = LAYER_QUALITY_WEIGHTS.get(str(item.get("evidence_layer") or ""), 12)
    priority = str(item.get("priority") or "").lower()
    score += PRIORITY_QUALITY_WEIGHTS.get(priority, 0)

    populated_fields = 0
    for field in REQUIRED_HYPOTHESIS_FIELDS:
        value = str(item.get(field) or "").strip()
        if value:
            populated_fields += 1
            score += 6
    evidence_text = str(item.get("evidence") or "")
    verification_text = str(item.get("verification_plan") or "")
    success_text = str(item.get("success_metric") or "")
    if re.search(r"\d|/|件|回|日|率|ログ|評価|比較", evidence_text):
        score += 8
    if any(term in verification_text for term in ("比較", "計測", "確認", "変更後", "前後", "1ヶ月", "7日", "30日")):
        score += 6
    if any(term in success_text for term in ("率", "件", "回", "減少", "増加", "低下", "向上", "時間")):
        score += 5
    if populated_fields <= 2:
        score -= 12
    return max(0, min(100, score))


def _selection_status(status: str) -> str:
    value = str(status or "").strip().lower()
    if value in APPROVED_STATUSES:
        return "selected"
    if value in REJECTED_STATUSES:
        return "selected_out"
    if value in DEFERRED_STATUSES:
        return "pending_selection"
    return "unselected"


def _genetic_profile(item: dict[str, Any]) -> dict[str, Any]:
    existing = item.get("genetic_profile")
    if isinstance(existing, dict) and existing:
        return existing

    selection = _selection_status(str(item.get("human_decision_status") or item.get("status") or ""))
    style = str(item.get("proposal_style") or "").strip()
    has_metric = bool(str(item.get("success_metric") or "").strip())
    has_risk = bool(str(item.get("risk") or "").strip())
    has_evidence = bool(str(item.get("evidence") or "").strip())

    fitness = 50
    if selection == "selected":
        fitness += 25
    elif selection == "selected_out":
        fitness -= 22
    elif selection == "pending_selection":
        fitness += 2
    if has_metric:
        fitness += 10
    if has_evidence:
        fitness += 8
    if has_risk:
        fitness += 4
    if style == "leap":
        fitness -= 4
    fitness = max(0, min(100, fitness))

    mutation_rate = 0.24 if style == "leap" else 0.1
    if selection == "selected":
        mutation_rate = 0.06
    elif selection == "selected_out":
        mutation_rate = 0.22
    elif selection == "pending_selection":
        mutation_rate = 0.14
    if not has_metric:
        mutation_rate += 0.04
    mutation_rate = round(max(0.04, min(0.35, mutation_rate)), 2)

    if selection == "selected":
        inheritance_policy = "preserve_and_refine"
    elif selection == "selected_out":
        inheritance_policy = "suppress_or_mutate"
    elif mutation_rate >= 0.2:
        inheritance_policy = "explore_variant"
    else:
        inheritance_policy = "observe_then_select"

    return {
        "fitness_score": fitness,
        "mutation_rate": mutation_rate,
        "selection_status": selection,
        "inheritance_policy": inheritance_policy,
    }


def _is_leap_proposal(item: dict[str, Any]) -> bool:
    text = " ".join(
        str(item.get(key) or "")
        for key in ("title", "hypothesis", "proposed_change", "summary", "risk", "verification_plan")
    )
    if str(item.get("proposal_style") or "").strip() == "leap":
        return True
    return any(marker in text for marker in LEAP_MARKERS)


def _proposal_maturity(score: int) -> str:
    if score >= 72:
        return "ready_for_review"
    if score >= 52:
        return "watch"
    return "thin"


def _review_status_label(maturity: str) -> str:
    return {
        "ready_for_review": "レビュー候補",
        "watch": "観測継続",
        "thin": "根拠不足",
    }.get(maturity, "観測継続")


def _proposal_rank_key(item: dict[str, Any]) -> tuple[int, int, str]:
    genetic = item.get("genetic_profile") if isinstance(item.get("genetic_profile"), dict) else {}
    fitness_bonus = int(float(genetic.get("fitness_score") or 0) * FITNESS_WEIGHT)
    return (
        int(item.get("quality_score") or 0),
        fitness_bonus,
        _proposal_sort_key(item),
    )


def _compact_text(value: Any, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _proposal_body(row: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get(key) or "")
        for key in (
            "body",
            "hypothesis",
            "evidence",
            "proposed_change",
            "suggestion",
            "reason",
            "review_point",
            "recommended_action",
            "detail",
            "description",
        )
        if str(row.get(key) or "").strip()
    )


def _canonical_key(title: str, body: str = "") -> str:
    title_text = _compact_text(title, 180)
    body_text = _compact_text(body, 600)
    if not title_text and not body_text:
        return ""
    digest = hashlib.sha1(f"{title_text}\n{body_text}".encode("utf-8")).hexdigest()[:16]
    return f"proposal:{digest}"


def _title_signature(title: str) -> str:
    text = re.sub(r"\s+", "", str(title or "").lower())
    text = re.sub(r"[「」『』（）()【】\[\]、。,.，．:：;；!！?？\-_ー/\\|]", "", text)
    return text[:180]


def _iter_report_paths() -> list[Path]:
    paths = list(REPORTS_DIR.glob("improvement_report_*.json"))
    latest = REPORTS_DIR / "latest.json"
    if latest.exists():
        paths.append(latest)
    return sorted(set(paths), reverse=True)[:120]


def _collect_resolved_self_proposal_refs() -> dict[str, Any]:
    """Return resolved proposal keys/titles from reports and the local improvement log.

    This is intentionally non-destructive: resolved self proposals disappear from the
    report section, while their source records remain available for audit.
    """
    resolved_statuses = {"applied", "approved", "deleted", "rejected", "suppressed", "deferred", "parked"}
    resolved_keys: set[str] = set()
    resolved_title_signatures: set[str] = set()

    def mark(item: dict[str, Any]) -> None:
        title = str(item.get("title") or item.get("matched_applied_title") or "").strip()
        body = _proposal_body(item)
        key = str(item.get("canonical_key") or item.get("key") or "").strip()
        if key:
            resolved_keys.add(key)
        elif title:
            resolved_keys.add(_canonical_key(title, body))
        signature = _title_signature(title)
        if signature:
            resolved_title_signatures.add(signature)

    for report_path in _iter_report_paths():
        report = _load_json(report_path)
        for bucket in ("applied", "applied_improvements", "suppressed_applied_duplicates"):
            rows = report.get(bucket)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    mark(row)

    for row in _load_jsonl(LOCAL_IMPROVEMENT_LOG):
        status = str(row.get("status") or row.get("action") or "").lower()
        event_type = str(row.get("event_type") or "").lower()
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        payload_status = str(payload.get("status") or payload.get("action") or "").lower()
        if event_type == "improvement_delete" or status in resolved_statuses or payload_status in resolved_statuses:
            mark({**payload, **row})

    # scripts/improvement_ledger.jsonl は追記形式で最後のエントリが有効(CLAUDE.md)。
    # ここを見ていないと、台帳でapplied/rejected済みにした自己提案が
    # 次回生成でも消えず出続けてしまう。
    latest_ledger_rows: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(REPO_LEDGER):
        ledger_key = str(row.get("key") or row.get("canonical_key") or "").strip()
        if not ledger_key:
            continue
        latest_ledger_rows[ledger_key] = row
    for row in latest_ledger_rows.values():
        status = str(row.get("status") or "").lower()
        if status in resolved_statuses:
            mark(row)

    return {
        "keys": resolved_keys,
        "title_signatures": resolved_title_signatures,
    }


def _is_resolved_proposal(item: dict[str, Any], refs: dict[str, Any]) -> bool:
    key = str(item.get("canonical_key") or "").strip()
    title = str(item.get("title") or "").strip()
    if key and key in refs.get("keys", set()):
        return True
    signature = _title_signature(title)
    return bool(signature and signature in refs.get("title_signatures", set()))


def _infer_evidence_layer(source: dict[str, Any]) -> str:
    explicit = str(source.get("evidence_layer") or "").strip()
    if explicit:
        return explicit
    source_name = str(source.get("source") or "")
    if source_name == "usage_loop":
        return "usage_based"
    if source_name in {"lease_system_gap_analysis", "system_audit"}:
        return "system_audit_based"
    return "feedback_based"


def collect_shion_self_proposals(limit: int = 10) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    counts_by_kind: dict[str, int] = {}
    counts_by_layer: dict[str, int] = {key: 0 for key in LAYER_LABELS}
    source_counts_by_kind: dict[str, int] = {}
    resolved_refs = _collect_resolved_self_proposal_refs()
    suppressed_resolved: list[dict[str, str]] = []

    for source in SOURCES:
        rows = _load_source_rows(source)
        source_counts_by_kind[source["kind"]] = len(rows)
        counts_by_kind.setdefault(source["kind"], 0)
        evidence_layer = _infer_evidence_layer(source)
        counts_by_layer.setdefault(evidence_layer, 0)
        for row in rows:
            title = str(row.get("title") or row.get("topic") or "").strip()
            if not title:
                continue
            body = _proposal_body(row)
            item = {
                "kind": source["kind"],
                "source": source["source"],
                "evidence_layer": evidence_layer,
                "evidence_layer_label": LAYER_LABELS.get(evidence_layer, evidence_layer),
                "title": title,
                "status": str(row.get("status") or "proposed"),
                "priority": str(row.get("priority") or ""),
                "generated_at": str(row.get("generated_at") or row.get("ts") or ""),
                "target": str(row.get("target_page") or row.get("topic") or ""),
                "hypothesis": str(row.get("hypothesis") or row.get("suggestion") or row.get("reason") or "").strip(),
                "evidence": str(row.get("evidence") or row.get("reason") or "").strip(),
                "proposed_change": str(row.get("proposed_change") or row.get("suggestion") or "").strip(),
                "success_metric": str(row.get("success_metric") or "").strip(),
                "verification_plan": str(row.get("verification_plan") or "").strip(),
                "risk": str(row.get("risk") or "").strip(),
                "proposal_schema": str(row.get("proposal_schema") or ""),
                "human_decision_status": str(row.get("human_decision_status") or row.get("status") or ""),
                "proposal_style": str(row.get("proposal_style") or ""),
                "genetic_profile": row.get("genetic_profile") if isinstance(row.get("genetic_profile"), dict) else {},
                "summary": _summary(row, source["summary_keys"]),
                "canonical_key": str(row.get("canonical_key") or row.get("key") or "").strip()
                or _canonical_key(title, body),
            }
            item["is_leap_proposal"] = _is_leap_proposal(item)
            item["genetic_profile"] = _genetic_profile(item)
            quality_score = _proposal_quality_score(item)
            maturity = _proposal_maturity(quality_score)
            item.update(
                {
                    "quality_score": quality_score,
                    "proposal_maturity": maturity,
                    "review_status_label": _review_status_label(maturity),
                    "review_policy": "human_decision_required",
                    "effect_tracking": "採用/保留/却下と、success_metric の事後変化で確認する",
                }
            )
            if _is_resolved_proposal(item, resolved_refs):
                suppressed_resolved.append(
                    {
                        "title": title,
                        "canonical_key": str(item.get("canonical_key") or ""),
                        "reason": "既に適用・削除・却下済みのため自己提案から自動除外",
                    }
                )
                continue
            counts_by_kind[source["kind"]] += 1
            counts_by_layer[evidence_layer] = counts_by_layer.get(evidence_layer, 0) + 1
            items.append(item)

    items.sort(key=_proposal_rank_key, reverse=True)
    review_limit = min(max(0, limit), DEFAULT_REVIEW_LIMIT)
    regular_items = [item for item in items if not item.get("is_leap_proposal")]
    leap_items = [item for item in items if item.get("is_leap_proposal")]
    reflection_items = [item for item in items if item.get("evidence_layer") == "reflection_based"]
    review_items = regular_items[:review_limit]
    if leap_items and len(review_items) >= review_limit and DEFAULT_LEAP_LIMIT > 0:
        review_items = review_items[: review_limit - DEFAULT_LEAP_LIMIT] + leap_items[:DEFAULT_LEAP_LIMIT]
    elif leap_items and len(review_items) < review_limit:
        review_items.extend(leap_items[: review_limit - len(review_items)])
    if (
        reflection_items
        and review_limit > 0
        and DEFAULT_REFLECTION_LIMIT > 0
        and not any(item.get("evidence_layer") == "reflection_based" for item in review_items)
    ):
        review_items = review_items[: review_limit - DEFAULT_REFLECTION_LIMIT] + reflection_items[:DEFAULT_REFLECTION_LIMIT]
    return {
        "label": "紫苑の自己提案",
        "note": "通常のneeds_reviewではなく、紫苑がログから出した改善仮説。採用判断は人間が行う。表示は上位3件に絞り、根拠層・検証可能性・効果追跡を優先する。",
        "count": len(items),
        "review_limit": review_limit,
        "displayed_count": len(review_items),
        "quality_policy": {
            "sort": "quality_score desc, fitness_score bonus, generated_at desc",
            "top_limit": DEFAULT_REVIEW_LIMIT,
            "leap_limit": DEFAULT_LEAP_LIMIT,
            "reflection_limit": DEFAULT_REFLECTION_LIMIT,
            "fitness_weight": FITNESS_WEIGHT,
            "required_fields": list(REQUIRED_HYPOTHESIS_FIELDS),
            "domestic_mode_input_fields": list(DOMESTIC_MODE_INPUT_FIELDS),
            "domestic_mode_doc": "docs/shion_domestic_mode_inputs.md",
            "genetic_loop": {
                "selected": "採用/applied 系統は低い mutation_rate で継承・微修正する",
                "selected_out": "却下/rejected 系統は同じ浅い案を抑え、必要なら別方向へ変異させる",
                "pending_selection": "保留/観測継続は成功指標と追加根拠を優先して再提案する",
                "leap": "跳躍提案は最大1件だけ混ぜ、測れる違和感として扱う",
            },
            "human_review_required": True,
        },
        "counts_by_kind": counts_by_kind,
        "source_counts_by_kind": source_counts_by_kind,
        "counts_by_layer": counts_by_layer,
        "layer_labels": LAYER_LABELS,
        "suppressed_resolved_count": len(suppressed_resolved),
        "suppressed_resolved": suppressed_resolved[:50],
        "items": review_items,
    }


def attach_to_report(path: Path, proposal_section: dict[str, Any]) -> bool:
    report = _load_json(path)
    if not report:
        return False

    section = dict(proposal_section)
    section["attached_at"] = datetime.now().isoformat(timespec="seconds")
    report["shion_self_proposals"] = section
    report["shion_self_proposal_count"] = int(section.get("count") or 0)
    summary = report.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["shion_self_proposal_count"] = int(section.get("count") or 0)
        summary["shion_self_proposal_counts_by_layer"] = section.get("counts_by_layer") or {}
    _write_json(path, report)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="Specific daily improvement report JSON.")
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST, help="reports/latest.json path.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    section = collect_shion_self_proposals(limit=args.limit)
    targets = []
    if args.report:
        targets.append(args.report.expanduser())
    if args.latest:
        targets.append(args.latest.expanduser())

    updated = []
    for target in dict.fromkeys(targets):
        if attach_to_report(target, section):
            updated.append(str(target))

    print(f"shion_self_proposals={section['count']} updated={len(updated)}")
    for path in updated:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
