#!/usr/bin/env python3
"""Build a read-only promotion-readiness report for judgment asset candidates.

architecture audit（2026-07-29）の agents/improver 層への提案「正式判断資産への
昇格条件を read-only レポートで定義し、いきなり自動適用しない」への対応。

昇格候補の並び順は既存の判断資産候補エンドポイント
(api/routers/feedback_loop.py::_load_judgment_asset_promotion_candidates) に
ヒューリスティックとして埋め込まれているだけで、基準が文書化されていなかった。
このスクリプトはその基準を明文化し、各候補を条件で分類するだけで、実際の昇格
（/api/judgment-asset-candidates/{id}/promote）には一切接続しない。
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES_JSONL = PROJECT_ROOT / "data" / "autoresearch_judgment_asset_candidates.jsonl"
DEFAULT_STATE_JSON = PROJECT_ROOT / "data" / "autoresearch_judgment_asset_candidate_state.json"
DEFAULT_CANONICAL_JSON = PROJECT_ROOT / "data" / "canonical_judgment_rules.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "judgment_asset_promotion_readiness_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "judgment_asset_promotion_readiness_latest.md"

GUARDRAIL = "review_only_no_auto_promotion_no_prompt_no_scoring_no_gcs_no_cloudrun_no_obsidian_write"

MIN_CLAIM_LENGTH = 12
SCORE_WEIGHTS = {"useful": 4, "edit": 3, "use": 1, "manual_bonus": 2, "rejected": -5}
SCORE_FORMULA = "useful_count*4 + edit_count*3 + use_count*1 + (2 if manual) - rejected_count*5"

BUCKET_ORDER = [
    "ready_for_review",
    "caution_mixed_signal",
    "insufficient_evidence",
    "duplicate_no_new_evidence",
    "not_eligible",
    "held_by_human",
    "already_promoted",
    "already_rejected",
]
BUCKET_CRITERIA = {
    "ready_for_review": "score > 0 かつ rejected_count == 0",
    "caution_mixed_signal": "score > 0 かつ rejected_count > 0（賛否混在、人間が精査）",
    "insufficient_evidence": "score <= 0 かつ manual 由来でない（実案件での評価がまだ足りない）",
    "duplicate_no_new_evidence": "既存の正規判断資産と同一文面で、新しい根拠（useful/edit）が無い",
    "not_eligible": f"文面が{MIN_CLAIM_LENGTH}文字未満など、そもそも候補として不十分",
    "held_by_human": "人間が既に「保留」を選択済み。スコアに関わらずready扱いにしない。",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canonical_rules(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    rules = canonical.get("rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    rules = canonical.get("canonical_rules")
    if isinstance(rules, list):
        return [item for item in rules if isinstance(item, dict)]
    return []


def _active_statements(canonical: dict[str, Any]) -> set[str]:
    statements: set[str] = set()
    for rule in _canonical_rules(canonical):
        if rule.get("status") != "active" or rule.get("private") is True:
            continue
        statement = str(rule.get("canonical_statement") or "").strip()
        if statement:
            statements.add(" ".join(statement.split()))
    return statements


def _score(*, useful_count: int, edit_count: int, use_count: int, rejected_count: int, is_manual: bool) -> int:
    return (
        useful_count * SCORE_WEIGHTS["useful"]
        + edit_count * SCORE_WEIGHTS["edit"]
        + use_count * SCORE_WEIGHTS["use"]
        + (SCORE_WEIGHTS["manual_bonus"] if is_manual else 0)
        + rejected_count * SCORE_WEIGHTS["rejected"]
    )


def evaluate_candidate(
    item: dict[str, Any],
    *,
    state: dict[str, Any],
    active_statements: set[str],
) -> dict[str, Any]:
    candidate_id = str(item.get("id") or "")
    merged = {**item, **dict(state.get(candidate_id) or {})}
    promotion_status = str(merged.get("promotion_status") or item.get("promotion_status") or "not_promoted")
    claim = str(merged.get("edited_claim") or merged.get("effective_claim") or merged.get("claim") or "").strip()
    use_count = int(merged.get("use_count") or 0)
    useful_count = int(merged.get("useful_count") or 0)
    rejected_count = int(merged.get("rejected_count") or 0)
    edit_count = int(merged.get("edit_count") or 0)
    is_manual = (
        str(merged.get("source_section") or "") == "manual_input"
        or str(merged.get("research_topic") or "").startswith("manual")
    )
    already_active_statement = bool(claim) and " ".join(claim.split()) in active_statements

    reasons: list[str] = []
    score: int | None = None

    if promotion_status in {"active", "promoted"}:
        bucket = "already_promoted"
        reasons.append("既に正規判断資産として昇格済み。")
    elif promotion_status in {"rejected_or_deprioritized", "rejected"}:
        bucket = "already_rejected"
        reasons.append("既に却下・優先度低として処理済み。")
    elif promotion_status == "held":
        bucket = "held_by_human"
        reasons.append("人間が既に「保留」を選択済み。スコアに関わらずready扱いにしない。")
    elif len(claim) < MIN_CLAIM_LENGTH:
        bucket = "not_eligible"
        reasons.append(f"文面が短すぎる（{len(claim)}文字 < {MIN_CLAIM_LENGTH}文字）。")
    elif already_active_statement and useful_count <= 0 and edit_count <= 0:
        bucket = "duplicate_no_new_evidence"
        reasons.append("既存の正規判断資産と同一文面で、新しい根拠（useful/edit）が無い。")
    else:
        score = _score(
            useful_count=useful_count, edit_count=edit_count, use_count=use_count,
            rejected_count=rejected_count, is_manual=is_manual,
        )
        if score <= 0 and not is_manual:
            bucket = "insufficient_evidence"
            reasons.append(f"score={score}（実案件での使用・評価がまだ少ない）。")
        elif rejected_count > 0:
            bucket = "caution_mixed_signal"
            reasons.append(f"score={score}だが rejected={rejected_count}件あり、賛否が割れている。")
        else:
            bucket = "ready_for_review"
            reasons.append(f"score={score}、reject無し。人間レビューへ回してよい状態。")

    return {
        "id": candidate_id,
        "claim": claim,
        "research_topic": str(merged.get("research_topic") or ""),
        "candidate_type": str(merged.get("candidate_type") or ""),
        "promotion_status": promotion_status,
        "use_count": use_count,
        "useful_count": useful_count,
        "rejected_count": rejected_count,
        "edit_count": edit_count,
        "is_manual": is_manual,
        "already_active_statement": already_active_statement,
        "score": score,
        "bucket": bucket,
        "reasons": reasons,
    }


def build_report(
    *,
    target_date: str,
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    active_statements = _active_statements(canonical)
    evaluated = [
        evaluate_candidate(item, state=state, active_statements=active_statements)
        for item in candidates
        if str(item.get("id") or "")
    ]

    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in BUCKET_ORDER}
    for entry in evaluated:
        buckets.setdefault(entry["bucket"], []).append(entry)

    summary = {key: len(buckets.get(key, [])) for key in BUCKET_ORDER}
    summary["total_candidates"] = len(evaluated)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target_date,
        "schema_version": 1,
        "mode": "local_review_only",
        "guardrail": GUARDRAIL,
        "criteria": {
            "min_claim_length": MIN_CLAIM_LENGTH,
            "score_weights": SCORE_WEIGHTS,
            "score_formula": SCORE_FORMULA,
            "buckets": BUCKET_CRITERIA,
        },
        "summary": summary,
        "buckets": buckets,
        "notes": [
            "このレポートは昇格基準の可視化専用。/api/judgment-asset-candidates/{id}/promote は一切呼ばない。",
            "ready_for_review は「自動昇格してよい」ではなく「人間が見るべき候補」を意味する。最終判断は人間。",
            "caution_mixed_signal は自動除外せず、reject理由を人間が読んでから判断する対象として残す。",
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    criteria = payload["criteria"]
    lines = [
        "# Judgment Asset Promotion Readiness",
        "",
        f"- Date: {payload['date']}",
        f"- Mode: {payload['mode']}",
        f"- Guardrail: {payload['guardrail']}",
        f"- Total candidates evaluated: {summary['total_candidates']}",
        "",
        "## 昇格基準（明文化）",
        "",
        f"- スコア式: `{criteria['score_formula']}`",
        f"- 最低文字数: {criteria['min_claim_length']}文字",
        "",
    ]
    for bucket_key in BUCKET_ORDER[:6]:
        lines.append(f"- **{bucket_key}**: {criteria['buckets'].get(bucket_key, '')}")
    lines.append("")

    labels = [
        ("人間レビュー推奨", "ready_for_review"),
        ("賛否混在・要精査", "caution_mixed_signal"),
        ("根拠不足", "insufficient_evidence"),
        ("重複・新規根拠なし", "duplicate_no_new_evidence"),
        ("対象外", "not_eligible"),
        ("人間が保留済み", "held_by_human"),
    ]
    for label, key in labels:
        entries = payload["buckets"].get(key) or []
        lines.extend([f"## {label} ({len(entries)})", ""])
        if not entries:
            lines.extend(["- None", ""])
            continue
        for entry in entries[:20]:
            claim_preview = entry["claim"][:80] + ("..." if len(entry["claim"]) > 80 else "")
            lines.append(f"- `{entry['id']}` {entry['research_topic']}: {claim_preview}")
            lines.append(f"  - {' / '.join(entry['reasons'])}")
        lines.append("")

    lines.extend(["## Notes", ""])
    for note in payload.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--candidates-jsonl", type=Path, default=DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--canonical-json", type=Path, default=DEFAULT_CANONICAL_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_report(
        target_date=args.date,
        candidates=_read_jsonl(args.candidates_jsonl),
        state=_read_json(args.state_json),
        canonical=_read_json(args.canonical_json),
    )
    _write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
