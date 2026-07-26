#!/usr/bin/env python3
"""Build a read-only memory effectiveness report for OKF-style knowledge notes.

This is an observation sidecar. It does not change RAG ranking, prompts,
Obsidian notes, scoring, canonical judgment assets, or feedback ledgers.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_base" / "okf_lease_concepts"
DEFAULT_MEMORY_INDEX = PROJECT_ROOT / "data" / "shion_memory_index.json"
DEFAULT_FEEDBACK_JSONL = PROJECT_ROOT / "data" / "judgment_asset_usage_feedback.jsonl"
DEFAULT_OKF_EVAL_JSON = PROJECT_ROOT / "reports" / "okf_rag_eval_latest.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "reports" / "obsidian_memory_effectiveness_latest.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "reports" / "obsidian_memory_effectiveness_latest.md"
DEFAULT_STATE_JSONL = PROJECT_ROOT / "data" / "obsidian_memory_effectiveness.jsonl"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
ALLOWED_STATES = {"dormant", "recalled", "used", "validated", "noisy"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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


def _load_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        import yaml

        loaded = yaml.safe_load(match.group(1)) or {}
    except Exception:
        loaded = {}
    return loaded if isinstance(loaded, dict) else {}, raw[match.end() :]


def _note_title(meta: dict[str, Any], body: str, path: Path) -> str:
    title = str(meta.get("title") or "").strip()
    if title:
        return title
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _as_terms(*values: Any) -> set[str]:
    terms: set[str] = set()
    for value in values:
        if isinstance(value, list):
            terms.update(_as_terms(*value))
            continue
        text = str(value or "").strip()
        if not text:
            continue
        terms.add(text.lower())
        for token in re.findall(r"[A-Za-z0-9_]{3,}|[一-龥ぁ-んァ-ンー]{2,}", text):
            terms.add(token.lower())
    return terms


def _memory_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("items", "memories", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _eval_hits(payload: Any) -> dict[str, dict[str, Any]]:
    hits: dict[str, dict[str, Any]] = defaultdict(lambda: {"hit_count": 0, "miss_count": 0, "best_rank": None})
    cases = payload.get("cases") if isinstance(payload, dict) else []
    if not isinstance(cases, list):
        return hits
    for case in cases:
        if not isinstance(case, dict):
            continue
        passed = bool(case.get("passed"))
        rank = case.get("rank")
        paths = case.get("expected_path_any") or case.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        for raw_path in paths:
            rel = str(raw_path or "").strip()
            if not rel:
                continue
            item = hits[rel]
            if passed:
                item["hit_count"] += 1
                try:
                    rank_int = int(rank)
                except (TypeError, ValueError):
                    rank_int = 999
                current = item["best_rank"]
                item["best_rank"] = rank_int if current is None else min(int(current), rank_int)
            else:
                item["miss_count"] += 1
    return hits


def _feedback_matches(rows: list[dict[str, Any]], terms_by_path: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {
        path: {"used": 0, "helped": 0, "neutral": 0, "challenged": 0, "rejected": 0}
        for path in terms_by_path
    }
    for row in rows:
        text = " ".join(
            str(row.get(key) or "")
            for key in ("rule_id", "judgment_asset_id", "asset_id", "case_id", "note", "source")
        ).lower()
        outcome = str(row.get("outcome") or row.get("status") or "").strip().lower()
        if outcome not in {"used", "helped", "neutral", "challenged", "rejected"}:
            continue
        for path, terms in terms_by_path.items():
            if any(term and term in text for term in terms):
                result[path]["used"] += 1
                if outcome != "used":
                    result[path][outcome] += 1
    return result


def _state(*, eval_hit_count: int, eval_miss_count: int, index_records: int, index_used_count: int, feedback: dict[str, int]) -> str:
    negative = int(feedback.get("challenged") or 0) + int(feedback.get("rejected") or 0)
    positive = int(feedback.get("helped") or 0)
    total_used = int(feedback.get("used") or 0) + index_used_count
    if negative > positive and (eval_hit_count > 0 or total_used > 0):
        return "noisy"
    if positive > 0:
        return "validated"
    if total_used > 0:
        return "used"
    if eval_hit_count > 0 or index_records > 0:
        return "recalled"
    if eval_miss_count > 0:
        return "noisy"
    return "dormant"


def _effectiveness_score(state: str, *, eval_hit_count: int, eval_miss_count: int, index_used_count: int, feedback: dict[str, int]) -> float:
    score = 0.0
    score += min(20.0, eval_hit_count * 6.0)
    score += min(20.0, index_used_count * 8.0)
    score += min(40.0, int(feedback.get("helped") or 0) * 18.0)
    score += min(10.0, int(feedback.get("neutral") or 0) * 5.0)
    score -= min(35.0, (int(feedback.get("challenged") or 0) + int(feedback.get("rejected") or 0)) * 14.0)
    score -= min(15.0, eval_miss_count * 4.0)
    if state == "dormant":
        score = min(score, 5.0)
    return round(max(0.0, min(100.0, score)), 1)


def _next_action(state: str) -> str:
    return {
        "dormant": "次回の審査レビューで想起されるか観測する。",
        "recalled": "回答・確認事項・稟議文面に実際に使われたかを確認する。",
        "used": "User評価を取り、helped / neutral / challenged を記録する。",
        "validated": "同種案件で再利用し、結果照合できる材料を待つ。",
        "noisy": "内容・適用条件・使わない条件を見直す。自動削除はしない。",
    }[state]


def build_report(
    *,
    knowledge_dir: Path,
    memory_index: Any,
    feedback_rows: list[dict[str, Any]],
    okf_eval: Any,
    target_date: str,
) -> dict[str, Any]:
    notes: list[dict[str, Any]] = []
    terms_by_path: dict[str, set[str]] = {}
    for path in sorted(knowledge_dir.rglob("*.md")):
        rel = path.relative_to(PROJECT_ROOT).as_posix() if path.is_absolute() else path.as_posix()
        meta, body = _load_frontmatter(path)
        title = _note_title(meta, body, path)
        terms = _as_terms(rel, path.stem, title, meta.get("tags"), meta.get("related"))
        terms_by_path[rel] = terms
        notes.append({"path": rel, "title": title, "meta": meta})

    index_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _memory_items(memory_index):
        rel = str(item.get("source_path") or "").strip()
        if rel:
            index_by_path[rel].append(item)

    eval_by_path = _eval_hits(okf_eval)
    feedback_by_path = _feedback_matches(feedback_rows, terms_by_path)

    records: list[dict[str, Any]] = []
    for note in notes:
        rel = note["path"]
        indexed = index_by_path.get(rel, [])
        index_used = sum(1 for item in indexed if str(item.get("last_used_at") or "").strip())
        eval_item = eval_by_path.get(rel, {"hit_count": 0, "miss_count": 0, "best_rank": None})
        feedback = feedback_by_path.get(rel, {})
        state = _state(
            eval_hit_count=int(eval_item.get("hit_count") or 0),
            eval_miss_count=int(eval_item.get("miss_count") or 0),
            index_records=len(indexed),
            index_used_count=index_used,
            feedback=feedback,
        )
        score = _effectiveness_score(
            state,
            eval_hit_count=int(eval_item.get("hit_count") or 0),
            eval_miss_count=int(eval_item.get("miss_count") or 0),
            index_used_count=index_used,
            feedback=feedback,
        )
        records.append(
            {
                "obsidian_ref": rel,
                "title": note["title"],
                "memory_type": str(note["meta"].get("type") or ""),
                "domain": str(note["meta"].get("domain") or ""),
                "state": state,
                "effectiveness_score": score,
                "recalled_count": int(eval_item.get("hit_count") or 0),
                "eval_miss_count": int(eval_item.get("miss_count") or 0),
                "index_records": len(indexed),
                "used_count": int(feedback.get("used") or 0) + index_used,
                "human_accept_count": int(feedback.get("helped") or 0),
                "challenged_count": int(feedback.get("challenged") or 0),
                "rejected_count": int(feedback.get("rejected") or 0),
                "best_eval_rank": eval_item.get("best_rank"),
                "status": str(note["meta"].get("status") or ""),
                "confidence": str(note["meta"].get("confidence") or ""),
                "next_action": _next_action(state),
            }
        )

    summary = {state: 0 for state in sorted(ALLOWED_STATES)}
    for record in records:
        summary[record["state"]] += 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "date": target_date,
        "schema_version": 1,
        "mode": "read_only_observation",
        "guardrail": "no_rag_rank_change_no_prompt_change_no_obsidian_write_no_auto_promotion",
        "knowledge_dir": str(knowledge_dir),
        "summary": {
            "total": len(records),
            **summary,
        },
        "records": sorted(records, key=lambda item: (item["state"], -float(item["effectiveness_score"]), item["title"])),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    lines = [
        "# Obsidian Memory Effectiveness",
        "",
        f"- Date: {payload['date']}",
        f"- Mode: {payload['mode']}",
        f"- Guardrail: {payload['guardrail']}",
        f"- Knowledge dir: `{payload['knowledge_dir']}`",
        "",
        "## Summary",
        "",
        f"- Total: {summary['total']}",
        f"- Dormant: {summary['dormant']}",
        f"- Recalled: {summary['recalled']}",
        f"- Used: {summary['used']}",
        f"- Validated: {summary['validated']}",
        f"- Noisy: {summary['noisy']}",
        "",
        "## Records",
        "",
    ]
    for item in records:
        lines.extend(
            [
                f"### {item['title']}",
                f"- Ref: `{item['obsidian_ref']}`",
                f"- Type: `{item['memory_type']}` / Domain: `{item['domain']}`",
                f"- State: `{item['state']}` / Score: {item['effectiveness_score']}",
                f"- Signals: recalled={item['recalled_count']}, used={item['used_count']}, helped={item['human_accept_count']}, challenged={item['challenged_count']}, rejected={item['rejected_count']}",
                f"- Next: {item['next_action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- このレポートは観測のみ。RAG順位、プロンプト、Obsidian本文、判断資産active storeは変更しない。",
            "- RAG評価で引けたものは `recalled` として扱うが、実業務で使われた証拠とは分ける。",
            "- `validated` は人間の helped 等の明示フィードバックがある場合だけ強く扱う。",
            "",
        ]
    )
    return "\n".join(lines)


def write_state_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("records") if isinstance(payload.get("records"), list) else []
    snapshot_date = str(payload.get("date") or "")
    existing = _read_jsonl(path)
    kept = [row for row in existing if str(row.get("date") or "") != snapshot_date]
    for row in rows:
        kept.append({"date": snapshot_date, **row})
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--memory-index", type=Path, default=DEFAULT_MEMORY_INDEX)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--okf-eval-json", type=Path, default=DEFAULT_OKF_EVAL_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--state-jsonl", type=Path, default=DEFAULT_STATE_JSONL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_report(
        knowledge_dir=args.knowledge_dir.expanduser(),
        memory_index=_read_json(args.memory_index.expanduser()),
        feedback_rows=_read_jsonl(args.feedback_jsonl.expanduser()),
        okf_eval=_read_json(args.okf_eval_json.expanduser()),
        target_date=args.date,
    )
    _write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    write_state_jsonl(args.state_jsonl, payload)
    summary = payload["summary"]
    print(
        "Obsidian Memory Effectiveness: "
        f"total={summary['total']} recalled={summary['recalled']} "
        f"used={summary['used']} validated={summary['validated']} noisy={summary['noisy']}"
    )
    print(f"report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
