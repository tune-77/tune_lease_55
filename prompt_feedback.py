"""Shared helpers for injecting prompt-improvement feedback."""

from __future__ import annotations

import difflib
import json
import os
from datetime import datetime
from hashlib import sha256
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDCA_RULES_FILE = os.path.join(_REPO_ROOT, "data", "pdca_ai_rules.json")


def load_pdca_rules(path: str | None = None) -> dict[str, Any]:
    """Load the latest PDCA rule set if it exists."""
    target = path or DEFAULT_PDCA_RULES_FILE
    if not os.path.exists(target):
        return {}
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_pdca_rules(data: dict[str, Any], path: str | None = None) -> bool:
    """Persist the PDCA rule set to disk."""
    target = path or DEFAULT_PDCA_RULES_FILE
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _dedupe_rules(rules: list[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        text = str(rule or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def build_pdca_prompt_block(
    path: str | None = None,
    *,
    max_addons: int = 3,
    include_summary: bool = True,
) -> str:
    """Build a compact prompt block that injects the latest learned rules."""
    data = load_pdca_rules(path)
    if not data:
        return ""

    addons = _dedupe_rules(data.get("ai_prompt_addons", []))[:max_addons]
    summary = str(data.get("reflection_summary", "")).strip()
    last_run = str(data.get("last_run", "")).strip()
    analyzed_count = data.get("analyzed_count", 0)

    if not addons and not (include_summary and summary):
        return ""

    lines = ["【自動学習システムからの特記事項（PDCA反映ルール）】"]
    if last_run or analyzed_count:
        run_date = last_run[:10] if len(last_run) >= 10 else last_run
        meta_bits = []
        if analyzed_count:
            meta_bits.append(f"直近 {analyzed_count} 件")
        if run_date:
            meta_bits.append(f"{run_date} 分析")
        if meta_bits:
            lines.append(f"（{' / '.join(meta_bits)}の結果）")
    if addons:
        lines.append("直近の審査傾向を踏まえ、以下のルールを必ず遵守して評価に反映させてください：")
        lines.extend(f"・{rule}" for rule in addons)
    if include_summary and summary:
        lines.append(f"（審査傾向サマリー: {summary[:200]}）")
    return "\n".join(lines)


def append_pdca_rule(
    rule: str,
    *,
    path: str | None = None,
    source: str = "manual",
    reflection_summary: str | None = None,
) -> dict[str, Any]:
    """Append a new learned rule while deduplicating existing addons."""
    target = path or DEFAULT_PDCA_RULES_FILE
    data = load_pdca_rules(target)
    addons = _dedupe_rules(list(data.get("ai_prompt_addons", [])))
    rule_text = str(rule or "").strip()
    appended = False
    if rule_text and rule_text not in addons:
        addons.append(rule_text)
        appended = True
    data["ai_prompt_addons"] = addons
    if reflection_summary is not None:
        data["reflection_summary"] = str(reflection_summary).strip()
    data.setdefault("last_run", data.get("last_run", ""))
    data["manual_rule_source"] = source
    data["manual_rule_count"] = int(data.get("manual_rule_count") or 0) + (1 if appended else 0)
    saved = save_pdca_rules(data, target)
    return {
        "ok": saved,
        "appended": appended,
        "rule": rule_text,
        "count": len(addons),
        "path": target,
        "data": data,
    }


def _shorten(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _digest(text: str) -> str:
    return sha256((text or "").encode("utf-8")).hexdigest()


def _unified_diff(before: str, after: str, *, fromfile: str = "before", tofile: str = "after", limit: int = 80) -> str:
    before_lines = (before or "").splitlines()
    after_lines = (after or "").splitlines()
    diff_lines = list(difflib.unified_diff(before_lines, after_lines, fromfile=fromfile, tofile=tofile, lineterm=""))
    if len(diff_lines) > limit:
        diff_lines = diff_lines[:limit] + ["... (diff truncated)"]
    return "\n".join(diff_lines)


def record_prompt_feedback(
    *,
    surface: str,
    question: str,
    base_prompt: str,
    final_prompt: str,
    response: str,
    log_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a prompt/response snapshot and its diff for later comparison."""
    target = log_path or os.path.join(_REPO_ROOT, "data", "prompt_feedback_log.jsonl")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    question_hash = _digest(question)
    final_hash = _digest(final_prompt)
    base_hash = _digest(base_prompt)
    response_hash = _digest(response)
    pdca_block = build_pdca_prompt_block()

    previous: dict[str, Any] | None = None
    if os.path.exists(target):
        try:
            with open(target, "r", encoding="utf-8") as f:
                for line in reversed(f.readlines()):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("surface") == surface and row.get("question_hash") == question_hash:
                        previous = row
                        break
        except Exception:
            previous = None

    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "surface": surface,
        "question": _shorten(question, 400),
        "question_hash": question_hash,
        "prompt_base_len": len(base_prompt or ""),
        "prompt_final_len": len(final_prompt or ""),
        "prompt_base_hash": base_hash,
        "prompt_final_hash": final_hash,
        "response_hash": response_hash,
        "pdca_applied": bool(pdca_block.strip()),
        "pdca_block_hash": _digest(pdca_block),
        "prompt_diff": _unified_diff(base_prompt, final_prompt, fromfile="base", tofile="pdca"),
        "response_text": _shorten(response, 4000),
        "response_preview": _shorten(response, 500),
        "response_len": len(response or ""),
    }

    if previous:
        prev_response = str(previous.get("response_text") or "")
        payload["previous_response_hash"] = previous.get("response_hash", "")
        payload["response_diff_from_previous"] = _unified_diff(prev_response, response, fromfile="previous", tofile="current")
    else:
        payload["previous_response_hash"] = ""
        payload["response_diff_from_previous"] = ""

    if extra:
        payload.update(extra)

    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload
