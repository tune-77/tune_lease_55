"""Shared helpers for injecting prompt-improvement feedback."""

from __future__ import annotations

import difflib
import json
import os
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any

from runtime_paths import get_data_path

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDCA_RULES_FILE = get_data_path("pdca_ai_rules.json")


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


def _meta_rule_status(data: dict[str, Any]) -> dict[str, bool]:
    today = datetime.now().date()
    meta = data.get("pdca_rule_meta") or []
    if not isinstance(meta, list):
        return {}
    status: dict[str, bool] = {}
    for item in meta:
        if not isinstance(item, dict):
            continue
        text = str(item.get("rule") or "").strip()
        if not text:
            continue
        active = str(item.get("status") or "active").strip() != "inactive"
        expires_at = str(item.get("expires_at") or "").strip()
        if expires_at:
            try:
                active = active and datetime.fromisoformat(expires_at[:10]).date() >= today
            except ValueError:
                pass
        status[text] = active
    return status


def _manual_rules_from_data(data: dict[str, Any]) -> list[str]:
    meta_status = _meta_rule_status(data)
    explicit = _dedupe_rules(list(data.get("manual_ai_prompt_addons") or []))
    if explicit:
        return [rule for rule in explicit if meta_status.get(rule, True)]
    try:
        count = int(data.get("manual_rule_count") or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return []
    addons = _dedupe_rules(list(data.get("ai_prompt_addons", [])))
    return [rule for rule in addons[-count:] if meta_status.get(rule, True)]


def _active_rule_texts(data: dict[str, Any]) -> list[str]:
    meta_status = _meta_rule_status(data)
    meta = data.get("pdca_rule_meta") or []
    if not isinstance(meta, list):
        meta = []
    active: list[str] = []
    meta_rules: set[str] = set()
    for item in meta:
        if not isinstance(item, dict):
            continue
        text = str(item.get("rule") or "").strip()
        if not text:
            continue
        meta_rules.add(text)
        if meta_status.get(text, True):
            active.append(text)
    legacy = [
        rule
        for rule in list(data.get("ai_prompt_addons", []))
        if str(rule or "").strip() not in meta_rules
    ]
    return _dedupe_rules(active + legacy)


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

    manual_addons = _manual_rules_from_data(data)
    addons = _dedupe_rules(manual_addons + _active_rule_texts(data))[:max_addons]
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
    ttl_days: int = 90,
) -> dict[str, Any]:
    """Append a new learned rule while deduplicating existing addons."""
    target = path or DEFAULT_PDCA_RULES_FILE
    data = load_pdca_rules(target)
    addons = _dedupe_rules(list(data.get("ai_prompt_addons", [])))
    manual_addons = _manual_rules_from_data(data)
    rule_text = str(rule or "").strip()
    appended = False
    if rule_text and rule_text not in addons:
        addons.append(rule_text)
        appended = True
    if rule_text and rule_text not in manual_addons:
        manual_addons.append(rule_text)
    data["ai_prompt_addons"] = addons
    data["manual_ai_prompt_addons"] = _dedupe_rules(manual_addons)
    meta = [item for item in (data.get("pdca_rule_meta") or []) if isinstance(item, dict)]
    existing_meta_rules = {str(item.get("rule") or "").strip() for item in meta}
    now = datetime.now()
    try:
        ttl = max(1, int(ttl_days))
    except (TypeError, ValueError):
        ttl = 90
    if rule_text and rule_text not in existing_meta_rules:
        meta.append(
            {
                "rule": rule_text,
                "source": source,
                "created_at": now.isoformat(timespec="seconds"),
                "expires_at": (now + timedelta(days=ttl)).date().isoformat(),
                "status": "active",
            }
        )
    elif rule_text:
        for item in meta:
            if str(item.get("rule") or "").strip() == rule_text:
                item["source"] = source
                item["renewed_at"] = now.isoformat(timespec="seconds")
                item["expires_at"] = (now + timedelta(days=ttl)).date().isoformat()
                item["status"] = "active"
                break
    data["pdca_rule_meta"] = meta
    if reflection_summary is not None:
        data["reflection_summary"] = str(reflection_summary).strip()
    data.setdefault("last_run", data.get("last_run", ""))
    data["manual_rule_source"] = source
    try:
        manual_count = int(data.get("manual_rule_count") or 0)
    except (TypeError, ValueError):
        manual_count = 0
    data["manual_rule_count"] = manual_count + (1 if appended else 0)
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
    target = log_path or get_data_path("prompt_feedback_log.jsonl")
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

    # Cloud Run はコンテナのローカルディスクが再起動・再デプロイで消えるため、
    # ローカル保存に加えて GCS にも複製する（夜間パイプラインが
    # data/prompt_feedback_log.jsonl へ合流させる）。K_SERVICE/CLOUDRUN_DATA_MODE
    # が無いローカル実行では record_cloudrun_input_event 内部で no-op になる。
    _writeback_to_cloudrun_gcs(surface=surface, payload=payload)

    return payload


def _writeback_to_cloudrun_gcs(*, surface: str, payload: dict[str, Any]) -> None:
    try:
        from api.cloudrun_writeback import record_cloudrun_input_event

        record_cloudrun_input_event(
            event_type="prompt_feedback",
            surface=surface,
            payload=payload,
        )
    except Exception as exc:
        print(f"[PromptFeedback] GCS writeback skipped: {exc}")
