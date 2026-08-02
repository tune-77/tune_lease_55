"""Distill Vertex Answer API output into durable local Obsidian notes.

Vertex AI Search's corpus is a copy of the Obsidian Vault already synced via
``scripts/sync_obsidian_to_vertex_agent_search.py`` — it does not create new
knowledge on its own. The only genuinely new artifact Vertex produces per chat
turn is the Answer API's synthesized, cited summary. This module captures that
summary as a Markdown note in the Vault so it survives after
``VERTEX_AGENT_SEARCH_ENABLED`` is turned off (e.g. once the GCP promo credit
that currently funds this pilot runs out).

This must never raise into ``/api/chat`` (a documented risk area in
CLAUDE.md): every failure mode (missing vault, write error) degrades to a
skipped capture, not an exception.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

OUTPUT_DIR_REL = Path("Projects") / "tune_lease_55" / "Research" / "Vertex Distilled"
VALIDITY_DAYS = 90


def _canonical_query_key(query: str) -> str:
    """Stable dedup key for a query, independent of whitespace/case."""
    normalized = re.sub(r"\s+", " ", str(query or "").strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def should_capture(vertex_answer_api: dict[str, Any] | None) -> bool:
    """Whether this turn's Vertex Answer API result is worth persisting.

    Requires an actually-synthesized answer, not just a raw search hit list —
    the summary text is the one thing that doesn't already exist in the Vault.
    """
    if not vertex_answer_api:
        return False
    if not vertex_answer_api.get("used") or vertex_answer_api.get("status") != "ok":
        return False
    return bool(str(vertex_answer_api.get("answer_text") or "").strip())


def build_note(query: str, vertex_answer_api: dict[str, Any], *, today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    valid_until = today + dt.timedelta(days=VALIDITY_DAYS)
    answer_text = str(vertex_answer_api.get("answer_text") or "").strip()
    refs = [str(r).strip() for r in (vertex_answer_api.get("refs") or []) if str(r or "").strip()]
    grounding_score = vertex_answer_api.get("grounding_score")
    grounding_score_source = vertex_answer_api.get("grounding_score_source") or "grounding_api"
    ref_lines = "\n".join(f"- {ref}" for ref in refs) or "- (参照なし)"

    query_yaml = json.dumps(str(query or "").strip(), ensure_ascii=False)
    grounding_line = (
        f"grounding_score: {grounding_score}" if grounding_score is not None else "grounding_score: null"
    )

    return f"""---
date: {today.isoformat()}
source: vertex_answer_api
knowledge_type: vertex_distilled_answer
question: {query_yaml}
{grounding_line}
grounding_score_source: {grounding_score_source}
valid_until: {valid_until.isoformat()}
review_status: needs_human_review
tags: ["リース審査", "vertex-distilled"]
---
# Vertex Distilled Answer

> このノートは Vertex AI Search（Answer API）が生成した要約の写しです。
> 審査判断の補助知識として扱い、個別案件の事実確認を省略しないこと。
> 自動否決・自動承認には使用しない。Vertex 連携がオフになっても
> この要約自体はローカルに残る想定で保存している。

## 質問
{query}

## 要約
{answer_text}

## 参照
{ref_lines}
"""


def capture_vertex_distillation(
    query: str,
    vertex_answer_api: dict[str, Any] | None,
    *,
    vault_path: str | Path,
    state_path: Path,
) -> dict[str, Any]:
    """Write a distilled Vertex answer to Obsidian, deduplicated by query.

    Never raises: any failure returns ``{"captured": False, "reason": ...}``.
    """
    if not should_capture(vertex_answer_api):
        return {"captured": False, "reason": "not_capturable"}

    vault = Path(vault_path) if vault_path else None
    if not vault or not vault.is_dir():
        return {"captured": False, "reason": "vault_not_found"}

    key = _canonical_query_key(query)

    state: dict[str, Any] = {}
    try:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}

    existing = state.get(key)
    if existing:
        return {"captured": True, "duplicate": True, "note_path": existing.get("note_path")}

    try:
        output_dir = vault / OUTPUT_DIR_REL
        output_dir.mkdir(parents=True, exist_ok=True)
        today = dt.date.today()
        filename = f"{today.isoformat()}_{key}.md"
        note_path = output_dir / filename
        note_path.write_text(build_note(query, vertex_answer_api, today=today), encoding="utf-8")
    except OSError as exc:
        return {"captured": False, "reason": "write_error", "error": str(exc)[:240]}

    rel_path = str((OUTPUT_DIR_REL / filename))
    state[key] = {
        "note_path": rel_path,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # ノート自体は書けているので、state 更新失敗は無視して先へ進む

    return {"captured": True, "duplicate": False, "note_path": rel_path}
