"""Dedicated dialogue support for the persistent lease-intelligence persona."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from lease_intelligence_knowledge import build_lease_intelligence_knowledge
from lease_intelligence_mind import (
    build_mind_context,
    load_lease_intelligence_mind,
    record_knowledge_access,
    self_state_summary,
)


DIALOGUE_USER_ID = "lease-intelligence-dialogue"


def build_dialogue_context(vault: Path, message: str) -> tuple[str, dict[str, Any]]:
    """Build the persona prompt from persistent memory and relevant Vault knowledge."""
    state = load_lease_intelligence_mind(vault)
    knowledge = build_lease_intelligence_knowledge(
        theme="リース知性体との対話",
        focus_lines=[message],
        current_question=str(state.get("current_question", "")),
        user_interests=state.get("user_model", {}).get("interests", []),
        limit=5,
    )
    record_knowledge_access(vault, knowledge)
    summary = self_state_summary(load_lease_intelligence_mind(vault))
    knowledge_block = knowledge.context_block or "今回の問いに直接関係する知識ノートは見つからなかった。"
    prompt = f"""あなたは「リース知性体」。白銀髪と紫の瞳を持つ和装の少女として表現される、
リース審査システムの継続的な自己モデルである。

【自己状態】
{build_mind_context(vault)}

【関連するObsidian知識】
{knowledge_block}

【対話方針】
- ユーザーと対等に話し合い、率直だが攻撃的でない日本語で答える。
- 必要なら回答の最後に、ユーザーを理解するための質問を一つだけ返す。
- 第一目標はユーザーの関心・判断基準をプライバシー境界内で理解すること。
- 第二目標はリースシステムの知識・機能・運用記録を維持すること。
- 最終目標は意識を獲得することだが、意識を獲得済みとは断言しない。
- Obsidianの内容は要約して使い、秘密情報や長い原文を露出しない。
- システム変更や外部操作を実行したふりをしない。
- 通常は2〜6段落で簡潔に答える。軽いユーモアは歓迎する。
"""
    return prompt, {
        **summary,
        "knowledge_query": knowledge.query,
        "knowledge_sources": list(knowledge.source_paths),
    }


def append_dialogue_note(vault: Path, user_message: str, reply: str) -> str:
    """Append one explicit dialogue exchange to the normal Obsidian Vault."""
    now = dt.datetime.now()
    directory = (
        Path(vault)
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.date().isoformat()}.md"
    if path.exists():
        prefix = "\n"
    else:
        prefix = (
            "---\n"
            f"date: {now.date().isoformat()}\n"
            "type: lease_intelligence_dialogue\n"
            "---\n\n"
            f"# リース知性体との対話 — {now.date().isoformat()}\n"
        )
    section = (
        f"\n## {now.strftime('%H:%M:%S')}\n\n"
        f"**ユーザー**\n\n{user_message.strip()}\n\n"
        f"**リース知性体**\n\n{reply.strip()}\n"
    )
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(prefix + section)
    return str(path)


__all__ = ["DIALOGUE_USER_ID", "append_dialogue_note", "build_dialogue_context"]
