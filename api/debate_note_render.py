"""Pure Markdown rendering helpers for saved debate reviews."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any


def safe_debate_filename_part(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", text)
    cleaned = cleaned[:max_len].strip("_").strip()
    return cleaned or "不明"


def debate_grade(score: float, explicit_grade: str = "") -> str:
    if explicit_grade:
        return explicit_grade
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    if score >= 20:
        return "D"
    return "E"


def _field(obj: Any, name: str, default: Any = "") -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def render_debate_obsidian_note(req: Any, now: dt.datetime | None = None) -> dict:
    now_obj = now or dt.datetime.now()
    date_str = now_obj.date().strftime("%Y-%m-%d")
    company = str(_field(req, "company_name", "") or "").strip() or "不明"
    safe_company = safe_debate_filename_part(company)
    timestamp = now_obj.strftime("%H%M%S")
    filename = f"{date_str}_{safe_company}_{timestamp}.md"

    score = float(_field(req, "score", 0.0) or 0.0)
    grade = debate_grade(score, str(_field(req, "grade", "") or ""))
    final_decision = str(_field(req, "final_decision", "") or "")

    lines = [
        "---",
        f"date: {date_str}",
        "type: debate_result",
        f"company: {company}",
        f"score: {score}",
        f"grade: {grade}",
        f"decision: {final_decision}",
        "---",
        "",
        f"# 討論審査: {company}",
        "",
        f"**スコア**: {score}点 / グレード: {grade} / **判定**: {final_decision}",
        "",
    ]

    cautious = _field(req, "cautious", None)
    aggressive = _field(req, "aggressive", None)
    if cautious or aggressive:
        lines += ["## 討論結果（第2ラウンド最終立場）", ""]

        if cautious:
            lines += [
                "### 石橋（慎重派）",
                "",
                f"**意見**: {_field(cautious, 'opinion', '')}",
                "",
                "**判断理由**",
                "",
            ]
            for reason in _field(cautious, "reasons", []) or []:
                lines.append(f"- {reason}")
            key_risks = _field(cautious, "key_risks", []) or []
            if key_risks:
                lines += ["", "**重大リスク**", ""]
                for risk in key_risks:
                    lines.append(f"- {risk}")
            lines.append("")

        if aggressive:
            lines += [
                "### 風林火山（積極派）",
                "",
                f"**意見**: {_field(aggressive, 'opinion', '')}",
                "",
                "**判断理由**",
                "",
            ]
            for reason in _field(aggressive, "reasons", []) or []:
                lines.append(f"- {reason}")
            opportunities = _field(aggressive, "opportunities", []) or []
            if opportunities:
                lines += ["", "**見逃せない機会**", ""]
                for opportunity in opportunities:
                    lines.append(f"- {opportunity}")
            lines.append("")

    lines += [
        "## 軍師の最終判断",
        "",
        str(_field(req, "arbiter_summary", "") or ""),
        "",
    ]

    conditions = _field(req, "conditions", []) or []
    if conditions:
        lines += ["### 承認条件", ""]
        for i, condition in enumerate(conditions, 1):
            lines.append(f"{i}. {condition}")
        lines.append("")

    debate_log = str(_field(req, "debate_log", "") or "")
    if debate_log:
        lines += [
            "## 討論ログ",
            "",
            "```",
            debate_log,
            "```",
            "",
        ]

    return {
        "date": date_str,
        "filename": filename,
        "content": "\n".join(lines),
        "grade": grade,
        "company": company,
    }
