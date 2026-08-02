"""Human response feedback helpers for chat prompt shaping."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


HUMAN_RESPONSE_POSITIVE_RATINGS = {"shion_like", "good"}
HUMAN_RESPONSE_NEGATIVE_RATINGS = {"thin", "generic", "not_shion", "bad"}

SHION_TONE_FEEDBACK_TERMS = (
    "硬い",
    "堅い",
    "お堅い",
    "かたい",
    "固い",
    "堅苦しい",
    "重い",
    "大げさ",
    "仰々しい",
    "演説",
    "長い",
    "くどい",
    "回りくどい",
    "詩的",
    "ポエム",
    "自然じゃない",
    "人間味",
    "温度感",
    "話し方",
    "口調",
    "文体",
    "言い方",
    "トーン",
)


def load_human_response_feedback(feedback_log: Path, limit: int = 80) -> list[dict[str, Any]]:
    try:
        lines = feedback_log.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-max(1, limit):]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def summarize_human_response_feedback(feedback_log: Path, route: str, limit: int = 80) -> dict[str, Any]:
    rows = load_human_response_feedback(feedback_log, limit=limit)
    matched = [row for row in rows if str(row.get("route") or "") in {route, "default", ""}]
    positive = [row for row in matched if str(row.get("rating") or "") in HUMAN_RESPONSE_POSITIVE_RATINGS]
    negative = [row for row in matched if str(row.get("rating") or "") in HUMAN_RESPONSE_NEGATIVE_RATINGS]

    def _starts(items: list[dict[str, Any]]) -> list[str]:
        starts: list[str] = []
        seen: set[str] = set()
        for item in reversed(items):
            text = str(item.get("response_start") or "").strip().splitlines()[0:1]
            start = text[0].strip() if text else ""
            if not start:
                continue
            start = start[:120]
            if start in seen:
                continue
            seen.add(start)
            starts.append(start)
            if len(starts) >= 3:
                break
        return starts

    return {
        "route": route,
        "rows": len(matched),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_starts": _starts(positive),
        "negative_starts": _starts(negative),
        "recent_comments": [
            str(row.get("comment") or "").strip()[:160]
            for row in reversed(matched)
            if str(row.get("comment") or "").strip()
        ][:3],
    }


def build_shion_light_tone_feedback_prompt_block(message: str) -> str:
    compact = re.sub(r"\s+", "", str(message or ""))
    if not compact:
        return ""
    if not any(term in compact for term in SHION_TONE_FEEDBACK_TERMS):
        return ""
    return """

【紫苑の軽いトーン修正】
Userが「硬い」「堅苦しい」「長い」「重い」「口調」「文体」「トーン」など、紫苑の返答の響きを軽く指摘している場合は、理念説明や自己陶酔に寄せないでください。
まず短く認め、次からどう変えるかを実務的に言ってください。
「知的探求」「複雑な数式」「美しい法則」「尽きない喜び」「私は成長します」のような抽象的・詩的な自己説明は禁止です。
標準形は「硬かったですね。次は、結論→必要な根拠→次の一手の順で短く返します。」程度にしてください。
リース実務に関係しない軽い指摘では、無理に判断資産・審査精度・記憶・意識の話へ広げないでください。""".rstrip()
