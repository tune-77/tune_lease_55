"""Vertex-assisted trend summary for lease news.

The classified news summary remains the primary source. Vertex AI Search is
used as supplementary lease-domain knowledge so the dashboard can explain why
recent news matters for screening without turning external retrieval into a
hard dependency.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Callable


def _compact(value: Any, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _plain_vertex_text(value: Any, limit: int = 1200) -> str:
    text = str(value or "")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _first_sentences(value: str, *, limit: int = 260) -> str:
    sentences = re.split(r"(?<=[。.!?])\s*", value)
    text = " ".join(sentence for sentence in sentences[:2] if sentence).strip()
    return _compact(text or value, limit)


def _categories(summary: dict[str, Any]) -> list[dict[str, Any]]:
    categories: list[dict[str, Any]] = []
    for axis in summary.get("axes") or []:
        if not isinstance(axis, dict):
            continue
        for category in axis.get("categories") or []:
            if not isinstance(category, dict):
                continue
            categories.append(category)
    categories.sort(key=lambda item: (-int(item.get("article_count") or 0), item.get("axis_label") or "", item.get("category") or ""))
    return categories


def _top_articles(summary: dict[str, Any], *, limit: int = 6) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    seen: set[str] = set()
    for category in _categories(summary):
        for article in category.get("articles") or []:
            if not isinstance(article, dict):
                continue
            key = _compact(article.get("article_url") or article.get("title") or article.get("file_path"), 240)
            if not key or key in seen:
                continue
            seen.add(key)
            articles.append(
                {
                    "date": _compact(article.get("date"), 20),
                    "title": _compact(article.get("title"), 160),
                    "source": _compact(article.get("source"), 80),
                    "article_url": _compact(article.get("article_url"), 300),
                }
            )
            if len(articles) >= limit:
                return articles
    return articles


def _direction_label(direction: str) -> str:
    return {"negative": "警戒", "positive": "追い風", "mixed": "濃淡あり"}.get(direction, "要確認")


def build_news_trend_prompt(summary: dict[str, Any]) -> str:
    """Build a compact Vertex query from already-classified recent news."""
    top_lines = []
    for category in _categories(summary)[:8]:
        implications = category.get("lease_implications") or {}
        top_lines.append(
            "- "
            f"{category.get('axis_label')}/{category.get('category')}: "
            f"{category.get('article_count')}件, "
            f"{_direction_label(str(implications.get('direction') or 'mixed'))}, "
            f"{_compact(category.get('trend'), 180)}"
        )
    return (
        "最近の業界ニュース分類を、リース審査向けに要約してください。"
        "返済能力、残価リスク、事業機会、追加確認事項に分け、断定しすぎず短く整理してください。\n"
        f"対象記事数: {summary.get('article_count') or 0}\n"
        "分類上位:\n"
        + "\n".join(top_lines)
    )


def _fallback_trend_summary(summary: dict[str, Any], *, generated_at: str | None = None) -> dict[str, Any]:
    categories = _categories(summary)
    top = categories[:5]
    negative = [
        item
        for item in categories
        if (item.get("lease_implications") or {}).get("direction") == "negative"
    ]
    watch = negative[:3] or top[:3]

    trend_lines = []
    for item in top[:4]:
        implications = item.get("lease_implications") or {}
        trend_lines.append(
            f"{item.get('axis_label')}の「{item.get('category')}」が{item.get('article_count')}件。"
            f"{_direction_label(str(implications.get('direction') or 'mixed'))}として、{_compact(item.get('trend'), 120)}"
        )

    caution_points = []
    for item in watch:
        for line in item.get("recommended_checks") or []:
            text = _compact(line, 180)
            if text and text not in caution_points:
                caution_points.append(text)
            if len(caution_points) >= 5:
                break
        if len(caution_points) >= 5:
            break
    if not caution_points and summary.get("available"):
        caution_points = [
            "ニュース単体で判断を変えず、個社の月次実績・資金繰り・受注残で裏取りする。",
            "投資効果は回収期間と稼働開始時期を分けて確認する。",
        ]

    screening_actions = [
        "影響が出ている業種では、直近3か月の売上・粗利・資金繰り予定を確認する。",
        "物件の中古流通性、納期、稼働開始時期を残価リスクの補正材料にする。",
        "金利・為替・倒産件数のニュースは、リース料負担と借入余力への感応度として見る。",
    ]

    if not summary.get("available"):
        return {
            "available": False,
            "generated_at": generated_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source": "rule_fallback",
            "trend_title": "最近ニュースの分類データがまだありません",
            "overall_summary": "ニュース収集後に、業種別・社会情勢・金融情報の傾向をここに表示します。",
            "trend_lines": [],
            "caution_points": [],
            "screening_actions": [],
            "watch_categories": [],
            "source_articles": [],
            "vertex": {"used": False, "status": "no_news"},
        }

    return {
        "available": True,
        "generated_at": generated_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "classified_news_rules",
        "trend_title": "最近ニュースの審査向けトレンド",
        "overall_summary": (
            f"直近ニュース{summary.get('article_count') or 0}件では、"
            f"{'、'.join(str(item.get('category')) for item in top[:3] if item.get('category'))}"
            "が主な確認テーマです。返済能力、残価、投資機会の順に個社実績で裏取りしてください。"
        ),
        "trend_lines": trend_lines,
        "caution_points": caution_points[:5],
        "screening_actions": screening_actions,
        "watch_categories": [
            {
                "label": f"{item.get('axis_label')} / {item.get('category')}",
                "count": int(item.get("article_count") or 0),
                "reason": _compact((item.get("lease_implications") or {}).get("repayment_capacity") or item.get("trend"), 180),
            }
            for item in watch
        ],
        "source_articles": _top_articles(summary),
        "vertex": {"used": False, "status": "not_requested"},
    }


def _vertex_refs(result: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for item in result.get("search_results") or result.get("results") or []:
        if not isinstance(item, dict):
            continue
        title = _compact(item.get("title") or item.get("uri") or item.get("source_path"), 160)
        source = _compact(item.get("uri") or item.get("source_path") or item.get("link"), 260)
        snippet = _compact(item.get("snippet") or item.get("excerpt"), 260)
        if title or source or snippet:
            refs.append({"title": title, "source": source, "snippet": snippet})
    if refs:
        return refs[:5]
    for ref in result.get("refs") or []:
        text = _compact(ref, 260)
        if text:
            refs.append({"title": text, "source": text, "snippet": ""})
        if len(refs) >= 5:
            break
    return refs


def build_vertex_assisted_news_trend_summary(
    summary: dict[str, Any],
    *,
    use_vertex: bool = True,
    answer_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return recent-news trend/summary/cautions, enriched by Vertex if possible."""
    base = _fallback_trend_summary(summary)
    if not use_vertex or not summary.get("available"):
        base["vertex"] = {"used": False, "status": "disabled" if not use_vertex else "no_news"}
        return base

    if answer_fn is None:
        from api.vertex_agent_search import answer_vertex_agent

        answer_fn = answer_vertex_agent

    prompt = build_news_trend_prompt(summary)
    preamble = (
        "あなたはリース審査のニュース分析補助AIです。入力された分類済みニュースを主材料に、"
        "Vertex AI Search上のリース審査ナレッジを補助根拠として使う。"
        "返済能力、残価リスク、事業機会、注意点を日本語で短く整理し、断定しすぎない。"
    )
    try:
        vertex = answer_fn(
            prompt,
            page_size=5,
            preamble=preamble,
            include_related_questions=False,
            include_grounding_supports=True,
            max_rephrase_steps=2,
        )
    except Exception as exc:  # noqa: BLE001 - dashboard must keep rendering
        vertex = {"used": False, "status": f"error: {type(exc).__name__}", "answer_text": "", "refs": []}

    answer_text = _plain_vertex_text(vertex.get("answer_text"), 1200)
    base["source"] = "vertex_ai_search" if vertex.get("used") and answer_text else base["source"]
    base["vertex"] = {
        "used": bool(vertex.get("used")),
        "status": _compact(vertex.get("status"), 80),
        "answer_text": answer_text,
        "grounding_score": vertex.get("grounding_score"),
        "low_support_claim_count": vertex.get("low_support_claim_count"),
        "refs": _vertex_refs(vertex),
    }
    if answer_text:
        base["overall_summary"] = f"{base['overall_summary']} Vertex補助所見: {_first_sentences(answer_text)}"
        base["trend_title"] = "Vertex 最近ニュース診断"
    return base
