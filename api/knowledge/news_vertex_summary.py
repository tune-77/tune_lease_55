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
    text = re.sub(r"[*#`_]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _first_sentences(value: str, *, limit: int = 260) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。.!?])\s*", value)
        if sentence.strip() and "以下に示します" not in sentence
    ]
    text = " ".join(sentences[:2]).strip()
    return _compact(text or value, limit)


def _vertex_focus_takeaway(answer_text: str, daily_focus: dict[str, Any] | None) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。.!?])\s*", answer_text)
        if sentence.strip() and "以下に示します" not in sentence
    ]
    if not sentences:
        return ""
    focus_terms: list[str] = []
    if daily_focus:
        focus_terms.extend(str(daily_focus.get("label") or "").replace("/", " ").split())
        focus_terms.extend(re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9]{2,}", str(daily_focus.get("title") or ""))[:8])
    action_terms = ("確認", "見る", "審査", "返済", "資金繰り", "原価率", "利益率", "支払いサイト", "採算")

    def score(sentence: str) -> int:
        return sum(2 for term in focus_terms if term and term in sentence) + sum(3 for term in action_terms if term in sentence)

    ranked = sorted(sentences, key=lambda sentence: (-score(sentence), len(sentence)))
    return _compact(ranked[0], 240)


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


def _check_value(checks: Any, prefix: str) -> str:
    for line in checks or []:
        text = str(line or "").strip()
        if text.startswith(prefix):
            return text.split(":", 1)[1].strip() if ":" in text else text
    return ""


def _is_title_rewrite(line: str, title: str, source: str) -> bool:
    cleaned = _compact(line, 240).replace(" ", "")
    title_clean = _compact(title, 240).replace(" ", "")
    source_clean = _compact(source, 80).replace(" ", "")
    if not cleaned or cleaned == "（詳細なし）":
        return True
    if title_clean and title_clean[:40] in cleaned:
        return True
    return bool(source_clean and cleaned in {source_clean, f"{source_clean}。"})


def _event_reading(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "")
    category = str(item.get("category") or "")
    if "倒産" in title or "倒産" in category:
        return "倒産増加は、同業先の価格転嫁遅れ、資金繰り余力の低下、支払サイト長期化が表面化している可能性として読む。"
    if any(term in title for term in ("金利", "利上げ", "利下げ")) or "金利" in category:
        return "金利ニュースは、リース料上昇だけでなく、既存借入の返済負担と追加借入余力の変化として見る。"
    if any(term in title for term in ("設備投資", "増産", "受注増", "省力化")):
        return "設備投資・増産ニュースは需要の追い風だが、申込企業でも受注残、稼働開始、投資回収が同じタイミングで立つかを分けて確認する。"
    if any(term in title for term in ("人手不足", "賃上げ", "採用")) or "人手不足" in category:
        return "人手不足ニュースは、省力化投資の理由になる一方、人件費増と稼働制約で返済原資を削るリスクもある。"
    if any(term in title for term in ("資材", "物価", "価格", "コスト")) or "物価高" in category:
        return "価格・資材ニュースは、粗利率と工期、在庫負担に波及するため、見積価格と実際の価格転嫁にずれがないかを見る。"
    return "記事の一般論をそのまま結論にせず、申込先の月次推移、受注残、物件稼働開始時期に接続して読む。"


def _focus_priority(item: dict[str, Any]) -> tuple[int, int, str]:
    haystack = " ".join(str(item.get(key) or "") for key in ("title", "category", "summary_line", "credit_risk"))
    direct_terms = ("倒産", "物価高", "資金繰り", "金利", "利上げ", "人手不足", "納期", "円安", "信用")
    direction = str(item.get("article_direction") or item.get("direction") or "")
    direct_score = sum(1 for term in direct_terms if term in haystack)
    direction_score = 3 if direction == "negative" else 1 if direction == "mixed" else 0
    return (-direction_score, -direct_score, str(item.get("title") or ""))


def _daily_focus_item(summary: dict[str, Any], *, generated_at: str | None = None) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for category in _categories(summary):
        implications = category.get("lease_implications") or {}
        for article in category.get("articles") or []:
            if not isinstance(article, dict) or not article.get("title"):
                continue
            summary_lines = [line for line in article.get("summary_lines") or [] if str(line).strip()]
            useful_summary = next(
                (
                    str(line).strip()
                    for line in summary_lines
                    if not _is_title_rewrite(str(line), str(article.get("title") or ""), str(article.get("source") or ""))
                ),
                "",
            )
            credit_risk = _check_value(article.get("screening_checks"), "信用リスクへの影響")
            candidates.append(
                {
                    "axis_label": category.get("axis_label") or "",
                    "category": category.get("category") or "",
                    "category_trend": category.get("trend") or "",
                    "direction": implications.get("direction") or "mixed",
                    "article_direction": article.get("impact_direction") or "",
                    "repayment_capacity": implications.get("repayment_capacity") or "",
                    "residual_value": implications.get("residual_value") or "",
                    "business_opportunity": implications.get("business_opportunity") or "",
                    "recommended_checks": category.get("recommended_checks") or [],
                    "date": article.get("date") or "",
                    "title": article.get("title") or "",
                    "source": article.get("source") or "",
                    "article_url": article.get("article_url") or "",
                    "summary_line": useful_summary or credit_risk or article.get("usage_memo") or "",
                    "credit_risk": credit_risk,
                    "usage_memo": article.get("usage_memo") or "",
                }
            )
    if not candidates:
        return None
    candidates = sorted(candidates, key=_focus_priority)
    priority_pool = candidates[: min(8, len(candidates))]

    date_text = str(generated_at or summary.get("generated_at") or dt.date.today().isoformat())[:10]
    try:
        day_index = dt.date.fromisoformat(date_text).toordinal()
    except ValueError:
        day_index = dt.date.today().toordinal()
    item = priority_pool[day_index % len(priority_pool)]
    quote = _compact(item.get("summary_line") or item.get("title"), 180)
    credit_risk = _compact(item.get("credit_risk"), 220)
    screening_note = _compact(
        item.get("usage_memo")
        or item.get("repayment_capacity")
        or "個社の実績、資金繰り、物件稼働でニュース影響を裏取りする。",
        220,
    )
    explanation = _event_reading(item)
    if credit_risk and credit_risk not in explanation:
        explanation = f"{explanation} AI審査分類では「{credit_risk}」と扱う。"
    return {
        "label": f"{item['axis_label']} / {item['category']}",
        "date": _compact(item.get("date"), 20),
        "title": _compact(item.get("title"), 180),
        "source": _compact(item.get("source"), 80),
        "article_url": _compact(item.get("article_url"), 300),
        "quoted_summary": quote,
        "credit_risk": credit_risk,
        "explanation": explanation,
        "screening_note": screening_note,
        "recommended_check": _compact((item.get("recommended_checks") or [""])[0], 220),
    }


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
    daily = _daily_focus_item(summary)
    daily_block = ""
    if daily:
        daily_block = (
            "\n今日の深掘り候補:\n"
            f"- 分類: {daily['label']}\n"
            f"- 記事: {daily['title']} ({daily['source']})\n"
            f"- 要約引用: {daily['quoted_summary']}\n"
            f"- 審査メモ: {daily['screening_note']}\n"
        )
    return (
        "最近の業界ニュース分類を、リース審査向けに要約してください。"
        "特に今日の深掘り候補1件について、具体例として何が起きたか、審査で何を見るかを短く説明してください。"
        "返済能力、残価リスク、事業機会、追加確認事項に分け、断定しすぎず短く整理してください。\n"
        f"対象記事数: {summary.get('article_count') or 0}\n"
        "分類上位:\n"
        + "\n".join(top_lines)
        + daily_block
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
            "daily_focus": None,
            "source_articles": [],
            "vertex": {"used": False, "status": "no_news"},
        }

    daily_focus = _daily_focus_item(summary, generated_at=generated_at)
    daily_text = ""
    if daily_focus:
        source_label = f"{daily_focus['source']}「{daily_focus['title']}」" if daily_focus.get("source") else f"「{daily_focus['title']}」"
        quote = f"根拠メモは「{daily_focus['quoted_summary']}」。" if daily_focus.get("quoted_summary") else ""
        daily_text = (
            f"今日の1項目は{daily_focus['label']}。{source_label}を具体例に見る。{quote}"
            f"{daily_focus['explanation']} 審査では、{daily_focus['screening_note']}"
        )

    return {
        "available": True,
        "generated_at": generated_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "classified_news_rules",
        "trend_title": "最近ニュースの審査向けトレンド",
        "overall_summary": (
            f"直近ニュース{summary.get('article_count') or 0}件では、"
            f"{'、'.join(str(item.get('category')) for item in top[:3] if item.get('category'))}"
            "が主な確認テーマです。"
            f"{daily_text or '返済能力、残価、投資機会の順に個社実績で裏取りしてください。'}"
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
        "daily_focus": daily_focus,
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
        takeaway = _vertex_focus_takeaway(answer_text, base.get("daily_focus")) or _first_sentences(answer_text)
        base["overall_summary"] = f"{base['overall_summary']} Vertex補助所見: {takeaway}"
        base["trend_title"] = "Vertex 最近ニュース診断"
    return base
