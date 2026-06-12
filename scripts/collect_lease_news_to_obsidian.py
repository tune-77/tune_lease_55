#!/usr/bin/env python3
"""Collect daily lease-related news and write a digest into iCloud 上の Obsidian Vault.

The collector uses Google News RSS search queries by default, deduplicates
articles, summarizes them into a compact Markdown note, and optionally appends
a short digest to the Daily note.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lease_news_digest import (
    get_lease_news_metrics,
    record_lease_news_collection,
    write_lease_news_focus_note,
    write_lease_news_reflection_note,
)


DEFAULT_VAULT_CANDIDATES = [
    Path(os.environ.get("OBSIDIAN_VAULT", "")).expanduser() if os.environ.get("OBSIDIAN_VAULT") else None,
    Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault",
    Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Obsidian Vault",
]

QUERY_SETS: dict[str, tuple[str, ...]] = {
    "lease_core": (
        "リース",
        "オペレーティングリース",
        "設備投資 リース",
    ),
    "finance_policy": (
        "リース 金利",
        "リース 審査",
        "リース 税制",
        "補助金 設備投資 リース",
    ),
    "industry_transport": (
        "物流 リース",
        "車両 リース",
        "トラック リース",
        "EV リース",
    ),
    "industry_construction": (
        "建設 リース",
        "建機 リース",
        "不動産 設備投資",
        "工場 建設 リース",
    ),
    "industry_manufacturing": (
        "製造 リース",
        "省力化投資 リース",
        "工場 自動化 リース",
        "ロボット リース",
    ),
}

QUERY_SET_LABELS: dict[str, str] = {
    "lease_core": "基礎",
    "finance_policy": "金融・政策",
    "industry_transport": "物流・車両",
    "industry_construction": "建設・不動産",
    "industry_manufacturing": "製造・DX",
}

RSS_FEEDS: dict[str, tuple[tuple[str, str], ...]] = {
    "official": (
        ("METI Latest Information", "https://www.meti.go.jp/ml_index_en_atom.xml"),
        ("METI Statistics Latest Information", "https://www.meti.go.jp/english/statistics/st_en_news.xml"),
        ("FSA Updated News", "https://www.fsa.go.jp/fsaEnNewsList_rss2.xml"),
        ("MLIT Press Release", "https://www.mlit.go.jp/pressrelease.rdf"),
        ("MLIT Latest Information", "https://www.mlit.go.jp/index.rdf"),
    ),
}

SOURCE_WEIGHTS = {
    "official": 3,
    "google": 2,
    "custom": 1,
}

PROFILE_DEFINITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "lease-core": {
        "query_sets": ("lease_core", "finance_policy"),
        "rss_sets": ("official",),
    },
    "industry-watch": {
        "query_sets": ("lease_core", "finance_policy", "industry_transport", "industry_construction", "industry_manufacturing"),
        "rss_sets": ("official",),
    },
    "policy-watch": {
        "query_sets": ("lease_core", "finance_policy"),
        "rss_sets": ("official",),
    },
    "all": {
        "query_sets": tuple(QUERY_SETS.keys()),
        "rss_sets": ("official",),
    },
}

DEFAULT_NEWS_DIR = "05-クリップ_記事/リースニュース"
DEFAULT_DAILY_DIR = "Daily"
DEFAULT_PROFILE = "industry-watch"

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "金利・政策": ("金利", "政策", "日銀", "利上げ", "利下げ", "為替"),
    "設備投資": ("設備投資", "投資", "生産性", "省力化", "更新"),
    "物流・車両": ("物流", "運送", "車両", "配送", "トラック", "EV"),
    "建設・不動産": ("建設", "工事", "不動産", "住宅", "工場"),
    "製造・DX": ("製造", "工場", "DX", "AI", "自動化", "ロボット", "半導体"),
    "金融・与信": ("与信", "融資", "審査", "貸出", "クレジット", "リース"),
    "法令・制度": ("法改正", "制度", "省庁", "税制", "補助金", "助成金"),
}


@dataclass
class Article:
    title: str
    link: str
    source: str
    published: dt.datetime | None
    summary: str
    query: str
    bucket: str = "general"
    source_kind: str = "google"
    source_weight: int = 1
    theme: str = "その他"
    tags: tuple[str, ...] = ()
    score: int = 0
    raw_source: str = ""
    industries: tuple[str, ...] = ()
    lease_assets: tuple[str, ...] = ()
    credit_risk_impact: str = ""
    screening_checks: tuple[str, ...] = ()
    impact_direction: str = "neutral"
    classification_confidence: float = 0.0
    source_reliability: str = "medium"
    valid_until: str = ""
    canonical_topic: str = ""
    classification_source: str = "rule"

    @property
    def published_iso(self) -> str:
        if not self.published:
            return ""
        return self.published.astimezone().isoformat(timespec="minutes")


def _candidate_vaults() -> list[Path]:
    out: list[Path] = []
    for path in DEFAULT_VAULT_CANDIDATES:
        if path and path.exists() and path.is_dir():
            out.append(path)
    return out


def find_vault(override: str | None = None) -> Path:
    if override:
        path = Path(override).expanduser()
        if path.exists() and path.is_dir():
            return path
        raise FileNotFoundError(f"iCloud 上の Obsidian Vault が見つかりません: {path}")
    candidates = _candidate_vaults()
    if not candidates:
        raise FileNotFoundError("iCloud 上の Obsidian Vault が見つかりません。OBSIDIAN_VAULT を設定するか --vault を指定してください。")
    return candidates[0]


def _safe_note_path(vault: Path, rel: str, *, create_parent: bool = True) -> Path:
    target = (vault / rel).expanduser().resolve()
    if vault not in target.parents and target != vault:
        raise ValueError("refusing to write outside the Obsidian vault")
    if target.suffix.lower() != ".md":
        target = target.with_suffix(".md")
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _append_text(path: Path, text: str) -> None:
    body = text.strip()
    if not body:
        return
    prefix = "\n\n" if path.exists() and path.read_text(encoding="utf-8").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + body + "\n")


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", html.unescape(text or "")).strip()


def _yaml_string(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _parse_published(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    except Exception:
        return None


def _match_theme(text: str) -> tuple[str, int]:
    combined = _normalize_text(text).lower()
    best_theme = "その他"
    best_score = 0
    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in combined)
        if score > best_score:
            best_theme = theme
            best_score = score
    return best_theme, best_score


def _extract_tags(text: str) -> tuple[str, ...]:
    combined = _normalize_text(text).lower()
    tag_map = [
        ("金利", ("金利", "利上げ", "利下げ", "政策", "日銀", "為替")),
        ("設備投資", ("設備投資", "投資", "更新", "省力化", "補助金", "助成金")),
        ("与信", ("与信", "審査", "融資", "貸出", "クレジット")),
        ("法令", ("法令", "制度", "税制", "規制", "法改正")),
        ("物流/車両", ("物流", "車両", "トラック", "配送", "運送", "EV")),
        ("建設/不動産", ("建設", "工事", "不動産", "住宅", "工場")),
        ("製造/DX", ("製造", "DX", "自動化", "ロボット", "AI", "半導体")),
    ]
    tags: list[str] = []
    for tag, keywords in tag_map:
        if any(kw.lower() in combined for kw in keywords):
            tags.append(tag)
    if not tags:
        tags.append("その他")
    return tuple(tags)


def _safe_filename(text: str, max_len: int = 30) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", text)
    cleaned = cleaned.strip("_").strip()
    return cleaned[:max_len] if cleaned else "記事"


def _canonical_url(url: str) -> str:
    value = _normalize_text(url)
    if not value:
        return ""
    try:
        parsed = urllib.parse.urlsplit(value)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        filtered = [
            (key, val)
            for key, val in query
            if not key.lower().startswith("utm_")
            and key.lower() not in {"gclid", "fbclid", "yclid", "mc_cid", "mc_eid"}
        ]
        path = parsed.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
                urllib.parse.urlencode(filtered, doseq=True),
                "",
            )
        )
    except Exception:
        return value


def _normalized_title(title: str, source: str = "") -> str:
    value = _normalize_text(title).lower()
    source_value = _normalize_text(source).lower()
    if source_value:
        value = re.sub(rf"\s*[-｜|]\s*{re.escape(source_value)}\s*$", "", value)
    value = re.sub(r"\s*[-｜|]\s*[^-｜|]{2,30}$", "", value)
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def _canonical_topic_from_title(article: Article) -> str:
    title = _normalize_text(article.title)
    source = _normalize_text(article.source)
    if source:
        title = re.sub(rf"\s*[-｜|]\s*{re.escape(source)}\s*$", "", title, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", title).strip()[:120]


def _default_valid_until(article: Article, base_date: dt.date | None = None) -> str:
    base = base_date or (article.published.date() if article.published else dt.date.today())
    text = " ".join([article.title, article.summary, article.theme, *article.tags])
    deadline = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", text)
    if deadline and any(term in text for term in ("期限", "締切", "募集", "施行")):
        try:
            return dt.date(int(deadline.group(1)), int(deadline.group(2)), int(deadline.group(3))).isoformat()
        except ValueError:
            pass
    if "金利" in article.tags:
        days = 60
    elif "法令" in article.tags:
        days = 365
    elif any(term in text for term in ("補助金", "助成金", "公募")):
        days = 180
    elif article.theme in {"設備投資", "製造・DX"}:
        days = 365
    else:
        days = 180
    return (base + dt.timedelta(days=days)).isoformat()


def _rule_classification(article: Article) -> dict[str, Any]:
    tag_industries = {
        "物流/車両": ["運送業", "物流業"],
        "建設/不動産": ["建設業", "不動産業"],
        "製造/DX": ["製造業"],
        "金融・与信": ["金融業"],
    }
    tag_assets = {
        "物流/車両": ["車両", "物流設備"],
        "建設/不動産": ["建設機械", "建物附属設備"],
        "製造/DX": ["生産設備", "IT・DX設備"],
        "設備投資": ["設備機器"],
    }
    industries: list[str] = []
    assets: list[str] = []
    for tag in article.tags:
        industries.extend(tag_industries.get(tag, []))
        assets.extend(tag_assets.get(tag, []))
    if article.theme == "金融・与信":
        industries.append("全業種")

    negative_terms = ("倒産", "赤字", "減益", "破産", "延滞", "悪化", "縮小", "停止", "不正", "違反")
    positive_terms = ("増益", "成長", "回復", "支援", "補助金", "効率化", "省力化", "拡大", "改善")
    text = " ".join([article.title, article.summary])
    negative = sum(term in text for term in negative_terms)
    positive = sum(term in text for term in positive_terms)
    direction = "negative" if negative > positive else "positive" if positive > negative else "neutral"

    checks = [_review_line(article.tags, article.theme)]
    if direction == "negative":
        checks.append("ニュース記載の悪化要因が対象企業の資金繰りと返済余力へ波及していないか確認する。")
    elif direction == "positive":
        checks.append("改善効果が一時的な期待ではなく、受注・利益・キャッシュフローに反映される時期を確認する。")

    reliability = "high" if article.source_kind == "official" else "medium"
    impact = {
        "positive": "業績・投資回収・返済余力を改善する可能性がある。",
        "negative": "業績・資金繰り・返済余力を悪化させる可能性がある。",
        "neutral": "直接的な信用リスクへの影響は限定的だが、関連業界の前提条件として確認する。",
    }[direction]
    return {
        "industries": list(dict.fromkeys(industries)) or ["全業種"],
        "lease_assets": list(dict.fromkeys(assets)) or ["対象物件未特定"],
        "credit_risk_impact": impact,
        "screening_checks": list(dict.fromkeys(checks)),
        "impact_direction": direction,
        "classification_confidence": 0.45,
        "source_reliability": reliability,
        "valid_until": _default_valid_until(article),
        "canonical_topic": _canonical_topic_from_title(article),
        "classification_source": "rule",
    }


def _get_gemini_key() -> str:
    try:
        from secret_manager import get_gemini_api_key

        value = get_gemini_api_key()
        return value.strip() if isinstance(value, str) else ""
    except Exception:
        value = os.environ.get("GEMINI_API_KEY", "")
        return value.strip() if isinstance(value, str) else ""


def _as_string_tuple(value: Any, limit: int = 8) -> tuple[str, ...]:
    if isinstance(value, str):
        values = re.split(r"[,、\n]", value)
    elif isinstance(value, list):
        values = value
    else:
        values = []
    cleaned = [_normalize_text(str(item))[:160] for item in values if _normalize_text(str(item))]
    return tuple(dict.fromkeys(cleaned))[:limit]


def _normalize_classification(article: Article, value: dict[str, Any] | None) -> dict[str, Any]:
    fallback = _rule_classification(article)
    value = value or {}
    direction = str(value.get("impact_direction") or fallback["impact_direction"]).strip().lower()
    if direction not in {"positive", "negative", "neutral", "mixed"}:
        direction = fallback["impact_direction"]
    reliability = str(value.get("source_reliability") or fallback["source_reliability"]).strip().lower()
    if reliability not in {"high", "medium", "low"}:
        reliability = fallback["source_reliability"]
    try:
        confidence = max(0.0, min(1.0, float(value.get("classification_confidence", fallback["classification_confidence"]))))
    except (TypeError, ValueError):
        confidence = fallback["classification_confidence"]
    valid_until = str(value.get("valid_until") or fallback["valid_until"]).strip()
    try:
        dt.date.fromisoformat(valid_until)
    except ValueError:
        valid_until = fallback["valid_until"]
    return {
        "industries": _as_string_tuple(value.get("industries")) or tuple(fallback["industries"]),
        "lease_assets": _as_string_tuple(value.get("lease_assets")) or tuple(fallback["lease_assets"]),
        "credit_risk_impact": _normalize_text(str(value.get("credit_risk_impact") or fallback["credit_risk_impact"]))[:500],
        "screening_checks": _as_string_tuple(value.get("screening_checks"), limit=6) or tuple(fallback["screening_checks"]),
        "impact_direction": direction,
        "classification_confidence": confidence,
        "source_reliability": reliability,
        "valid_until": valid_until,
        "canonical_topic": _normalize_text(str(value.get("canonical_topic") or fallback["canonical_topic"]))[:120],
        "classification_source": str(value.get("classification_source") or "gemini"),
    }


def _apply_classification(article: Article, value: dict[str, Any] | None) -> None:
    normalized = _normalize_classification(article, value)
    article.industries = normalized["industries"]
    article.lease_assets = normalized["lease_assets"]
    article.credit_risk_impact = normalized["credit_risk_impact"]
    article.screening_checks = normalized["screening_checks"]
    article.impact_direction = normalized["impact_direction"]
    article.classification_confidence = normalized["classification_confidence"]
    article.source_reliability = normalized["source_reliability"]
    article.valid_until = normalized["valid_until"]
    article.canonical_topic = normalized["canonical_topic"]
    article.classification_source = normalized["classification_source"]


def classify_articles(articles: list[Article], use_ai: bool = True) -> None:
    for article in articles:
        _apply_classification(article, _rule_classification(article))
    api_key = _get_gemini_key() if use_ai else ""
    if not api_key or not articles:
        return
    try:
        from google import genai
        from google.genai import types

        payload = [
            {
                "article_index": index,
                "title": article.title,
                "summary": article.summary,
                "source": article.source,
                "published": article.published_iso,
                "query": article.query,
                "rule_tags": list(article.tags),
            }
            for index, article in enumerate(articles)
        ]
        prompt = (
            "以下のリース関連ニュースを、リース審査で後から検索・再利用できるよう分類してください。"
            "記事にない事実は推測せず、不明な対象物件は「対象物件未特定」としてください。"
            "impact_directionはpositive/negative/neutral/mixed、source_reliabilityはhigh/medium/low、"
            "valid_untilはYYYY-MM-DDです。法令は原則1年、市況は60日、企業ニュースは180日、"
            "設備・技術動向は1年を目安にしてください。canonical_topicは同一事象を束ねられる短い名称です。\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=6000,
                response_mime_type="application/json",
                response_json_schema={
                    "type": "object",
                    "properties": {
                        "classifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "article_index": {"type": "integer"},
                                    "industries": {"type": "array", "items": {"type": "string"}},
                                    "lease_assets": {"type": "array", "items": {"type": "string"}},
                                    "credit_risk_impact": {"type": "string"},
                                    "screening_checks": {"type": "array", "items": {"type": "string"}},
                                    "impact_direction": {"type": "string"},
                                    "classification_confidence": {"type": "number"},
                                    "source_reliability": {"type": "string"},
                                    "valid_until": {"type": "string"},
                                    "canonical_topic": {"type": "string"},
                                },
                                "required": [
                                    "article_index",
                                    "industries",
                                    "lease_assets",
                                    "credit_risk_impact",
                                    "screening_checks",
                                    "impact_direction",
                                    "classification_confidence",
                                    "source_reliability",
                                    "valid_until",
                                    "canonical_topic",
                                ],
                            },
                        }
                    },
                    "required": ["classifications"],
                },
                http_options=types.HttpOptions(timeout=30000),
            ),
        )
        parsed = json.loads(response.text or "{}")
        for item in parsed.get("classifications", []):
            index = int(item.get("article_index", -1))
            if 0 <= index < len(articles):
                item["classification_source"] = "gemini"
                _apply_classification(articles[index], item)
    except Exception as exc:
        print(f"[news] Gemini classification skipped; rule fallback used: {exc}", file=sys.stderr)


def _infer_region(article: "Article") -> str:
    text = " ".join([article.title, article.source, article.raw_source, article.query]).lower()
    foreign_sources = ("reuters", "bloomberg", "wsj", "ft.com", "cnbc", "bbc", "ap ", "afp")
    us_keywords = ("fed", "federal reserve", "us ", "united states", "wall street", "sec ", "nyse")
    eu_keywords = ("ecb", "european", "euro", "deutschland", "boe ", "bank of england")
    asia_ex_jp = ("china", "korea", "taiwan", "asean", "singapore", "hong kong", "india", "southeast asia")
    if any(kw in text for kw in us_keywords):
        return "米国"
    if any(kw in text for kw in eu_keywords):
        return "欧州"
    if any(kw in text for kw in asia_ex_jp):
        return "アジア"
    if any(kw in text for kw in foreign_sources):
        return "米国"
    return "国内"


def _infer_importance(article: "Article") -> str:
    if article.source_kind == "official" and article.score >= 1:
        return "高"
    if article.score >= 2 or "法令" in article.tags or "金利" in article.tags:
        return "高"
    if article.score >= 1:
        return "中"
    return "低"


def _review_line(tags: tuple[str, ...], theme: str) -> str:
    joined = " / ".join(tags)
    if "金利" in tags:
        return "提示金利・競合提示金利差・基準金利の説明を厚くする。"
    if "設備投資" in tags:
        return "リース期間・中古価値・再リース余地を確認する。"
    if "与信" in tags:
        return "取引区分・与信枠・返済余力を再確認する。"
    if "法令" in tags:
        return "契約条件・税制・補助金要件の再確認を優先する。"
    if "物流/車両" in tags:
        return "稼働率・車両更新・保守費用の見通しを確認する。"
    if "建設/不動産" in tags:
        return "工期・更新投資・移設可能性・現場稼働への影響を確認する。"
    if "製造/DX" in tags:
        return "更新投資の回収期間と生産性改善効果を確認する。"
    return f"{theme}の変化を踏まえ、提示条件と審査コメントを更新する。"


def _google_news_feed_url(query: str) -> str:
    return GOOGLE_NEWS_RSS.format(query=urllib.parse.quote(query))


def _expand_profile(profile: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    normalized = (profile or DEFAULT_PROFILE).strip().lower()
    if normalized not in PROFILE_DEFINITIONS:
        normalized = DEFAULT_PROFILE
    profile_def = PROFILE_DEFINITIONS[normalized]
    queries: list[tuple[str, str]] = []
    for set_name in profile_def["query_sets"]:
        label = QUERY_SET_LABELS.get(set_name, set_name)
        for query in QUERY_SETS.get(set_name, ()):
            queries.append((label, query))
    feeds: list[tuple[str, str, str]] = []
    for feed_set in profile_def["rss_sets"]:
        for feed_name, feed_url in RSS_FEEDS.get(feed_set, ()):
            feeds.append((feed_set, feed_name, feed_url))
    return queries, feeds


def _fetch_feed(url: str, query: str, source_label: str, bucket: str) -> list[Article]:
    headers = {"User-Agent": "Mozilla/5.0 (lease-news-bot)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items: list[Article] = []
    for item in channel.findall("item"):
        title = _normalize_text(item.findtext("title") or "")
        link = _normalize_text(item.findtext("link") or "")
        description = _normalize_text(_strip_tags(item.findtext("description") or ""))
        source = _normalize_text(item.findtext("source") or source_label or "Google News")
        published = _parse_published(item.findtext("pubDate"))
        text_blob = " ".join([title, description, source, query])
        theme, score = _match_theme(text_blob)
        tags = _extract_tags(text_blob)
        source_kind = "official" if source_label != "Google News RSS" and source_label != "Google News" else "google"
        source_weight = SOURCE_WEIGHTS.get(source_kind, 1)
        if source_kind == "google" and bucket in PROFILE_DEFINITIONS.get("all", {}).get("query_sets", ()):
            source_weight = 2
        items.append(
            Article(
                title=title,
                link=link,
                source=source,
                published=published,
                summary=description,
                query=query,
                bucket=bucket,
                source_kind=source_kind,
                source_weight=source_weight,
                theme=theme,
                tags=tags,
                score=score,
                raw_source=source_label,
            )
        )
    return items


def collect_articles(
    query_specs: Iterable[tuple[str, str]],
    feed_specs: Iterable[tuple[str, str, str]],
    per_query: int = 6,
    per_feed: int = 6,
) -> list[Article]:
    seen: set[str] = set()
    articles: list[Article] = []
    for bucket, query in query_specs:
        query = query.strip()
        if not query:
            continue
        url = _google_news_feed_url(query)
        try:
            fetched = _fetch_feed(url, query, "Google News RSS", bucket)
        except Exception as exc:
            print(f"[news] fetch failed for query={query}: {exc}", file=sys.stderr)
            continue
        for art in fetched[:per_query]:
            key = (art.link or art.title).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            articles.append(art)
    for bucket, feed_name, feed_url in feed_specs:
        try:
            fetched = _fetch_feed(feed_url, feed_name, feed_name, bucket)
        except Exception as exc:
            print(f"[news] fetch failed for feed={feed_name}: {exc}", file=sys.stderr)
            continue
        for art in fetched[:per_feed]:
            key = (art.link or art.title).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            articles.append(art)
    articles.sort(
        key=lambda a: (
            a.published or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
            a.source_weight,
            a.score,
            a.title,
        ),
        reverse=True,
    )
    return articles


def _theme_counts(articles: list[Article]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for art in articles:
        counts[art.theme] = counts.get(art.theme, 0) + 1
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _bucket_counts(articles: list[Article]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for art in articles:
        counts[art.bucket] = counts.get(art.bucket, 0) + 1
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _tag_counts(articles: list[Article]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for art in articles:
        for tag in art.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _source_counts(articles: list[Article]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for art in articles:
        counts[art.source_kind] = counts.get(art.source_kind, 0) + 1
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _build_article_content(
    article: Article,
    date_str: str,
    week: str,
    month: str,
    profile: str,
) -> str:
    region = _infer_region(article)
    importance = _infer_importance(article)
    tags_json = json.dumps(list(article.tags), ensure_ascii=False)
    pub = article.published_iso or ""
    review = _review_line(article.tags, article.theme)
    canonical_url = _canonical_url(article.link)
    dedup_id = hashlib.sha256(
        (canonical_url or _normalized_title(article.title, article.source)).encode("utf-8")
    ).hexdigest()[:16]
    summary_text = article.summary or ""
    lines: list[str] = []
    if summary_text:
        words = summary_text.split("。")
        s1 = words[0].strip() + ("。" if words[0].strip() else "")
        s2 = (words[1].strip() + "。") if len(words) > 1 and words[1].strip() else ""
        s3 = review
    else:
        s1 = article.title
        s2 = ""
        s3 = review

    content = f"""---
date: {date_str}
week: {week}
month: {month}
tags: {tags_json}
region: {region}
source: {_yaml_string(article.source or "Google News")}
importance: {importance}
profile: {profile}
industries: {_yaml_string(", ".join(article.industries))}
lease_assets: {_yaml_string(", ".join(article.lease_assets))}
impact_direction: {article.impact_direction}
source_reliability: {article.source_reliability}
classification_confidence: {article.classification_confidence:.2f}
valid_until: {article.valid_until}
canonical_topic: {_yaml_string(article.canonical_topic)}
canonical_url: {_yaml_string(canonical_url)}
dedup_id: {dedup_id}
classification_source: {article.classification_source}
---
# {article.title}

## 3行要約
- {s1}
- {s2 or "（詳細なし）"}
- {s3}

## 活用メモ
{review}

## AI審査分類
- 対象業種: {", ".join(article.industries)}
- リース物件: {", ".join(article.lease_assets)}
- 信用リスクへの影響: {article.credit_risk_impact}
- 影響方向: {article.impact_direction}
- 情報の信頼度: {article.source_reliability}
- 分類確信度: {article.classification_confidence:.2f}
- 有効期限: {article.valid_until}
- 同一トピック: {article.canonical_topic}

### 審査上の確認事項
{chr(10).join(f"- {item}" for item in article.screening_checks)}

## 詳細
- query: {article.query}
- theme: {article.theme}
- published: {pub}
- link: {article.link}
"""
    return content.strip() + "\n"


def _load_existing_news(vault: Path, news_dir: str) -> list[dict[str, Any]]:
    target_dir = vault / news_dir
    if not target_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        import yaml
    except Exception:
        yaml = None
    for path in target_dir.glob("*.md"):
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            continue
        meta: dict[str, Any] = {}
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if match and yaml:
            try:
                loaded = yaml.safe_load(match.group(1)) or {}
                meta = loaded if isinstance(loaded, dict) else {}
            except Exception:
                meta = {}
        title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        link_match = re.search(r"^-\s+link:\s*(.+)$", raw, re.MULTILINE)
        records.append(
            {
                "path": path,
                "raw": raw,
                "title": title_match.group(1).strip() if title_match else path.stem,
                "canonical_url": _canonical_url(str(meta.get("canonical_url") or (link_match.group(1).strip() if link_match else ""))),
                "canonical_topic": _normalize_text(str(meta.get("canonical_topic") or "")),
            }
        )
    return records


def _find_duplicate(article: Article, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    article_url = _canonical_url(article.link)
    article_title = _normalized_title(article.title, article.source)
    article_topic = _normalize_text(article.canonical_topic).lower()
    for record in records:
        if article_url and record["canonical_url"] and article_url == record["canonical_url"]:
            return record
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for record in records:
        title_score = SequenceMatcher(
            None,
            article_title,
            _normalized_title(str(record["title"])),
        ).ratio()
        record_topic = str(record["canonical_topic"]).lower()
        same_topic = bool(article_topic and record_topic and article_topic == record_topic)
        threshold = 0.84 if same_topic else 0.92
        if title_score >= threshold and title_score > best[0]:
            best = (title_score, record)
    return best[1]


def _merge_related_report(record: dict[str, Any], article: Article) -> bool:
    raw = str(record["raw"])
    canonical_url = _canonical_url(article.link)
    if canonical_url and canonical_url in raw:
        return False
    line = (
        f"- {article.published_iso or dt.date.today().isoformat()} | "
        f"{article.source or '不明'} | [{article.title}]({article.link})"
    )
    heading = "## 関連報道"
    if heading in raw:
        raw = raw.rstrip() + "\n" + line + "\n"
    else:
        raw = raw.rstrip() + f"\n\n{heading}\n{line}\n"
    path = Path(record["path"])
    path.write_text(raw, encoding="utf-8")
    record["raw"] = raw
    return True


def _save_articles_to_obsidian(
    articles: list[Article],
    vault: Path,
    news_dir: str,
    date_str: str,
    profile: str,
) -> list[Path]:
    today_obj = dt.date.fromisoformat(date_str)
    iso_cal = today_obj.isocalendar()
    week = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
    month = today_obj.strftime("%Y-%m")
    target_dir = vault / news_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    existing = _load_existing_news(vault, news_dir)
    for art in articles:
        duplicate = _find_duplicate(art, existing)
        if duplicate:
            if _merge_related_report(duplicate, art):
                saved.append(Path(duplicate["path"]))
            continue
        fname = f"{date_str}_リースニュース_{_safe_filename(art.title)}.md"
        fpath = _safe_note_path(vault, f"{news_dir}/{fname}")
        content = _build_article_content(art, date_str, week, month, profile)
        fpath.write_text(content, encoding="utf-8")
        saved.append(fpath)
        existing.append(
            {
                "path": fpath,
                "raw": content,
                "title": art.title,
                "canonical_url": _canonical_url(art.link),
                "canonical_topic": art.canonical_topic,
            }
        )
    return list(dict.fromkeys(saved))


def _trigger_rag_index(file_paths: list[Path]) -> None:
    try:
        from api.knowledge.obsidian_loader import _chunk_by_h2, _parse_frontmatter
        from api.knowledge.vector_store import get_store
        store = get_store()
        all_chunks = []
        for fpath in file_paths:
            try:
                raw = fpath.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(raw)
                chunks = _chunk_by_h2(body, str(fpath), fpath.name, meta, fpath.stat().st_mtime)
                all_chunks.extend(chunks)
            except Exception:
                pass
        if all_chunks:
            store.upsert_chunks(all_chunks)
            print(f"[rag] indexed {len(all_chunks)} chunks from {len(file_paths)} files")
    except Exception as exc:
        print(f"[rag] index skipped: {exc}", file=sys.stderr)


def _trigger_auto_wikilink(file_paths: list[Path], vault: Path) -> None:
    """保存済みファイルに wikiリンクを自動付与する。"""
    try:
        sys.path.insert(0, str(_SCRIPT_DIR))
        from auto_wikilink import run_on_files
        results = run_on_files(file_paths, vault)
        changed = sum(1 for r in results if r.get("changes", 0) > 0)
        if changed:
            print(f"[wikilink] {changed} files updated with wikilinks")
    except Exception as exc:
        print(f"[wikilink] skipped: {exc}", file=sys.stderr)


def _append_daily_digest(vault: Path, daily_dir: str, news_dir: str, date_str: str, articles: list[Article]) -> Path:
    daily_rel = f"{daily_dir.strip('/')}/{date_str}.md"
    path = _safe_note_path(vault, daily_rel)
    counts = _theme_counts(articles)
    theme_summary = ", ".join(f"{k} {v}件" for k, v in counts[:3]) if counts else "なし"
    body = "\n".join([
        f"## {dt.datetime.now().strftime('%H:%M')} Lease News",
        f"- 収集件数: {len(articles)}",
        f"- 主なテーマ: {theme_summary}",
        f"- 保存先: [[{news_dir}/|リースニュース/]]",
    ])
    _append_text(path, body)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect daily lease-related news into iCloud 上の Obsidian Vault.")
    parser.add_argument("--vault", default=None, help="iCloud 上の Obsidian Vault パス（省略時は自動検出）")
    parser.add_argument("--news-dir", default=DEFAULT_NEWS_DIR, help="Obsidian note directory for news digests.")
    parser.add_argument("--daily-dir", default=DEFAULT_DAILY_DIR, help="Daily note directory to append a short digest.")
    parser.add_argument(
        "--profile",
        default=os.environ.get("LEASE_NEWS_PROFILE", DEFAULT_PROFILE),
        help="Built-in profile name. Defaults to LEASE_NEWS_PROFILE or lease-core.",
    )
    parser.add_argument(
        "--queries",
        default=os.environ.get("LEASE_NEWS_QUERIES", ""),
        help="Comma-separated search queries. Overrides defaults and LEASE_NEWS_QUERIES.",
    )
    parser.add_argument("--limit", type=int, default=18, help="Maximum articles to write.")
    parser.add_argument("--per-query", type=int, default=6, help="Maximum articles to take per query.")
    parser.add_argument("--per-feed", type=int, default=6, help="Maximum articles to take per RSS feed.")
    parser.add_argument(
        "--no-ai-classify",
        action="store_true",
        help="Skip Gemini classification and use deterministic rule classification.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write notes, only print a summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        vault = find_vault(args.vault)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    profile = (args.profile or DEFAULT_PROFILE).strip().lower()
    query_specs, feed_specs = _expand_profile(profile)
    if args.queries.strip():
        query_specs = [("custom", q.strip()) for q in args.queries.split(",") if q.strip()]
        profile = f"{profile}+custom"
    articles = collect_articles(
        query_specs,
        feed_specs,
        per_query=max(1, int(args.per_query)),
        per_feed=max(1, int(args.per_feed)),
    )
    articles = articles[: max(1, int(args.limit))]
    classify_articles(
        articles,
        use_ai=not args.no_ai_classify and os.environ.get("LEASE_NEWS_AI_CLASSIFY", "1") != "0",
    )

    date_str = dt.date.today().isoformat()
    news_dir = args.news_dir.strip("/") or DEFAULT_NEWS_DIR
    daily_dir = args.daily_dir.strip("/") or DEFAULT_DAILY_DIR

    if args.dry_run:
        print(f"DRY-RUN: {len(articles)} articles")
        print(f"target_dir={vault / news_dir}")
        for art in articles[:3]:
            fname = f"{date_str}_リースニュース_{_safe_filename(art.title)}.md"
            print(f"  {fname}")
        return 0

    saved_paths = _save_articles_to_obsidian(articles, vault, news_dir, date_str, profile)

    source_counts = _source_counts(articles)
    tag_counts = _tag_counts(articles)
    source_summary = ", ".join(f"{name} {count}件" for name, count in source_counts[:3]) if source_counts else ""
    tag_summary = ", ".join(f"{name} {count}件" for name, count in tag_counts[:5]) if tag_counts else ""
    first_path = saved_paths[0] if saved_paths else vault / news_dir
    record_lease_news_collection(
        date_str,
        str(first_path.relative_to(vault)) if saved_paths else news_dir,
        len(articles),
        source_summary=source_summary,
        tag_summary=tag_summary,
    )

    daily_path = _append_daily_digest(vault, daily_dir, news_dir, date_str, articles)
    focus_result = write_lease_news_focus_note(date_str=date_str, vault=vault)
    reflection_result = write_lease_news_reflection_note(date_str=date_str, vault=vault, focus=None)
    if reflection_result:
        try:
            from lease_news_digest import get_latest_lease_news_reflection
            from novelist_agent import generate_daily_grumble_illustration

            reflection = get_latest_lease_news_reflection(vault=vault)
            illustration_url = generate_daily_grumble_illustration(
                date_str=date_str,
                lines=list(reflection.thought_lines),
            )
            print(f"reflection_illustration={illustration_url}")
        except Exception as exc:
            print(f"reflection_illustration_error={exc}")
    focus_paths = list(saved_paths)
    if focus_result:
        focus_path = Path(focus_result.note_path)
        focus_paths.append(focus_path)
        print(f"focus_note={focus_path}")
    if reflection_result:
        reflection_path = Path(reflection_result.note_path)
        focus_paths.append(reflection_path)
        print(f"reflection_note={reflection_path}")
    _trigger_auto_wikilink(focus_paths, vault)
    _trigger_rag_index(focus_paths)

    print(f"articles={len(articles)}")
    print(f"saved_files={len(saved_paths)}")
    if focus_result:
        print(f"focus_saved=1")
    if reflection_result:
        print(f"reflection_saved=1")
    print(f"news_dir={vault / news_dir}")
    print(f"daily_note={daily_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
