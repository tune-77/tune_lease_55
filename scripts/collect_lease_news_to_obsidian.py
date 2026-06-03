#!/usr/bin/env python3
"""Collect daily lease-related news and write a digest into Obsidian.

The collector uses Google News RSS search queries by default, deduplicates
articles, summarizes them into a compact Markdown note, and optionally appends
a short digest to the Daily note.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lease_news_digest import get_lease_news_metrics, record_lease_news_collection


DEFAULT_VAULT_CANDIDATES = [
    Path(os.environ.get("OBSIDIAN_VAULT", "")).expanduser() if os.environ.get("OBSIDIAN_VAULT") else None,
    Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault",
    Path.home() / "Documents" / "Obsidian Vault",
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

DEFAULT_NEWS_DIR = "リースニュース"
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
        raise FileNotFoundError(f"Obsidian vault not found: {path}")
    candidates = _candidate_vaults()
    if not candidates:
        raise FileNotFoundError("Obsidian vault not found. Set OBSIDIAN_VAULT or pass --vault.")
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
source: {article.source or "Google News"}
importance: {importance}
profile: {profile}
---
# {article.title}

## 3行要約
- {s1}
- {s2 or "（詳細なし）"}
- {s3}

## 活用メモ
{review}

## 詳細
- query: {article.query}
- theme: {article.theme}
- published: {pub}
- link: {article.link}
"""
    return content.strip() + "\n"


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
    for art in articles:
        fname = f"{date_str}_リースニュース_{_safe_filename(art.title)}.md"
        fpath = _safe_note_path(vault, f"{news_dir}/{fname}")
        content = _build_article_content(art, date_str, week, month, profile)
        fpath.write_text(content, encoding="utf-8")
        saved.append(fpath)
    return saved


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
    parser = argparse.ArgumentParser(description="Collect daily lease-related news into Obsidian.")
    parser.add_argument("--vault", default=None, help="Source Obsidian vault path. Defaults to auto-detect.")
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
    _trigger_auto_wikilink(saved_paths, vault)
    _trigger_rag_index(saved_paths)

    print(f"articles={len(articles)}")
    print(f"saved_files={len(saved_paths)}")
    print(f"news_dir={vault / news_dir}")
    print(f"daily_note={daily_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
