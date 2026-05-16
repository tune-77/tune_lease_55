"""Lightweight web search bridge for AI chat."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests


_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._collect_title = False
        self._collect_snippet = False
        self._title_buf: list[str] = []
        self._snippet_buf: list[str] = []
        self._current: dict[str, str] | None = None

    @staticmethod
    def _class_has(attrs: list[tuple[str, str | None]], target: str) -> bool:
        for key, value in attrs:
            if key == "class" and value and target in value.split():
                return True
        return False

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], target: str) -> str:
        for key, value in attrs:
            if key == target and value:
                return value
        return ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a" and self._class_has(attrs, "result__a"):
            self._collect_title = True
            self._title_buf = []
            href = self._attr(attrs, "href")
            self._current = {
                "url": _unwrap_ddg_url(href),
                "title": "",
                "snippet": "",
            }
            return
        if tag in {"a", "div", "span"} and self._class_has(attrs, "result__snippet"):
            self._collect_snippet = True
            self._snippet_buf = []

    def handle_data(self, data: str) -> None:
        if self._collect_title:
            self._title_buf.append(data)
        elif self._collect_snippet:
            self._snippet_buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._collect_title:
            self._collect_title = False
            if self._current is not None:
                self._current["title"] = _clean_text("".join(self._title_buf))
                if self._current["title"]:
                    self.results.append(self._current)
            self._current = None
            self._title_buf = []
            return
        if tag in {"a", "div", "span"} and self._collect_snippet:
            snippet = _clean_text("".join(self._snippet_buf))
            if snippet:
                if self.results:
                    self.results[-1]["snippet"] = snippet
                elif self._current is not None:
                    self._current["snippet"] = snippet
            self._collect_snippet = False
            self._snippet_buf = []


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _unwrap_ddg_url(href: str) -> str:
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return href


def _search_duckduckgo(query: str, limit: int = 3, timeout: float = 8.0) -> list[dict[str, str]]:
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": _UA},
        timeout=timeout,
    )
    resp.raise_for_status()
    parser = _DDGParser()
    parser.feed(resp.text)
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in parser.results:
        url = item.get("url", "").strip()
        title = item.get("title", "").strip()
        domain = urlparse(url).netloc
        if not url or not title or url in seen:
            continue
        if domain.endswith("duckduckgo.com") or domain.endswith("bing.com"):
            continue
        seen.add(url)
        results.append({
            "title": title,
            "url": url,
            "snippet": item.get("snippet", "").strip(),
            "domain": domain,
        })
        if len(results) >= limit:
            break
    return results


def _fetch_page_summary(url: str, timeout: float = 6.0) -> dict[str, str]:
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    text = resp.text

    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    meta_desc = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        text,
        re.I | re.S,
    )
    paragraph = ""
    for raw in re.findall(r"<p[^>]*>(.*?)</p>", text, re.I | re.S):
        cleaned = _clean_text(raw)
        if len(cleaned) >= 40:
            paragraph = cleaned
            break

    summary = " ".join(part for part in [
        _clean_text(title_match.group(1)) if title_match else "",
        _clean_text(meta_desc.group(1)) if meta_desc else "",
        paragraph,
    ] if part)
    return {
        "page_title": _clean_text(title_match.group(1)) if title_match else "",
        "page_summary": summary[:600],
    }


def collect_web_context(query: str, limit: int = 3) -> list[dict[str, str]]:
    query = _clean_text(query)
    if len(query) < 2:
        return []
    try:
        results = _search_duckduckgo(query, limit=limit)
    except Exception:
        return []

    hits: list[dict[str, str]] = []
    for idx, item in enumerate(results):
        hit = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", ""),
            "domain": item.get("domain", ""),
        }
        if idx == 0:
            try:
                hit.update(_fetch_page_summary(hit["url"]))
            except Exception:
                pass
        hits.append(hit)
    return hits
