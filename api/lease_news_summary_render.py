"""Pure rendering/parsing helpers for lease-news summaries."""

from __future__ import annotations

import datetime as dt
import json
import re


NEWS_SUMMARY_CODE_TEXT = {
    "CAPEX": "設備投資・更新需要に動きがあり、リース提案の接点になり得ます。",
    "RATE": "金利・資金調達環境の変化が、月額負担や契約条件の説明材料になります。",
    "REGULATION": "制度・規制変更が、顧客の投資判断や導入時期に影響し得ます。",
    "MARKET": "市場環境や需給の変化が、業界別の提案優先度を左右します。",
    "RISK": "信用・資金繰り・事業継続面の確認を強めるべき材料があります。",
    "TECH": "DX・AI・省力化・脱炭素設備など、戦略投資の切り口があります。",
    "ASSET": "対象物件の価値、保全、中古流通を確認する材料があります。",
}

NEWS_USAGE_CODE_TEXT = {
    "PROPOSAL_TIMING": "顧客の投資タイミング確認に使う。",
    "RATE_EXPLAIN": "金利・月額負担・総支払額の説明に使う。",
    "RISK_CHECK": "審査時の追加確認項目を洗い出す。",
    "ASSET_MATCH": "物件選定、残価、保全条件の確認に使う。",
    "INDUSTRY_TALK": "業界動向の会話導入に使う。",
    "FOLLOW_UP": "既存顧客へのフォロー論点にする。",
}


def safe_news_filename(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", text)
    cleaned = cleaned.strip("_").strip()
    return cleaned[:max_len] if cleaned else "ニュース"


def normalize_code_list(values: object, allowed: set[str], limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = []
    for value in values:
        code = str(value).strip().upper()
        if code in allowed and code not in normalized:
            normalized.append(code)
        if len(normalized) >= limit:
            break
    return normalized


def normalize_phrase_list(values: object, limit: int = 5) -> list[str]:
    if not isinstance(values, list):
        return []
    phrases = []
    for value in values:
        phrase = str(value).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase[:40])
        if len(phrases) >= limit:
            break
    return phrases


def render_news_summary(result: dict, source: str) -> dict:
    summary_codes = normalize_code_list(
        result.get("summary_codes"),
        set(NEWS_SUMMARY_CODE_TEXT.keys()),
        3,
    )
    usage_codes = normalize_code_list(
        result.get("usage_codes"),
        set(NEWS_USAGE_CODE_TEXT.keys()),
        2,
    )
    key_phrases = normalize_phrase_list(result.get("key_phrases"), limit=5)

    if not summary_codes:
        summary_codes = ["MARKET"]
    phrase_tail = f"（関連語: {'、'.join(key_phrases[:3])}）" if key_phrases else ""
    lines = []
    for code in summary_codes[:3]:
        line = NEWS_SUMMARY_CODE_TEXT.get(code, NEWS_SUMMARY_CODE_TEXT["MARKET"])
        lines.append(f"{line}{phrase_tail}" if not lines and phrase_tail else line)
    while len(lines) < 3:
        lines.append("原文を確認し、顧客業種・物件・投資時期との関係を見極めます。")

    usage_parts = [
        NEWS_USAGE_CODE_TEXT.get(code, "")
        for code in usage_codes
        if NEWS_USAGE_CODE_TEXT.get(code)
    ]
    if not usage_parts:
        usage_parts = ["顧客との会話導入と、提案・審査時の確認論点整理に使う。"]

    rendered = {
        **result,
        "summary_codes": summary_codes,
        "usage_codes": usage_codes,
        "key_phrases": key_phrases,
        "summary_lines": lines[:3],
        "usage_memo": " ".join(usage_parts),
    }
    if not rendered.get("title"):
        rendered["title"] = "業界リスクニュース"
    if not rendered.get("tags"):
        rendered["tags"] = ["要確認"]
    if not rendered.get("source_hint") and source:
        rendered["source_hint"] = source[:80]
    return rendered


def render_news_obsidian_note(summary: dict, source: str, today: dt.date | None = None) -> dict:
    today_obj = today or dt.date.today()
    date_str = today_obj.isoformat()
    iso_cal = today_obj.isocalendar()
    week = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
    month = today_obj.strftime("%Y-%m")

    title = summary.get("title", "ニュース")
    filename = f"{date_str}_業界リスクニュース_{safe_news_filename(title)}.md"

    tags_yaml = json.dumps(summary.get("tags", []), ensure_ascii=False)
    summary_codes_yaml = json.dumps(summary.get("summary_codes", []), ensure_ascii=False)
    usage_codes_yaml = json.dumps(summary.get("usage_codes", []), ensure_ascii=False)
    key_phrases_yaml = json.dumps(summary.get("key_phrases", []), ensure_ascii=False)
    lines = summary.get("summary_lines", [])
    memo = summary.get("usage_memo", "")
    region = summary.get("region", "国内")

    content = f"""---
date: {date_str}
week: {week}
month: {month}
tags: {tags_yaml}
summary_codes: {summary_codes_yaml}
usage_codes: {usage_codes_yaml}
key_phrases: {key_phrases_yaml}
region: {region}
source: {source or "手動入力"}
importance: {summary.get("importance", "中")}
---
# {title}

## 3行要約
- {lines[0] if len(lines) > 0 else ""}
- {lines[1] if len(lines) > 1 else ""}
- {lines[2] if len(lines) > 2 else ""}

## 活用メモ
{memo}
"""
    return {
        "date": date_str,
        "week": week,
        "month": month,
        "filename": filename,
        "content": content,
    }


def recent_news_dedupe_key(item: dict) -> str:
    url = str(item.get("article_url") or item.get("source") or "").strip().lower()
    if url.startswith(("http://", "https://")):
        return f"url:{url.rstrip('/')}"
    title = re.sub(r"\s+", "", str(item.get("title") or "")).lower()
    first_summary = ""
    for line in item.get("summary_lines") or []:
        first_summary = re.sub(r"\s+", "", str(line or "")).lower()
        if first_summary:
            break
    return f"text:{title}|{first_summary[:80]}"


def clean_recent_news_summary_lines(item: dict) -> list[str]:
    title_compact = re.sub(r"\s+", "", str(item.get("title") or "")).lower()
    title_plain = re.sub(r"[^\w]", "", str(item.get("title") or "").lower())
    memo_compact = re.sub(r"\s+", "", str(item.get("usage_memo") or "")).lower()
    clean_summary_lines: list[str] = []
    seen_summary_lines: set[str] = set()
    for line in item.get("summary_lines") or []:
        text = str(line or "").strip()
        compact = re.sub(r"\s+", "", text).lower()
        plain = re.sub(r"[^\w]", "", text.lower())
        if not text or text == "（詳細なし）":
            continue
        if memo_compact and compact == memo_compact:
            continue
        if title_compact and title_compact in compact:
            continue
        if title_plain and plain and (title_plain in plain or plain in title_plain):
            continue
        if compact in seen_summary_lines:
            continue
        seen_summary_lines.add(compact)
        clean_summary_lines.append(text)
    return clean_summary_lines[:3]


def parse_recent_news_note(raw: str, *, file_path: str, file_stem: str) -> dict:
    item: dict = {
        "date": "",
        "title": file_stem,
        "summary_lines": [],
        "usage_memo": "",
        "summary_codes": [],
        "usage_codes": [],
        "key_phrases": [],
        "tags": [],
        "region": "国内",
        "importance": "通常",
        "source": "",
        "article_url": "",
        "file_path": file_path,
        "week": "",
        "month": "",
    }

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if fm_match:
        fm = fm_match.group(1)
        for line in fm.splitlines():
            if line.startswith("date:"):
                item["date"] = line.split(":", 1)[1].strip()
            elif line.startswith("tags:"):
                try:
                    item["tags"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("summary_codes:"):
                try:
                    item["summary_codes"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("usage_codes:"):
                try:
                    item["usage_codes"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("key_phrases:"):
                try:
                    item["key_phrases"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("region:"):
                item["region"] = line.split(":", 1)[1].strip()
            elif line.startswith("source:"):
                item["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("importance:"):
                item["importance"] = line.split(":", 1)[1].strip()
            elif line.startswith("week:"):
                item["week"] = line.split(":", 1)[1].strip()
            elif line.startswith("month:"):
                item["month"] = line.split(":", 1)[1].strip()

    title_match = re.search(r"^# (.+)$", raw, re.MULTILINE)
    if title_match:
        item["title"] = title_match.group(1).strip()

    summary_section = re.search(r"## 3行要約\s*\n((?:- .+\n?){1,3})", raw)
    if summary_section:
        item["summary_lines"] = [
            line.lstrip("- ").strip()
            for line in summary_section.group(1).strip().splitlines()
            if line.strip()
        ]

    memo_match = re.search(r"## 活用メモ\s*\n(.+?)(?:\n##|\Z)", raw, re.DOTALL)
    if memo_match:
        item["usage_memo"] = memo_match.group(1).strip()

    link_match = re.search(r"^- link:\s*(.+)$", raw, re.MULTILINE)
    if link_match:
        item["article_url"] = link_match.group(1).strip()
    elif item["source"].startswith(("http://", "https://")):
        item["article_url"] = item["source"]

    item["summary_lines"] = clean_recent_news_summary_lines(item)
    return item
