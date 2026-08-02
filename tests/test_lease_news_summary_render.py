from api.lease_news_summary_render import (
    clean_recent_news_summary_lines,
    normalize_code_list,
    normalize_phrase_list,
    parse_recent_news_note,
    recent_news_dedupe_key,
    render_news_summary,
    render_news_obsidian_note,
    safe_news_filename,
)
import datetime as dt


def test_safe_news_filename_sanitizes_and_falls_back():
    assert safe_news_filename('A/B:C*"D') == "A_B_C__D"
    assert safe_news_filename("   ") == "ニュース"


def test_normalize_lists_dedupe_and_limit():
    assert normalize_code_list(["market", "BAD", "MARKET", "risk"], {"MARKET", "RISK"}, 2) == ["MARKET", "RISK"]
    assert normalize_phrase_list(["設備投資", "設備投資", "x" * 50], limit=3) == ["設備投資", "x" * 40]


def test_render_news_summary_defaults_and_usage():
    rendered = render_news_summary(
        {
            "summary_codes": ["capex", "risk"],
            "usage_codes": ["risk_check"],
            "key_phrases": ["工作機械", "省力化"],
            "tags": [],
        },
        "https://example.com/news",
    )

    assert rendered["title"] == "業界リスクニュース"
    assert rendered["summary_codes"] == ["CAPEX", "RISK"]
    assert "工作機械" in rendered["summary_lines"][0]
    assert rendered["usage_memo"] == "審査時の追加確認項目を洗い出す。"
    assert rendered["tags"] == ["要確認"]
    assert rendered["source_hint"] == "https://example.com/news"


def test_render_news_obsidian_note_builds_frontmatter_and_filename():
    note = render_news_obsidian_note(
        {
            "title": "物流/燃料費",
            "tags": ["物流"],
            "summary_codes": ["RISK"],
            "usage_codes": ["RISK_CHECK"],
            "key_phrases": ["軽油"],
            "summary_lines": ["燃料費が上がる", "運送業の採算を見る", "価格転嫁を見る"],
            "usage_memo": "審査時の追加確認に使う。",
            "region": "国内",
            "importance": "高",
        },
        "https://example.com/news/",
        today=dt.date(2026, 8, 2),
    )

    assert note["date"] == "2026-08-02"
    assert note["week"] == "2026-W31"
    assert note["month"] == "2026-08"
    assert note["filename"] == "2026-08-02_業界リスクニュース_物流_燃料費.md"
    assert "summary_codes: [\"RISK\"]" in note["content"]
    assert "source: https://example.com/news/" in note["content"]
    assert "- 燃料費が上がる" in note["content"]


def test_parse_recent_news_note_extracts_metadata_and_cleans_summary():
    raw = """---
date: 2026-08-02
week: 2026-W31
month: 2026-08
tags: ["物流"]
summary_codes: ["RISK"]
usage_codes: ["RISK_CHECK"]
key_phrases: ["軽油"]
region: 国内
source: https://example.com/news/
importance: 高
---
# 物流燃料費

## 3行要約
- 物流燃料費
- 燃料費上昇で運送業の採算確認が必要
- 燃料費上昇で運送業の採算確認が必要

## 活用メモ
審査時の追加確認に使う。
"""

    item = parse_recent_news_note(raw, file_path="/tmp/news.md", file_stem="news")

    assert item["date"] == "2026-08-02"
    assert item["title"] == "物流燃料費"
    assert item["article_url"] == "https://example.com/news/"
    assert item["summary_lines"] == ["燃料費上昇で運送業の採算確認が必要"]
    assert item["summary_codes"] == ["RISK"]
    assert item["usage_codes"] == ["RISK_CHECK"]
    assert recent_news_dedupe_key(item) == "url:https://example.com/news"


def test_recent_news_cleaning_falls_back_to_text_key_without_url():
    item = {
        "title": "設備投資",
        "summary_lines": ["設備更新の需要がある", "設備更新の需要がある", "（詳細なし）"],
        "usage_memo": "設備更新の需要がある",
        "source": "",
        "article_url": "",
    }

    assert clean_recent_news_summary_lines(item) == []
    item["summary_lines"] = ["省力化投資が増えている"]
    key = recent_news_dedupe_key(item)
    assert key.startswith("text:設備投資|省力化投資が増えている")
