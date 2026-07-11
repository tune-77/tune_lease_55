from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_lease_news_to_obsidian.py"
_SPEC = importlib.util.spec_from_file_location("collect_lease_news_to_obsidian", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
news = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = news
_SPEC.loader.exec_module(news)


def _article(**overrides):
    values = {
        "title": "建設会社がAI導入で事務作業を効率化",
        "link": "https://example.com/news/1?utm_source=test",
        "source": "Example News",
        "published": dt.datetime(2026, 6, 6, tzinfo=dt.timezone.utc),
        "summary": "建設会社がAIを導入し、事務作業の削減を進める。",
        "query": "建設業 倒産",
        "theme": "製造・DX",
        "tags": ("建設/不動産", "製造/DX"),
        "score": 2,
    }
    values.update(overrides)
    return news.Article(**values)


def test_rule_classification_populates_searchable_fields():
    article = _article()

    news.classify_articles([article], use_ai=False)

    assert "建設業" in article.industries
    assert "建設機械" in article.lease_assets
    assert article.impact_direction == "positive"
    assert article.source_reliability == "medium"
    assert article.valid_until == "2027-06-06"
    assert article.classification_source == "rule"


def test_article_content_contains_ai_screening_classification():
    article = _article()
    news.classify_articles([article], use_ai=False)

    content = news._build_article_content(
        article,
        date_str="2026-06-06",
        week="2026-W23",
        month="2026-06",
        profile="industry-watch",
    )

    assert 'industries: "建設業, 不動産業, 製造業"' in content
    assert "impact_direction: positive" in content
    assert "valid_until: 2027-06-06" in content
    assert 'canonical_url: "https://example.com/news/1"' in content
    assert "## AI審査分類" in content
    assert "### 審査上の確認事項" in content


def test_duplicate_article_is_merged_into_related_reports(tmp_path):
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"
    first = _article()
    second = _article(
        link="https://another.example.com/story/99",
        source="Second Source",
    )
    news.classify_articles([first, second], use_ai=False)

    saved_first = news._save_articles_to_obsidian(
        [first],
        vault,
        news_dir,
        "2026-06-06",
        "industry-watch",
    )
    saved_second = news._save_articles_to_obsidian(
        [second],
        vault,
        news_dir,
        "2026-06-06",
        "industry-watch",
    )

    files = list((vault / news_dir).glob("*.md"))
    assert len(saved_first) == 1
    assert saved_second == saved_first
    assert len(files) == 1
    merged = files[0].read_text(encoding="utf-8")
    assert "## 関連報道" in merged
    assert "Second Source" in merged
    assert "https://another.example.com/story/99" in merged


def test_exact_duplicate_does_not_append_twice(tmp_path):
    vault = tmp_path / "vault"
    news_dir = "05-クリップ_記事/業界リスクニュース"
    article = _article()
    news.classify_articles([article], use_ai=False)

    news._save_articles_to_obsidian([article], vault, news_dir, "2026-06-06", "industry-watch")
    second_save = news._save_articles_to_obsidian(
        [article],
        vault,
        news_dir,
        "2026-06-06",
        "industry-watch",
    )

    assert second_save == []
    content = next((vault / news_dir).glob("*.md")).read_text(encoding="utf-8")
    assert "## 関連報道" not in content
